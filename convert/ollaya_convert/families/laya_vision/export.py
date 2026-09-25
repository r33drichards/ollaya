"""Export Laya Vision (SmolVLM vision tower + connector + 20-layer text model + Laya head) to one ONNX graph.

    uv run --with pillow --with 'torchvision==0.29.*' \
        python -m ollaya_convert.families.laya_vision.export --out out/laya-vision

Contract V ("vision markers"), one row per question (dynamic along rows `n`, sequence `s`, markers `k` and
images `i`):
    input_ids      int64   [n, s]        right-padded; every row carries the request's full image run
    attention_mask int64   [n, s]
    option_span    int64   [n, 2]        [first option token, end of the row): bidirectional inside
    marker_pos     int64   [n, k]        the `\\n` terminating each option's line
    marker_mask    bool    [n, k]
    qtype          int64   [n]           choice 0, score 1, noul 2 (the head's type embedding)
    pixel_values   float32 [i, 3, 512, 512]  (uint8 / 255 - 0.5) / 0.5, the request's images in state order;
                                         a request without images feeds one all-zero image that no row reads
Outputs:
    logits         float32 [n, k]        raw option scores; masked slots are -1e4
    act_logits     float32 [n, 2]        act/escalate head

The graph is `VLMDecisionModel.forward` with the two data-dependent steps of the Hugging Face forward
replaced by fixed-shape equivalents, as laya-vision's own `laya/static_step.py` does:
  * vision position ids: an unsplit 512-pixel tile always gets the same ones (recorded once, checked);
  * the merge of image features into the text embeddings (`masked_scatter`): the k-th `<image>` token of a
    row takes the k-th image-feature vector, a cumsum + gather + where. Every row holds every image, in
    order, so rows share the images' features: the vision tower runs once per image, not once per row.
Temperature and softmax stay in the runtime (docs/families/laya-vision.md).
"""
import argparse
import json
import os
import shutil
import sys

import onnx
import torch

from . import cases as vcases
from . import ref

INPUT_NAMES = ["input_ids", "attention_mask", "option_span", "marker_pos", "marker_mask", "qtype", "pixel_values"]
OUTPUT_NAMES = ["logits", "act_logits"]
OPSET = 20
# The act head's features take topk(2) of the option softmax, so the marker axis is padded to 2 slots.
MIN_MARKERS = 2
MAX_IMAGES = 16


class Graph(torch.nn.Module):
    """VLMDecisionModel.forward on precomputed pixels, fixed-shape. Parameter names are `model.<checkpoint key>`."""

    def __init__(self, model, vision_pos: torch.Tensor, image_token_id: int):
        super().__init__()
        self.model = model
        self.register_buffer("vision_pos", vision_pos, persistent=False)
        self.image_token_id = image_token_id
        self.option_block_mask = sys.modules["laya_vision.vlm"].option_block_mask

    def forward(self, input_ids, attention_mask, option_span, marker_pos, marker_mask, qtype, pixel_values):
        enc = self.model.encoder
        vm = enc.vision_model
        x = vm.embeddings.patch_embedding(pixel_values.to(vm.embeddings.patch_embedding.weight.dtype))
        x = x.flatten(2).transpose(1, 2) + vm.embeddings.position_embedding(self.vision_pos)
        x = vm.post_layernorm(vm.encoder(inputs_embeds=x).last_hidden_state)
        feats = enc.connector(x)  # [i, 64, d]
        flat = feats.reshape(-1, feats.size(-1))

        emb = enc.get_input_embeddings()(input_ids)
        is_img = input_ids == self.image_token_id
        idx = (torch.cumsum(is_img.to(torch.int64), 1) - 1).clamp(min=0)
        idx = torch.minimum(idx, torch.full_like(idx, flat.size(0) - 1))
        emb = torch.where(is_img[..., None], flat[idx].to(emb.dtype), emb)

        mask = self.option_block_mask(attention_mask, option_span, emb.dtype)
        pos = torch.arange(input_ids.size(1), device=input_ids.device)[None].expand(input_ids.size(0), -1)
        h = enc.text_model(inputs_embeds=emb, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
        logits, act = self.model._readout(h, attention_mask, marker_pos, marker_mask, qtype)
        return logits.float(), act.float()


def vision_positions(agent) -> torch.Tensor:
    """The vision tower's position ids for one full, unpadded tile, as its own forward computes them."""
    enc = agent.model.encoder
    got = {}
    S = agent.prep.image_size
    emb = enc.vision_model.embeddings.position_embedding
    hook = emb.register_forward_hook(lambda m, i, o: got.update(pos=i[0][:1].clone()))
    try:
        with torch.no_grad():
            enc.get_image_features(torch.ones((1, 1, 3, S, S)), torch.ones((1, 1, S, S), dtype=torch.bool))
    finally:
        hook.remove()
    pos = got["pos"]
    n = (S // enc.config.vision_config.patch_size) ** 2
    if not torch.equal(pos, torch.arange(n)[None]):
        raise RuntimeError("unexpected vision position ids for a full tile")
    return pos


def sample_inputs(agent):
    """A real request with two images and rows of different lengths and marker counts."""
    _, state, qs = next(c for c in vcases.cases() if c[0] == "lv/two_images")
    qs = dict(qs, clutter=vcases.CLUTTER_Q)
    enc = ref.encode(agent, vcases.materialize(state), qs)
    b = ref.collate(enc, agent.processor.tokenizer.pad_token_id, MIN_MARKERS)
    return tuple(torch.from_numpy(b[n]) for n in INPUT_NAMES)


def export(out_dir: str) -> str:
    # nn.TransformerEncoderLayer's fused fast path has no ONNX lowering; the decomposed path is the same module.
    torch.backends.mha.set_fastpath_enabled(False)
    agent = ref.load("cpu")
    tok = agent.processor.tokenizer
    image_token_id = agent.model.encoder.config.image_token_id
    graph = Graph(agent.model, vision_positions(agent), image_token_id).eval()
    args = sample_inputs(agent)

    n = torch.export.Dim("n", min=1, max=1024)
    s = torch.export.Dim("s", min=8, max=agent.cfg["max_len"])
    k = torch.export.Dim("k", min=2, max=255)
    i = torch.export.Dim("i", min=1, max=MAX_IMAGES)
    dynamic_shapes = {
        "input_ids": {0: n, 1: s},
        "attention_mask": {0: n, 1: s},
        "option_span": {0: n},
        "marker_pos": {0: n, 1: k},
        "marker_mask": {0: n, 1: k},
        "qtype": {0: n},
        "pixel_values": {0: i},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.onnx")
    with torch.no_grad():
        program = torch.onnx.export(
            graph, args, dynamo=True, opset_version=OPSET,
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            dynamic_shapes=dynamic_shapes, optimize=True,
        )
    program.save(path, external_data=True)
    onnx.checker.check_model(path, full_check=True)
    got = [x.name for x in onnx.load(path, load_external_data=False).graph.input]
    if got != INPUT_NAMES:
        raise RuntimeError("graph inputs %s != %s" % (got, INPUT_NAMES))

    # The tokenizer ships unchanged from the checkpoint; parity.py proves the Rust `tokenizers` core
    # (Tokenizer.from_file) reproduces the Python tokenizer on every piece the layout encodes.
    snap = ref.snapshot_dir()
    shutil.copy(os.path.join(snap, "processor", "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))
    vlm = sys.modules["laya_vision.vlm"]
    proc = agent.processor
    ip = proc.image_processor
    decision = {
        "engine": "onnx",
        "family": "laya-vision",
        "layout": "laya-vision-terminator-v1",
        "contract": "vision-markers",
        "source": {"repo": ref.REPO, "revision": ref.REVISION, "reference": ref.UPSTREAM,
                   "code": "https://github.com/%s/tree/%s" % (ref.SOURCE_REPO, ref.SOURCE_COMMIT),
                   "license": "Apache-2.0", "author": "thaitea (Laya Vision), a fork of Laya by Convai Innovations"},
        "backbone": agent.cfg["backbone"],
        "max_len": agent.cfg["max_len"],
        "head_max_len": agent.cfg["head_max_len"],
        "option_max_tokens": 48,
        "prompt_format_version": vlm.PROMPT_FORMAT_VERSION,
        "text": {
            "prefix": vlm.PREFIX_TEXT,
            "question": vlm.QUESTION_TEXT,
            "option_bullet": vlm.OPTION_BULLET,
            "option_end": vlm.OPTION_END,
            "end_of_utterance": "<end_of_utterance>",
            "add_special_tokens": False,
        },
        "image": {
            "state_keys": ["image", "images"],
            "size": agent.prep.image_size,
            "stage1_longest_edge": 2048,
            "resample": "lanczos3-antialias (torchvision, uint8 rounded and clamped after each hop)",
            "processor_class": type(ip).__name__,
            "rescale_factor": float(ip.rescale_factor),
            "image_mean": [float(v) for v in ip.image_mean],
            "image_std": [float(v) for v in ip.image_std],
            "image_seq_len": agent.prep.image_seq_len,
            "tokens": {"fake": proc.fake_image_token, "global": getattr(proc, "global_image_tag", None)
                       or proc.global_image_token, "image": proc.image_token},
            "image_token_id": image_token_id,
            "max_images": MAX_IMAGES,
        },
        "special_tokens": {"pad": tok.pad_token_id, "option_end": tok(vlm.OPTION_END, add_special_tokens=False)["input_ids"][0]},
        "inputs": INPUT_NAMES,
        "outputs": OUTPUT_NAMES,
        "min_markers": MIN_MARKERS,
        "opset": OPSET,
    }
    calibration = {
        "temperature": [float(t) for t in agent.cfg["temperature"]],
        "temperature_by_options": dict(agent.cfg.get("temperature_by_options") or {}),
    }
    with open(os.path.join(out_dir, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/laya-vision")
    a = ap.parse_args()
    path = export(a.out)
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.startswith("model.onnx"))
    print("wrote %s (%.0f MB with external data)" % (path, size / 2**20))


if __name__ == "__main__":
    main()
