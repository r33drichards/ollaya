"""The PyTorch reference for Laya Vision: load `thaitea/laya-vision` with its own code, pinned.

    uv run --with pillow --with 'torchvision==0.29.*' python -m ollaya_convert.families.laya_vision.ref   # smoke test

Upstream (Apache-2.0):
  * code:    github.com/r33drichards/laya-vision @ `SOURCE_COMMIT` (package `laya`, version 0.2.0.dev0).
  * weights: huggingface.co/thaitea/laya-vision @ `REVISION`: SmolVLM-256M cut to 20 of 30 language layers,
             options attend to each other (`option_attention: "block"`), `readout: "terminator"`.

The fork's package is also called `laya`, which clashes with the text-only `laya==0.3.7` this project pins.
So the pinned source tree is fetched once into `convert/out/src/` and imported under the name `laya_vision`
(its modules only use relative imports on the paths used here). `LAYA_VISION_SRC=<checkout>` points at a
local checkout instead; it must be at `SOURCE_COMMIT`.

Delegated to upstream code, so it cannot drift:
  * state split and rendering   laya.vlm.split_state (keys "image"/"images" are images, the rest JSON text)
  * question conversion         VLMAgent._to_internal, laya.common.render_options
  * image preprocessing         the checkpoint's Idefics3 processor (PIL LANCZOS to longest edge 2048, then
                                to 512x512, then (v/255 - 0.5) / 0.5), via laya.vlm.vlm_prefix
  * sequence ids                laya.vlm.build_vlm_inputs (per row: image run, state, question, options)
  * the network                 VLMDecisionModel.encode_images + forward, as VLMAgent.predict calls them
Re-stated here: the temperature pick and softmax (VLMAgent._checkpoint_temperature, predict).

Ollaya uses one option order per question (`predict(n_permutations=1)`, the default).
"""
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO = "thaitea/laya-vision"
REVISION = "8b318c99d7ad3ce19c24369263463882eada9d1e"
SOURCE_REPO = "r33drichards/laya-vision"
SOURCE_COMMIT = "568feeeada793f70f736756b0f3a7643d1e75910"
UPSTREAM = "laya-vision @ %s" % SOURCE_COMMIT[:7]

SRC_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "out", "src")
QTYPES = {"choice": 0, "score": 1, "noul": 2}


# --------------------------------------------------------------------------------------------- source


def _fetch_source() -> str:
    """The pinned laya-vision tree (its `laya/` package directory), downloaded once."""
    local = os.environ.get("LAYA_VISION_SRC")
    if local:
        head = os.path.join(local, ".git")
        if os.path.isdir(head):
            import subprocess

            got = subprocess.run(["git", "-C", local, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            if got != SOURCE_COMMIT:
                raise RuntimeError("LAYA_VISION_SRC is at %s, expected %s" % (got, SOURCE_COMMIT))
        return os.path.join(local, "laya")
    root = os.path.abspath(os.path.join(SRC_CACHE, "laya-vision-%s" % SOURCE_COMMIT))
    pkg = os.path.join(root, "laya")
    if not os.path.isdir(pkg):
        url = "https://codeload.github.com/%s/tar.gz/%s" % (SOURCE_REPO, SOURCE_COMMIT)
        data = urllib.request.urlopen(url).read()
        os.makedirs(root, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for m in tf.getmembers():
                parts = m.name.split("/", 1)
                if len(parts) == 2 and parts[1].startswith("laya/") and (m.isfile() or m.isdir()):
                    m.name = parts[1]
                    tf.extract(m, root, filter="data")
    return pkg


def upstream():
    """The pinned laya-vision package, imported as `laya_vision`."""
    if "laya_vision" in sys.modules:
        return sys.modules["laya_vision"]
    pkg = _fetch_source()
    spec = importlib.util.spec_from_file_location("laya_vision", os.path.join(pkg, "__init__.py"),
                                                  submodule_search_locations=[pkg])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["laya_vision"] = mod
    spec.loader.exec_module(mod)
    return mod


def load(device: str = "cpu"):
    """VLMAgent in fp32 on `device`, from the pinned snapshot."""
    from huggingface_hub import snapshot_download

    lv = upstream()
    snap = snapshot_download(REPO, revision=REVISION)
    agent = lv.load_vlm(snap, device=device)
    agent.model.eval()
    return agent


def snapshot_dir() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, revision=REVISION)


# --------------------------------------------------------------------------------------------- images


def pixels_uint8(pv: torch.Tensor) -> np.ndarray:
    """[3, H, W] normalised pixels -> the uint8 [H, W, 3] image the processor normalised.

    The processor resizes in uint8 and then applies (v/255 - 0.5)/0.5 in float32, so the map is invertible;
    the round trip is checked, so the hash below is exactly what the Rust preprocessing must reproduce.
    """
    x = pv.detach().to(torch.float32).cpu().numpy()
    u = np.rint((x * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    back = ((u.astype(np.float32) / np.float32(255.0)) - np.float32(0.5)) / np.float32(0.5)
    if not np.allclose(back, x, atol=1e-6, rtol=0):
        raise AssertionError("pixel_values are not uint8 pixels normalised with mean/std 0.5")
    return np.ascontiguousarray(u.transpose(1, 2, 0))


def image_digest(pv: torch.Tensor) -> Dict[str, Any]:
    u = pixels_uint8(pv)
    return {"shape": list(u.shape), "sha256": hashlib.sha256(u.tobytes()).hexdigest(),
            "mean": [round(float(u[..., c].mean()), 4) for c in range(3)]}


# --------------------------------------------------------------------------------------------- rows


def encode(agent, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The rows `VLMAgent.predict` builds (one option order), plus the request's pixels.

    Returns {"items": [{qid, qtype, labels, ids, markers, option_span, truncation}], "pixel_values": [I,3,S,S]
    or None, "images": [digest per image], "state_text"}.
    """
    upstream()
    vlm = sys.modules["laya_vision.vlm"]
    common = sys.modules["laya_vision.common"]
    images, text = vlm.split_state(state)
    prefix = vlm.vlm_prefix(agent.processor, images, agent.prep)
    max_len, head_max_len = agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256)
    items: List[Dict[str, Any]] = []
    for qid, qdef in questions.items():
        q = agent._to_internal(qdef)
        k = len(common.render_options(q))
        it = vlm.build_vlm_inputs(agent.processor, state, q, max_len, head_max_len, prefix=prefix)
        if len(it["markers"]) != k:
            raise ValueError("question %r options exceed head_max_len" % qid)
        items.append({
            "qid": qid, "qtype": QTYPES[q["t"]], "labels": common.option_labels(q),
            "ids": list(it["ids"]), "markers": list(it["markers"]), "option_span": list(it["option_span"]),
            "truncation": common.truncation_answer(it["truncation"], q) or None,
        })
    pv = prefix["pixel_values"]
    return {
        "items": items,
        "pixel_values": pv,
        "images": [image_digest(pv[i]) for i in range(pv.shape[0])] if pv is not None else [],
        "state_text": text,
    }


def collate(enc: Dict[str, Any], pad_id: int, min_markers: int, image_size: int = 512) -> Dict[str, np.ndarray]:
    """Right-padded numpy batch in the graph's contract (see export.py).

    `pixel_values` always has at least one image: a request without images gets one all-zero image that no
    row reads (no row has an `<image>` token).
    """
    items = enc["items"]
    n = len(items)
    s = max(len(it["ids"]) for it in items)
    k = max(min_markers, max(len(it["markers"]) for it in items))
    b = {
        "input_ids": np.full((n, s), pad_id, dtype=np.int64),
        "attention_mask": np.zeros((n, s), dtype=np.int64),
        "option_span": np.zeros((n, 2), dtype=np.int64),
        "marker_pos": np.zeros((n, k), dtype=np.int64),
        "marker_mask": np.zeros((n, k), dtype=bool),
        "qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
    }
    for r, it in enumerate(items):
        L, m = len(it["ids"]), len(it["markers"])
        b["input_ids"][r, :L] = it["ids"]
        b["attention_mask"][r, :L] = 1
        b["option_span"][r] = it["option_span"]
        b["marker_pos"][r, :m] = it["markers"]
        b["marker_mask"][r, :m] = True
    pv = enc["pixel_values"]
    b["pixel_values"] = (pv.to(torch.float32).numpy() if pv is not None
                         else np.zeros((1, 3, image_size, image_size), dtype=np.float32))
    return b


# --------------------------------------------------------------------------------------------- network


class Exact:
    """The same network with its backbone (vision tower, connector, text model) in float64, a deep copy: the
    numbers the goldens and export checks compare against.

    The decision head stays in float32 because upstream pins it there: `_readout` starts with `h.float()`, so
    the head computes in fp32 whatever the backbone's dtype (bf16 in training, fp32 on CPU). Rounding the
    fp64 hidden states to fp32 there is upstream's own arithmetic, not an approximation of it.
    """

    def __init__(self, agent, device: str = "cpu"):
        self.model = copy.deepcopy(agent.model).to(device).eval()
        self.model.encoder.to(torch.float64)
        self.device = device


@torch.no_grad()
def forward(agent, enc: Dict[str, Any], exact: Optional[Exact] = None):
    """Raw (logits [n, k], act_logits [n, 2]) per row, as `VLMAgent.predict` computes them (full path, no
    prefix cache): images encoded once, their features repeated for every row. Rows run one at a time,
    unpadded, so padding cannot leak into the reference."""
    model = exact.model if exact is not None else agent.model
    dev = torch.device(exact.device) if exact is not None else agent.device
    dtype = torch.float64 if exact is not None else torch.float32
    feats = None
    if enc["pixel_values"] is not None:
        pv = enc["pixel_values"].to(dev, dtype)
        feats = model.encode_images(pv, torch.ones(pv.shape[0], pv.shape[2], pv.shape[3], dtype=torch.bool, device=dev))
    out_l, out_a = [], []
    for it in enc["items"]:
        ids = torch.tensor([it["ids"]], device=dev)
        am = torch.ones_like(ids)
        mp = torch.tensor([it["markers"]], device=dev)
        mm = torch.ones_like(mp, dtype=torch.bool)
        qt = torch.tensor([it["qtype"]], device=dev)
        span = torch.tensor([it["option_span"]], device=dev)
        lg, act = model(ids, am, mp, mm, qt, image_hidden_states=feats, option_span=span)
        out_l.append(lg[0].double().cpu().numpy())
        out_a.append(act[0].double().cpu().numpy())
    return out_l, out_a


def temperature(agent, qtype: int, k: int) -> float:
    """VLMAgent._checkpoint_temperature: the per-option-count bucket first, else the per-type value."""
    return float(agent._checkpoint_temperature(qtype, k))


def probabilities(agent, it: Dict[str, Any], logits: np.ndarray) -> np.ndarray:
    k = len(it["labels"])
    z = np.asarray(logits[:k], dtype=np.float64) / max(1e-3, temperature(agent, it["qtype"], k))
    p = np.exp(z - z.max())
    return p / p.sum()


def main():
    from . import cases as vcases

    agent = load("cpu")
    cid, state, qs = vcases.cases()[0]
    enc = encode(agent, vcases.materialize(state), qs)
    lg, act = forward(agent, enc)
    ans = agent.predict(vcases.materialize(state), qs)["answers"]
    for it, row in zip(enc["items"], lg):
        p = probabilities(agent, it, row)
        print(it["qid"], len(it["ids"]), dict(zip(it["labels"], np.round(p, 4))), ans[it["qid"]])
    print(json.dumps(enc["images"]))


if __name__ == "__main__":
    main()
