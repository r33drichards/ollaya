"""Check the Laya Vision ONNX export against the PyTorch reference on the case set.

    uv run --with pillow --with 'torchvision==0.29.*' \
        python -m ollaya_convert.families.laya_vision.parity out/laya-vision

Three checks per question:
  1. layout     `layout.encode_row` driven by the Rust `tokenizers` core (`Tokenizer.from_file(<dir>/tokenizer.json)`,
                no special tokens) reproduces upstream's row ids, markers and option span.
  2. export     ONNX (onnxruntime, one padded batch per request with the request's images) against the
                upstream network run one unpadded row at a time in float64 (`ref.Exact`; `--precision fp32`
                for upstream's own fp32 forward): logits, act logits, calibrated probabilities, argmax.
  3. upstream   the reference's calibrated probabilities against `VLMAgent.predict` (rounded to 4 dp).
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer

from . import cases as vcases
from . import layout, ref
from .export import INPUT_NAMES, MIN_MARKERS


def wire(x):
    """The request as it arrives over HTTP (JSON has only string keys)."""
    return json.loads(json.dumps(x, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--precision", choices=["fp64", "fp32"], default="fp64")
    ap.add_argument("--no-upstream", action="store_true")
    a = ap.parse_args()

    torch.backends.mha.set_fastpath_enabled(False)
    agent = ref.load("cpu")
    exact = ref.Exact(agent) if a.precision == "fp64" else None
    pad = agent.processor.tokenizer.pad_token_id
    rust_tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    print("tokenizer.json truncation=%s padding=%s -> disabled" % (rust_tok.truncation, rust_tok.padding))
    rust_tok.no_truncation()
    rust_tok.no_padding()
    encode = lambda s: rust_tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(a.model_dir, "model.onnx"), so, providers=[a.provider])

    stats = defaultdict(list)
    worst = []
    n_q = agree = layout_ok = 0
    up_n = up_agree = 0
    t_ort = 0.0
    for cid, state, questions in vcases.cases():
        state, questions = wire(state), wire(questions)
        enc = ref.encode(agent, vcases.materialize(state), questions)
        for it in enc["items"]:
            mine = layout.encode_row(encode, state, questions[it["qid"]], agent.cfg["max_len"], agent.cfg["head_max_len"])
            ok = (mine["ids"] == it["ids"] and mine["markers"] == it["markers"]
                  and mine["option_span"] == it["option_span"])
            layout_ok += int(ok)
            if not ok:
                print("  LAYOUT DIFFERS %s %s" % (cid, it["qid"]))
        ref_l, ref_a = ref.forward(agent, enc, exact)
        b = ref.collate(enc, pad, MIN_MARKERS)
        t0 = time.perf_counter()
        got_l, got_a = sess.run(None, {n: b[n] for n in INPUT_NAMES})
        t_ort += time.perf_counter() - t0
        masked = got_l[~b["marker_mask"]]
        if masked.size and not np.all(masked == -1e4):
            raise AssertionError("masked slots must be -1e4")
        answers = None if a.no_upstream else agent.predict(vcases.materialize(state), questions)["answers"]
        for r, it in enumerate(enc["items"]):
            k = len(it["markers"])
            stats["logit"].append(float(np.abs(got_l[r, :k] - ref_l[r][:k]).max()))
            stats["act_logit"].append(float(np.abs(got_a[r] - ref_a[r]).max()))
            p_ref = ref.probabilities(agent, it, ref_l[r])
            p_got = ref.probabilities(agent, it, got_l[r, :k])
            d = float(np.abs(p_ref - p_got).max())
            same = int(p_ref.argmax() == p_got.argmax())
            n_q += 1
            agree += same
            stats["prob/%d" % it["qtype"]].append(d)
            worst.append((d, cid, it["qid"], same))
            if answers is not None:
                ans = answers[it["qid"]]
                if "probabilities" in ans:
                    p_up = np.array([ans["probabilities"][lab] for lab in it["labels"]])
                else:
                    p_up = np.array([1 - ans["noul"], ans["noul"]])
                up_n += 1
                up_agree += int(p_up.argmax() == p_ref.argmax() or np.abs(p_up - p_ref).max() < 1e-4)
                stats["upstream_prob"].append(float(np.abs(p_up - p_ref).max()))

    n_rows = n_q
    print("questions: %d   ort time: %.1fs (%s)" % (n_q, t_ort, a.provider))
    print("layout.py + tokenizer.json (Rust core) reproduce upstream rows: %d/%d" % (layout_ok, n_rows))
    print("argmax agreement  onnx vs %s reference: %.4f" % (a.precision, agree / n_q))
    if up_n:
        print("reference vs VLMAgent.predict (4 dp): %d questions, argmax agreement %.4f" % (up_n, up_agree / up_n))
    for key in sorted(stats):
        v = np.array(stats[key])
        print("  %-14s max %.2e   p99 %.2e   mean %.2e" % (key, v.max(), np.quantile(v, 0.99), v.mean()))
    print("worst probability deltas (onnx vs reference):")
    for d, cid, qid, same in sorted(worst, reverse=True)[:6]:
        print("  %.2e  %s  %s%s" % (d, cid, qid, "" if same else "  ARGMAX DIFFERS"))


if __name__ == "__main__":
    main()
