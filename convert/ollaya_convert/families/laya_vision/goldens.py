"""Golden fixtures for the Rust port of `laya-vision-terminator-v1`.

    uv run --with pillow --with 'torchvision==0.29.*' \
        python -m ollaya_convert.families.laya_vision.goldens --out out/goldens-laya-vision.jsonl \
        [--pixels-dir out/goldens-laya-vision-pixels]

CPU is enough (about 2 min for the case set).

One JSON line per case (requests are JSON round-tripped first, as they arrive over HTTP; images are `data:`
URLs under the state keys "image" / "images"):
    {"id", "state", "questions",
     "options",                     # native request options: {} or {"resize": false}
     "state_text",                  # the state's text once its images are removed (the state piece of a row)
     "images": [{"shape", "sha256", "mean"}],   # per image, in state order: the 512x512x3 uint8 pixels the
                                    # processor produced (before rescale/normalise), for the Rust preprocessing
     "items": [{"qid", "qtype", "labels",
                "ids", "markers", "option_span",   # the row, as upstream's build_vlm_inputs builds it
                "truncation",                      # upstream's report of what was cut, or null
                "logits", "act_logits",            # the network, backbone in float64 (`ref.Exact`), one
                                                   # unpadded row at a time
                "temperature", "probabilities"}],  # calibrated, label order ([false, true] for noul)
     "answers"}                     # upstream's own VLMAgent.predict (fp32), for the answer rendering; null
                                    # with resize off (upstream always resizes)
Then one line per request the runtime must refuse: {"id", "state", "questions", "options", "error": {"code",
"detail": [validation issue]}} (resize off with an image that is not 512x512).

`--pixels-dir` also writes each image's 512x512 uint8 pixels as `<case>-<n>.npy`, for debugging the resampler.
"""
import argparse
import json
import os

import numpy as np
import torch

from . import cases as vcases
from . import ref


def wire(x):
    return json.loads(json.dumps(x, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/goldens-laya-vision.jsonl")
    ap.add_argument("--pixels-dir", default=None)
    ap.add_argument("--precision", choices=["fp64", "fp32"], default="fp64",
                    help="reference numbers: the backbone in float64 (default), or upstream's fp32 forward")
    a = ap.parse_args()

    torch.backends.mha.set_fastpath_enabled(False)
    agent = ref.load("cpu")
    exact = ref.Exact(agent) if a.precision == "fp64" else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.pixels_dir:
        os.makedirs(a.pixels_dir, exist_ok=True)
    n = 0
    with open(a.out, "w") as f:
        for cid, state, questions, opts in vcases.cases():
            state, questions = wire(state), wire(questions)
            resize = opts.get("resize", True)
            enc = ref.encode(agent, vcases.materialize(state), questions, resize=resize)
            logits, act = ref.forward(agent, enc, exact)
            items = []
            for it, lg, ac in zip(enc["items"], logits, act):
                k = len(it["labels"])
                items.append({
                    "qid": it["qid"], "qtype": it["qtype"], "labels": it["labels"],
                    "ids": it["ids"], "markers": it["markers"], "option_span": it["option_span"],
                    "truncation": it["truncation"],
                    "logits": [float(x) for x in lg[:k]],
                    "act_logits": [float(x) for x in ac],
                    "temperature": ref.temperature(agent, it["qtype"], k),
                    "probabilities": [float(x) for x in ref.probabilities(agent, it, lg)],
                })
            if a.pixels_dir and enc["pixel_values"] is not None:
                for i in range(enc["pixel_values"].shape[0]):
                    np.save(os.path.join(a.pixels_dir, "%s-%d.npy" % (cid.replace("/", "_"), i)),
                            ref.pixels_uint8(enc["pixel_values"][i]))
            # upstream's predict always resizes; for resize-off requests there is no upstream answer to record
            answers = agent.predict(vcases.materialize(state), questions)["answers"] if resize else None
            rec = {"id": cid, "state": state, "questions": questions, "options": opts, "state_text": enc["state_text"],
                   "images": enc["images"], "items": items, "answers": answers}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print("%3d %s" % (n, cid), flush=True)
        for cid, state, questions, opts, issue in vcases.rejected():
            f.write(json.dumps({"id": cid, "state": wire(state), "questions": wire(questions), "options": opts,
                                "error": {"code": "INVALID_REQUEST", "detail": [issue]}}, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d cases to %s (%.1f MB)" % (n, a.out, os.path.getsize(a.out) / 2**20))


if __name__ == "__main__":
    main()
