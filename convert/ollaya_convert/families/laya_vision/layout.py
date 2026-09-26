"""`laya-vision-terminator-v1`: the row layout, written out independently of upstream as the Rust port's spec.

`parity.py` runs this with the Rust `tokenizers` core (Tokenizer.from_file, no special tokens) and checks
every row against upstream's `build_vlm_inputs`, so this file and docs/families/laya-vision.md are what
`ollaya_decision::laya_vision` implements. Only the tokenizer is shared with upstream.

Row = prefix + state + tail, each piece tokenized on its own and concatenated:
    prefix  "<|im_start|>User:" + ("<fake_token_around_image><global-img>" + "<image>" * 64
            + "<fake_token_around_image>") per image, tokenized as one string
    state   the state's text (images removed), cut from the right to what max_len leaves
    tail    "\\n{type} question: {instructions}<end_of_utterance>\\nAssistant: Options:\\n"
            then per option: "- " + option text (inner "\\n" -> " "), cut to 48 tokens, then the "\\n" token
Markers are the positions of the options' "\\n" tokens; the option span runs from the first option token to the
end of the row.
"""
import json
from typing import Any, Callable, Dict, List, Tuple

PREFIX = "<|im_start|>User:"
QUESTION = "\n%s question: %s<end_of_utterance>\nAssistant: Options:\n"
BULLET = "- "
END = "\n"
EOU = "<end_of_utterance>"
IMAGE_RUN = "<fake_token_around_image><global-img>" + "<image>" * 64 + "<fake_token_around_image>"
OPTION_MAX = 48
NOUL_FALSE = "no, the statement does not hold"
NOUL_TRUE = "yes, the statement holds"


def split_state(state: Any) -> Tuple[int, str]:
    """(number of images, state text). Only a JSON object can carry images, under "image" and "images"."""
    if isinstance(state, str):
        return 0, state
    if not isinstance(state, dict):
        return 0, json.dumps(state, ensure_ascii=False)
    n = int(state.get("image") is not None) + len(state.get("images") or [])
    rest = {k: v for k, v in state.items() if k not in ("image", "images")}
    return n, json.dumps(rest, ensure_ascii=False) if rest else ""


def options(qdef: Dict[str, Any]) -> List[str]:
    t, crit = qdef["type"], qdef.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            return list(crit)
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or NOUL_FALSE), "true: " + (crit.get("true") or NOUL_TRUE)]


def encode_row(encode: Callable[[str], List[int]], state: Any, qdef: Dict[str, Any], max_len: int = 1024,
               head_max_len: int = 256) -> Dict[str, Any]:
    n_images, text = split_state(state)
    prefix = encode(PREFIX + IMAGE_RUN * n_images)
    (end_id,) = encode(END)
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)  # ensure_ascii=True, as upstream's _to_internal
    opt_ids = [encode(BULLET + o.replace(END, " "))[:OPTION_MAX] for o in options(qdef)]
    head = encode(QUESTION % (qdef["type"], ins.replace(EOU, " ")))
    budget = head_max_len - sum(len(o) + 1 for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)) - 1)
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) + 1 for o in opt_ids)
    if len(head) > max(8, budget):
        keep = max(8, budget)
        head = head[: keep // 2] + head[-(keep - keep // 2):]
    tail, markers = list(head), []
    for o in opt_ids:
        tail += o + [end_id]
        markers.append(len(tail) - 1)
    if len(prefix) + len(tail) > max_len:
        raise ValueError("question + options + images exceed max_len=%d" % max_len)
    room = max_len - len(prefix) - len(tail)
    st = encode(text) if text else []
    st = st[:room]
    off = len(prefix) + len(st)
    return {"ids": prefix + st + tail, "markers": [m + off for m in markers],
            "option_span": [len(head) + off, len(tail) + off], "images": n_images}
