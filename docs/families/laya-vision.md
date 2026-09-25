# laya-vision (`laya-vision-terminator-v1`)

Laya Vision (`thaitea/laya-vision`) makes typed, calibrated decisions about an **image plus optional
text**. Its API and outputs are Laya's: `predict(state, questions)` gives one logit per option, then a
per-type temperature and a softmax, in one forward pass with no text generation. The backbone is a
small vision-language model (SmolVLM-256M cut to 20 of its 30 language layers) in place of
ModernBERT. The same model also plays simple games from pixels: the screen is the image and the
buttons are the options.

It is the first image-input family. It needs three things no other family has:

- images in the request;
- image preprocessing in the runner;
- a graph that runs a vision tower.

Everything after the logits is Laya's: calibration, answers, the wire format.

**Status: conversion done, runtime not started.**

| Step | Status |
|---|---|
| 1. Export, ORT parity against the PyTorch reference, goldens (`convert/ollaya_convert/families/laya_vision/`) | **done**, measured below |
| 2. Rust layout (`ollaya_decision::laya_vision`) against the goldens' ids | to do |
| 3. Rust image preprocessing against the goldens' pixel hashes | to do |
| 4. Engine, API, CLI and MCP wiring (`ollaya_runner::laya_vision`, `--image`, body limits) | to do |
| 5. Library entry, registry manifest, site page | to do |

| | |
|---|---|
| Upstream | [huggingface.co/thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision) @ `8b318c99d7ad3ce19c24369263463882eada9d1e`; code: [github.com/r33drichards/laya-vision](https://github.com/r33drichards/laya-vision) @ `568feee` (package `laya` 0.2.0.dev0) |
| License | Apache-2.0. Laya Vision is an independent fork of Laya (Convai Innovations). Base model: SmolVLM-256M-Instruct (Hugging Face), Apache-2.0 |
| Size | 201.2M parameters, fp32, 805 MB: vision tower 86.4M, connector 7.1M, text model (20 layers) 99.2M, decision head 8.5M |
| Contract | V (`vision-markers`), one row per question; every row carries the request's images |
| Context | `max_len` 1024 tokens; each image costs 64 + 3 tokens |
| Language | English |
| Reference | `convert/ollaya_convert/families/laya_vision/ref.py`. It calls the pinned laya-vision code for state splitting, preprocessing, sequence ids and the network |

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (4.0 MB, graph only) | derived, hosted by Ollaya (`convert/out/laya-vision-wl/model.onnx`) |
| weights | **`thaitea/laya-vision` @ `8b318c99…` / `model.safetensors`** (804,696,880 B, sha256 `31220803…4be68`), referenced in place by byte offset |
| `tokenizer.json` | `thaitea/laya-vision` @ `8b318c99…` / `processor/tokenizer.json`, used as is |
| `decision.json` | derived, hosted by Ollaya: layout, prompt strings, image settings, token ids |
| `calibration.json` | derived, hosted by Ollaya: the checkpoint's three temperatures |

- The weights file is one plain safetensors file, so `weightless.py` references it directly (prefix
  `model.`).
- 415 of the 416 checkpoint tensors are external, 221 of them through a Transpose. The one left over
  is the `temperature` buffer, which `calibration.json` carries. The only inline data is 9.4 KB of
  constants: shape scalars, the vision position ids (0..1023) and the RoPE frequencies.
- The scorer's output projection (576 → 1) is pre-transposed by the exporter and has only 576
  elements, below the size above which `weightless.py` matches transposed weights by value. The new
  `--small-transposes` flag matches it anyway, but only when exactly one unused checkpoint tensor
  has the transposed shape and equal values. Without it, those 576 trained floats would be copied
  into the graph.

## Request → rows

### Images in the request (proposed, step 4)

Images travel inside `state`, which is already an untyped JSON value on every endpoint
(`crates/ollaya-api/src/decide.rs`). No schema changes, and `/v1/systemone` stays wire-identical to
TypeSafe. The convention is upstream's own (`laya.vlm.split_state`):

```json
{ "model": "laya-vision",
  "state": { "image": "data:image/jpeg;base64,/9j/4AAQ…", "note": "customer says it arrived broken" },
  "questions": { "damage": { "type": "score", "criteria": ["none", "cosmetic", "functional", "destroyed"] } } }
```

- Only a JSON **object** state can carry images: `"image"` holds one, `"images"` a list. The order
  is `image` first, then `images` in list order. The other keys are the state's text, serialized
  like a Laya state: `json.dumps(ensure_ascii=False)`, or `""` when no other key is left.
- An image is a `data:` URL (`data:image/png;base64,…`); plain base64 without the prefix is also
  accepted. Formats: PNG, JPEG, WebP and GIF (first frame).
- The daemon never fetches a URL or reads a path. The CLI and the MCP tool read files and send
  data URLs (`ollaya run laya-vision --image photo.jpg "note"`).
- A string, array or image-less object state is a text-only request. It works: the rows simply have
  no image run.
- On a model without the vision layout, a state with `image`/`images` keys is just JSON text, as
  today. The `laya` router must not send an image state to a text model.

### Row text and ids (`laya.vlm.build_vlm_inputs`)

One row per question. Each piece is tokenized on its own with the checkpoint's tokenizer (GPT-2 BPE,
no special tokens), and the pieces are concatenated:

```
prefix  "<|im_start|>User:" + per image "<fake_token_around_image><global-img>" + "<image>"×64 + "<fake_token_around_image>"
state   the state text, cut from the right to what max_len leaves
tail    "\n{type} question: {instructions}<end_of_utterance>\nAssistant: Options:\n"
        per option: "- " + option text (inner "\n" → " "), cut to 48 tokens, then the "\n" token (id 198)
```

- **prefix:** one string, tokenized once, for the whole image run. `<image>` is id 49190.
- **Option text** is Laya's `render_options`:
  - choice: `key`, or `key: description`;
  - score: `level {i}: {criterion}`;
  - noul: `false: …`, then `true: …`. The defaults are "no, the statement does not hold" and
    "yes, the statement holds".
- **Instructions:** a non-string value is rendered with `json.dumps` (ASCII-escaped).
  `<end_of_utterance>` inside the instructions becomes a space.
- **Budgets** (`head_max_len` 256):
  - If the options plus terminators leave fewer than 16 tokens, every option is cut to
    `max(4, (256 − 16) // k − 1)` tokens.
  - If the question piece is longer than `max(8, what the options leave)`, its middle is dropped.
    The first `keep // 2` tokens and the last `keep − keep // 2` are kept, which keeps the
    "Assistant: Options:" cue.
- **State:** `room = max_len − len(prefix) − len(tail)`, and the state keeps its first `room`
  tokens. If `len(prefix) + len(tail) > max_len`, the question is rejected (400, naming it).
- **Markers** are the positions of each option's `\n` terminator. **option_span** is
  `[first option token, end of row)`.
- **Truncation report:** upstream reports what was cut as `truncated` on each answer. Ollaya reports
  a cut state through the existing `state_truncated` field (`/api/decide`), as it does for Laya. Cut
  options and instructions are not reported, also as for Laya. The goldens keep upstream's report
  (`truncation`) in case that changes.

`layout.py` in the family folder states these rules without calling upstream. It is the spec for
`ollaya_decision::laya_vision`. Run with the Rust tokenizer core, it reproduces every upstream row
(below).

### One option order

Upstream's `predict(n_permutations=1)` is the default and what the published numbers use. Ollaya
does the same: one row per question, options in the caller's order.

## Images → pixels

The checkpoint's processor is `Idefics3ImageProcessor`, the torchvision backend that
`AutoProcessor` picks in transformers 5.x. Splitting is off (`image_split_edge: 0`). Per image:

1. **Decode** the file and convert it to RGB, as PIL's `convert("RGB")` does:
   - alpha is dropped, not composited;
   - greyscale is replicated to three channels;
   - palette images are expanded.
   EXIF orientation is **not** applied, because upstream does not apply it.
2. **Resize 1 (Lanczos, antialiased):** longest edge to 2048. The other edge is
   `int(2048 · short / long)`, rounded up to even. This upscales small images.
3. Round and clamp to uint8.
4. **Resize 2 (Lanczos, antialiased):** to 512 × 512, ignoring the aspect ratio. Round and clamp
   to uint8.
5. **Normalize:** `(v / 255 − 0.5) / 0.5` in float32, giving `pixel_values [3, 512, 512]`. The pixel
   attention mask is all ones, so the graph does not take it.

**The resampler must be torchvision's, not PIL's.** The fast processor calls
`torchvision.transforms.v2.functional.resize(..., LANCZOS, antialias=True)` on a uint8 tensor
(torchvision ≥ 0.27, CPU). On CUDA, or with an older torchvision, transformers silently falls back
to BICUBIC. The goldens use the CPU path, which is what the checkpoint's published numbers used.
Measured on the case set, PIL's `Idefics3ImageProcessorPil` differs from the torchvision output by up
to 2 grey levels on 0.1–3.5% of pixels.

So the Rust port reproduces torchvision's kernel:

- ATen's antialiased separable resample: Lanczos-3, support `3 · max(1, scale)`, centres at
  `(i + 0.5) · scale`, weights clipped at the borders and renormalised;
- the uint8 rounding after each hop.

laya-vision's `laya/preprocess.py` (`_axis_weights`) already re-derives these weights and checks them
against torchvision to float64 agreement. The goldens pin each image's final uint8 pixels by sha256,
so the Rust preprocessing is tested bit for bit, separately from the network. `--pixels-dir` dumps
them as `.npy` for debugging. JPEG decoding is a second risk: Rust decoders are not guaranteed to
match libjpeg-turbo to the bit, and one golden case is a JPEG. If that case is off by a grey level,
the fix is a bit-exact decoder, never a tolerance.

## ONNX contract (V)

| Tensor | Type | Shape |
|---|---|---|
| `input_ids` | int64 | [n, s], right-padded with 2 (`<\|im_end\|>`) |
| `attention_mask` | int64 | [n, s] |
| `option_span` | int64 | [n, 2] |
| `marker_pos` | int64 | [n, k] |
| `marker_mask` | bool | [n, k] |
| `qtype` | int64 | [n]; choice 0, score 1, noul 2 |
| `pixel_values` | float32 | [i, 3, 512, 512], the request's images in state order |
| → `logits` | float32 | [n, k]; masked slots are −1e4 |
| → `act_logits` | float32 | [n, 2] |

- Dynamic axes: n 1..1024, s 8..1024, k 2..255 (pad the marker axis to 2), i 1..16. Opset 20.
- A request without images feeds **one all-zero image**. No row has an `<image>` token, so nothing
  reads it; it costs one vision-tower pass. A text-only graph would avoid that; see
  [Later](#later).
- The graph is `VLMDecisionModel.forward`, with the two data-dependent steps of the Hugging Face
  forward replaced as in laya-vision's own `laya/static_step.py`:
  - **Vision position ids.** A full 512 tile always gets 0..1023, recorded once at export and
    checked there.
  - **Merging image features** (`masked_scatter`). The j-th `<image>` token of a row takes the j-th
    image-feature vector: `cumsum` + `gather` + `where`. Every row holds every image, in order, so
    the vision tower and connector run **once per image per request**, not once per row.
- **Text model.** It gets an additive 4-D mask: causal, plus bidirectional attention inside
  `option_span`, plus padding keys masked with `finfo.min` (upstream's `option_block_mask`). Its
  position ids are `0..s`.
- **Decision head** (upstream's `_readout`):
  - `h.float()` plus the type embedding at every position.
  - Two `nn.TransformerEncoderLayer`s (d 576, 9 heads, ff 2304, pre-norm, **ReLU**) over the whole
    row, with a key-padding mask.
  - A gather at the markers, then the scorer: LayerNorm, Linear, GELU, Linear → 1.
  - The act head reads the last real token and `[top1, top1 − top2, normalised entropy, k/255]`.

## Calibration and answers (server)

Laya's, unchanged. `service.rs` needs no new code for this family.

- `T = temperature_by_options[bucket]` if present (empty for this checkpoint), else
  `temperature[qtype]`. The values are choice 3.8574, score 2.0998, noul 3.0531.
- `p = softmax(logits / max(T, 1e-3))`.
- The answers are Laya's:
  - choice: argmax;
  - score: `Σ i·p_i`;
  - noul: `p[true]`, with confidence `max(p, 1 − p)`;
  - `act_probability` = `softmax(act_logits)[0]`.

## Limits

- **Context:** 1024 tokens. Each image costs 67 tokens (64 image tokens plus 3 framing tokens), so
  at most 15 images fit next to a short question. `max_images` is 16 in the graph.
- **Request body:** a 5 MP JPEG is about 2 MB as base64, and axum's `Json` extractor rejects bodies
  over 2 MB by default. The daemon and the runner's `/decide` need `DefaultBodyLimit`. Proposed: an
  `OLLAYA_MAX_BODY` setting defaulting to 32 MB, listed with the other settings in `docs/api.md`
  §15.
- **Decoding guards:** reject an image over 50 megapixels before decoding (a decompression bomb),
  or a file that does not decode. Both return a 400 naming the state key.
- **Memory:** stage 1 allocates a 2048-pixel uint8 intermediate per image (at most 12 MB).
- **Usage:** `input_tokens` counts the image tokens (67 per image per row), as upstream's
  `usage.input_tokens` does. `/api/decide` adds `images`.
- **Logging:** `state` is user data and is not logged (PROJECT_NOTES). That covers images.

## Measured parity

`uv run --with pillow --with 'torchvision==0.29.*' python -m ollaya_convert.families.laya_vision.parity out/laya-vision-wl`

The set is 20 requests and 30 questions (`cases.py`), all with synthetic images generated
deterministically:

- PNG in RGB, RGBA, greyscale and palette, plus a JPEG;
- sizes from 48×36 (upscaled) through exactly 512 to 2600×1100 and 700×3000 (the two-hop path), and
  a 1200×40 strip;
- two images in one state, `image` and `images` together, and text-only states;
- state truncation, the option budget, long instructions, 10-level scores, an Atari frame, and
  non-ASCII text.

The reference is upstream's network, one unpadded row at a time, with the backbone in float64
(`ref.Exact`). The head stays in fp32 because upstream casts to fp32 before it. ONNX ran on the CPU
EP, ONNX Runtime 1.30, one padded batch per request.

| Check | Result |
|---|---|
| `layout.py` + `tokenizer.json` (Rust core) vs upstream rows (ids, markers, option span) | 30/30 identical |
| argmax, ONNX vs reference | 30/30 |
| option logits, max abs | 7.6e-5 (p99 6.7e-5) |
| act logits, max abs | 1.4e-5 |
| probabilities, max abs | choice 7.2e-7, score 2.8e-6, noul 6.5e-6 |
| reference vs `VLMAgent.predict` (4 dp) | 30/30, max 4.9e-5 (predict's rounding and fp32 backbone) |

- Before export, the graph module run eagerly matched the reference to 0.0 on single-row requests
  and 2.8e-5 on padded batches.
- The weightless graph (`-wl`) gives the same numbers as the exported one.
- The fp16 graph (`fp16.py`) needs a CUDA GPU to measure and **has not been measured**. The
  vision tower's LayerNorms and the `finfo.min` mask are the usual fp16 suspects.

Goldens: `python -m ollaya_convert.families.laya_vision.goldens` writes
`out/goldens-laya-vision.jsonl` (20 cases, 6.6 MB) in about 1 minute on 4 CPU cores. Each line
holds:

- the wire request, with images as data URLs;
- the state text;
- per image, the sha256 of the 512×512×3 uint8 pixels;
- per question, the row ids, markers, option span, truncation, logits, act logits, temperature and
  probabilities;
- upstream's own `predict` answers.

## Runtime plan

### 2. Layout: `crates/ollaya-decision/src/laya_vision.rs`

- `split_state(&Value) -> (Vec<ImageRef>, String)`, using the existing `pyjson` for the text part.
  It returns image **references** (state key, index); the bytes stay in the request.
- `LayaVisionLayout::encode(state, question) -> Encoded { ids, markers, option_span, state_truncated }`,
  with the rules above and the image-run string from `decision.json`.
- Tests: an ids golden test over `goldens-laya-vision.jsonl`, like the other families.

### 3. Preprocessing: `crates/ollaya-runner/src/image.rs`

- Decode with the `image` crate (PNG, JPEG, WebP, GIF), with no EXIF rotation. Convert to RGB8 as
  PIL does.
- A port of ATen's antialiased Lanczos (separable, float weights, uint8 round and clamp per hop),
  run twice as above, then normalize into the `pixel_values` buffer.
- Tests: the sha256 of every golden image's pixels, bit-exact. This is a new `image` crate
  dependency in the runner only; it's pure Rust, and the daemon does not decode images.

### 4. Wiring

- **Runner engine** (`crates/ollaya-runner/src/laya_vision.rs`):
  - add `laya-vision-terminator-v1` to `LAYOUTS` with a match arm in `engine::load`;
  - `Engine::run(state, questions)` keeps its signature, since the images are inside `state`;
  - decode and preprocess once per request, then one `session.run` per batch, with the same
    `pixel_values` fed to every batch;
  - batches are split by token budget as today, so image rows count their 67 tokens per image.
- **IPC:** `POST /decide` on the runner stays JSON, with the images inside it as base64. The size is
  bounded by the body limit. A binary side channel is not worth it at these sizes.
- **Server:**
  - `DefaultBodyLimit` on the daemon's and the runner's routers;
  - image count and pixel guards return 400s with the existing `{error}` body;
  - `usage.images` goes on `/api/decide` only.
- **CLI:** `ollaya run <model> --image PATH` (repeatable) builds `{"image": …}` or
  `{"images": […]}` plus the prompt text as `"text"`. `ollaya show` lists "vision" under
  capabilities.
- **MCP:** the `decide` tool gains an `images` argument (paths or data URLs) that the tool turns
  into data URLs before calling the daemon.
- **Registry and manifest:** no new media types. The `decision` layer carries the image settings;
  `ModelConfig.family` is `laya-vision`.
- **Precision:** fp32 on CPU and fp16 on GPU, as for Laya, once fp16 parity is measured. Until then,
  ship fp32 only.

### 5. Library entry

`laya-vision` (`:latest` = `:201m`), with the manifest pointing at `thaitea/laya-vision` @
`8b318c99…`. Add a site page and a README table row.

## Later

- **Vision-tower cache.** Split the graph into `vision.onnx` (pixels → `[i, 64, 576]` features) and
  `decide.onnx` (features in place of pixels). The runner could then cache features by image hash
  across requests: the same photo with new questions, or the previous frame in two-frame game play.
  It would also stop text-only requests from paying for a dummy image. This needs multi-graph
  support in `ModelFiles` / `RunnerConfig`, which no family has yet.
- **Prefix KV cache.** Upstream's `encode_prefix` / `forward_prefixed`: the image run and state are
  shared by every row. It only pays off with many questions per request, and needs KV outputs from
  the graph.
- **Other checkpoints.** `laya-vision-modernvbert-250m` is a second layout (`[MASK]` readout,
  bidirectional, state after the options) and reuses this preprocessing and image run.
- **Other vision families.** Qwen3.5, the base under `kev` and `decider`, is a VLM. An image-aware
  kev would need 3-D mRoPE position ids as a graph input, which `llm_common/qwen35.py` collapses
  today.

## Reproduce

```sh
cd convert
W="--with pillow --with torchvision==0.29.*"
uv run $W python -m ollaya_convert.families.laya_vision.export --out out/laya-vision
uv run python -m ollaya_convert.weightless out/laya-vision \
    ~/.cache/huggingface/hub/models--thaitea--laya-vision/snapshots/8b318c99d7ad3ce19c24369263463882eada9d1e/model.safetensors \
    --prefix model. --small-transposes --out out/laya-vision-wl
uv run $W python -m ollaya_convert.families.laya_vision.parity out/laya-vision-wl
uv run $W python -m ollaya_convert.families.laya_vision.goldens --out out/goldens-laya-vision.jsonl
```

- `pillow` and `torchvision` are added per run, as `von-sdk` is for von. The convert lockfile cannot
  be re-resolved because `laya==0.3.7` is no longer on PyPI.
- The upstream code is fetched once, at the pinned commit, into `convert/out/src/` and imported as
  `laya_vision`. It is also a package named `laya`, which would clash with the text model's package.
  `LAYA_VISION_SRC=<checkout>` uses a local clone instead, which must be at the pinned commit.
- Everything runs on CPU: export in about 1 minute, parity in 1.5 minutes, goldens in about
  1 minute, on 4 cores with 16 GB of RAM.

## Attribution

Laya Vision, Apache-2.0, by thaitea (r33drichards/laya-vision), an independent fork of Laya by Convai
Innovations. SmolVLM-256M-Instruct © Hugging Face, Apache-2.0.
