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

**Status: runtime implemented and at parity on CPU; library entry not published yet.**

| Step | Status |
|---|---|
| 1. Export, ORT parity against the PyTorch reference, goldens (`convert/ollaya_convert/families/laya_vision/`) | **done**, [measured](#measured-parity) |
| 2. Rust layout (`ollaya_decision::laya_vision`) against the goldens' ids | **done**: 40/40 rows identical |
| 3. Rust image preprocessing (`ollaya_runner::image`) against the goldens' pixel hashes | **done**: 26/26 images bit-exact |
| 4. Engine, API, CLI and MCP wiring | **done**, [below](#runtime) |
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

### Images in the request

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
  data URLs. `ollaya run laya-vision --image a.jpg --image b.jpg "note"` sends
  `{"images": [a, b], "text": "note"}`, with the images in argument order.
- A string, array or image-less object state is a text-only request. It works: the rows simply have
  no image run.
- On a model without the vision layout, a state with `image`/`images` keys is just JSON text, as
  today. The `laya` router must not send an image state to a text model.
- A bad image is a 400 whose validation issue points at it (`docs/api.md` §4.4): `string_type` /
  `list_type` (wrong JSON type), `image_data` (not base64), `image_type` (not PNG, JPEG, WebP or
  GIF), `image_decode`, `image_too_large` (over 50 megapixels, checked before decoding),
  `too_many_images` (over 16), `image_size` (resize off, not 512×512).

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

Ollaya deviations, on inputs that never occur with string criteria:

1. A non-string criterion value renders as JSON (Laya's rule, which the shared question parser
   applies to every Laya layout); upstream laya-vision uses Python `str()`.
2. noul criteria keys match case-insensitively, as the shared parser does; upstream reads only
   `"true"` / `"false"`.

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
5. **Normalize:** `(v − 127.5) / 127.5` in float32, giving `pixel_values [3, 512, 512]`. This is
   the processor's own arithmetic, bit for bit; `(v / 255 − 0.5) / 0.5` is the same map but is off by
   one ulp on some values. The pixel attention mask is all ones, so the graph does not take it.

Steps 2–4 are the default. `resize: false` skips them ([below](#resize-is-optional)).

**The resampler must be torchvision's, not PIL's.** The fast processor calls
`torchvision.transforms.v2.functional.resize(..., LANCZOS, antialias=True)` on a uint8 tensor
(torchvision ≥ 0.27, CPU). On CUDA, or with an older torchvision, transformers silently falls back
to BICUBIC. The goldens use the CPU path, which is what the checkpoint's published numbers used.
Measured on the case set, PIL's `Idefics3ImageProcessorPil` differs from the torchvision output by up
to 2 grey levels on 0.1–3.5% of pixels.

**The exact kernel.** On the CPU, torchvision hands a uint8 image straight to ATen
(`interpolate(mode="lanczos", antialias=True)`), whose uint8 path is Pillow-SIMD's fixed-point
convolution. Pinned down empirically, bit-exact against torch on random images at 14 sizes
(including the real 2048 and 512 hops), then on every golden image:

- separable, **horizontal pass first**, the intermediate rounded to uint8; an axis whose size does
  not change is not resampled;
- per output pixel `i`: centre `(i + 0.5) · scale`, support `3 · max(scale, 1)`, bounds
  `int(centre ± support + 0.5)` clipped to the image, weights `lanczos3((x − centre + 0.5) / max(scale, 1))`
  normalised to sum 1 (in f64);
- weights quantized to integers at the largest precision `p ≤ 22` for which
  `int(0.5 + max_weight · 2^(p+1)) < 2^15` (they fit int16), rounding half away from zero;
- `acc = 2^(p−1) + Σ pixel · weight`, output `clamp(acc >> p, 0, 255)`.

A float-weight implementation (laya-vision's `_axis_weights`, or any "Lanczos" from an image
library) is off by up to 2 levels on 0.2–1.7% of pixels (measured on the same random images). `ollaya_runner::image` implements the fixed-point
kernel; the goldens pin each image's final pixels by sha256, so the preprocessing is tested bit for
bit, separately from the network (`--pixels-dir` dumps them as `.npy` for debugging).

**JPEG is decoded by libjpeg.** Upstream decodes with PIL, which uses libjpeg-turbo (3.1.4 in
Pillow 12). The pure-Rust decoders do not match it: zune-jpeg (the `image` crate's) and
jpeg-decoder differ by up to 3 levels on 0.4–8% of pixels, even on greyscale files, so the IDCT
differs, not only chroma upsampling. The `mozjpeg` crate (libjpeg-turbo's decoder, built from C
with the system compiler) matches libjpeg-turbo bit for bit on 4:4:4, 4:2:2, 4:2:0 and greyscale
files and on both golden JPEGs. The runner uses it for JPEG only, with its SIMD (nasm) build turned
off so every platform runs the same C code; PNG, WebP and GIF stay on the pure-Rust `image`
decoders. This is the runner's first C dependency besides ONNX Runtime. libjpeg's fatal errors
unwind to a `catch_unwind` and become `image_decode` issues.

Decoding deviations, untested by the goldens:

- A **truncated JPEG** decodes with its missing part grey (libjpeg's behaviour); PIL refuses it.
- A **CMYK JPEG** is refused (`image_decode`); PIL converts it.
- **16-bit PNG** is scaled to 8 bits by the `image` crate (rounded); PIL keeps the high byte.
- **WebP** uses the pure-Rust decoder; PIL uses libwebp. Lossless WebP should match; lossy WebP
  has not been compared.

### Resize is optional

Resizing is **on by default**: it is the path the checkpoint was trained and evaluated on. A caller
can turn it off when its images are already the model's input size. Two typical cases are game
frames rendered at 512×512, and images a pipeline has already resized. Turning it off:

- skips steps 2–4, including the 2048-pixel intermediate: the most expensive part of preprocessing,
  12.7 ms of CPU for a 210×160 frame, as upstream measured it;
- requires **every** image in the request to be exactly 512×512 after decoding. Any other size is
  refused with a 400 `image_size` issue that names the image (`loc` `["body","state","image"]` or
  `["body","state","images",n]`, `ctx`
  `{"width": 512, "height": 512, "actual_width": w, "actual_height": h}`). Ollaya does not pad, crop
  or fall back to a resize. For any other size, the processor's own `do_resize=False` path still
  resizes, by a route the default never takes. Rejecting avoids porting a second resampler whose
  output no checkpoint number describes.
- sends the decoded pixels, only normalized, to the graph. This is the checkpoint's own processor
  with `do_resize=False`, which leaves a 512×512 image untouched (`ref.unresized_pixels` checks that
  bit for bit on every golden). Rows, graph and calibration do not change.

How to set it:

| Where | How |
|---|---|
| `/api/decide` | `"options": {"resize": false}`. `options` is a closed set (`docs/api.md` §7.3): a non-boolean `resize` is a `bool_type` issue, another key an `extra_forbidden` issue. Models without images ignore `resize`. |
| Modelfile | `PARAMETER resize false`, next to `PARAMETER precision`, for a derived model such as `FROM laya-vision` / `PARAMETER resize false`. This is how `/v1/*` callers get it, because `/v1/*` stays TypeSafe-identical and takes no native options. A request's `options.resize` wins over the model's parameter. |
| CLI | `ollaya run laya-vision --image frame.png --no-resize …` |
| MCP | `decide` takes `resize` (boolean) next to `images`. |

**What skipping costs.** Resizing a 512×512 image is not the identity: 512 → 2048 → 512 softens it
slightly. So with resize off the model sees sharper pixels than it was trained on. Measured with the
reference (backbone in float64) on the golden cases, resize off against on:

| Image (512×512) | Largest probability change | Answer |
|---|---|---|
| photo-like scenes: PNG, JPEG, RGBA, two images (7 questions) | 0.002–0.044 | same |
| 16-pixel checkerboard, all hard edges (2 questions) | 0.07, 0.18 | changed on 1 of 2 (`clutter` level 0 instead of 1) |

That is why the default stays on. Turn it off for speed on images that are already 512×512, and
expect the most change on synthetic, hard-edged content.

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
- **Request body:** the daemon's limit is now 32 MiB (`ollaya_api::MAX_BODY_BYTES`, was 8 MiB),
  room for a few photos as base64. The runner's `/decide` used axum's `Json` default of 2 MB,
  which would have refused most photos between the daemon and the runner; it now allows 256 MiB
  (only the daemon calls it, after its own limit).
- **Decoding guards:** an image over 50 megapixels is refused from its header, before decoding (a
  decompression bomb); a file that does not decode is refused. Both are 400s naming the image.
- **Memory:** stage 1 allocates a 2048-pixel uint8 intermediate per image (at most 12 MB).
- **Usage:** `input_tokens` counts the image tokens (67 per image per row), as upstream's
  `usage.input_tokens` does. A separate image count in `usage` is not added.
- **Logging:** `state` is user data and is not logged (PROJECT_NOTES). That covers images.

## Measured parity

`uv run --with pillow --with 'torchvision==0.29.*' python -m ollaya_convert.families.laya_vision.parity out/laya-vision-wl`

The set is 26 requests and 40 questions (`cases.py`), all with synthetic images generated
deterministically:

- PNG in RGB, RGBA, greyscale and palette, plus a JPEG;
- sizes from 48×36 (upscaled) through exactly 512 to 2600×1100 and 700×3000 (the two-hop path), and
  a 1200×40 strip;
- two images in one state, `image` and `images` together, and text-only states;
- state truncation, the option budget, long instructions, 10-level scores, an Atari frame, and
  non-ASCII text;
- six requests with `resize: false` (PNG, JPEG, RGBA, a checkerboard, two images, no image), plus
  two it must reject (`cases.rejected()`).

The reference is upstream's network, one unpadded row at a time, with the backbone in float64
(`ref.Exact`). The head stays in fp32 because upstream casts to fp32 before it. ONNX ran on the CPU
EP, ONNX Runtime 1.30, one padded batch per request.

| Check | Result |
|---|---|
| `layout.py` + `tokenizer.json` (Rust core) vs upstream rows (ids, markers, option span) | 40/40 identical |
| argmax, ONNX vs reference | 40/40 |
| option logits, max abs | 7.6e-5 (p99 6.4e-5) |
| act logits, max abs | 1.4e-5 |
| probabilities, max abs | choice 1.1e-6, score 2.8e-6, noul 6.5e-6 |
| reference vs `VLMAgent.predict` (4 dp), resize-on requests | 30/30, max 4.9e-5 (predict's rounding and fp32 backbone) |
| resize-off requests (10 questions, 6 requests), ONNX vs reference | 10/10 argmax; included in the rows above |
| resize off with a wrong-sized image rejected | 2/2 |

- Before export, the graph module run eagerly matched the reference to 0.0 on single-row requests
  and 2.8e-5 on padded batches.
- The weightless graph (`-wl`) gives the same numbers as the exported one.
- The fp16 graph (`fp16.py`) needs a CUDA GPU to measure and **has not been measured**. The
  vision tower's LayerNorms and the `finfo.min` mask are the usual fp16 suspects.

Goldens: `python -m ollaya_convert.families.laya_vision.goldens` writes
`out/goldens-laya-vision.jsonl` (26 cases plus 2 rejections, 10.4 MB) in about 1 minute on 4 CPU
cores. Each line holds:

- the wire request, with images as data URLs, and its `options` (`{}` or `{"resize": false}`);
- the state text;
- per image, the sha256 of the 512×512×3 uint8 pixels;
- per question, the row ids, markers, option span, truncation, logits, act logits, temperature and
  probabilities;
- upstream's own `predict` answers (null with resize off: upstream has no such switch).

The two rejection lines hold the request and the expected `INVALID_REQUEST` body with its
`image_size` issue.

## Rust runtime parity

`cargo run --release -p ollaya-runner --example parity_laya_vision -- convert/out/laya-vision-wl convert/out/goldens-laya-vision.jsonl cpu --latency`

The runtime runs every golden request as it serves one (decode, resize, tokenize, one padded batch
through the fp32 graph on the CPU EP), and checks each layer separately:

| Check | Result |
|---|---|
| state text (images removed) | 26/26 identical |
| image pixels, sha256 of the prepared 512×512×3 uint8 array | **26/26 bit-exact** (PNG RGB, RGBA, greyscale, palette, JPEG; upscaled, one hop, two hops; resize off) |
| rows: ids, markers, option span | 40/40 identical |
| argmax, runtime vs goldens | 40/40 |
| option logits, max abs | 1.2e-4 (tolerance 1e-3, as von's) |
| act logits, max abs | 2.7e-5 |
| calibrated probabilities, max abs | 1.0e-5 |
| requests refused with the golden's `image_size` issue | 2/2 |

Before JPEG went through libjpeg, the two JPEG cases failed the pixel check and moved logits by up
to 0.23 (answers unchanged): the decoder, not the model, was the largest source of error.

**Speed** (this sandbox: 4 CPU cores, fp32, one request at a time; not a benchmark): 0.6–0.9 s per
request with one image and 1–4 questions, 1.3–1.5 s with two images. Encoding (decode, resize,
tokenize) is 50–130 ms per request with one image and resize on (the 2048-pixel intermediate), and
about 3 ms with resize off. GPU and fp16
are not measured.

A smoke test through the real runner process (`ollaya runner --graph-fp32 … --device cpu`, HTTP)
answered a 512×512 JPEG of a green triangle "green" and "triangle" with resize on and off, took two
images in one request, and refused a non-image payload with an `image_type` issue at
`state.images.0`.

## Runtime

| Piece | Where |
|---|---|
| Layout: `split_state`, `prefix_ids`, `encode` → ids, markers, option span; image issues | `crates/ollaya-decision/src/laya_vision.rs` |
| Decode (libjpeg for JPEG, `image` for PNG/WebP/GIF), exact Lanczos, `resize` off, normalize | `crates/ollaya-runner/src/image.rs` |
| Engine: contract V, one `session.run` per token-budget batch, pixels fed to every batch | `crates/ollaya-runner/src/laya_vision.rs` |
| `RunOptions { resize }`, `Engine::run_with`; runner `/decide` takes `options`, returns image issues as `detail` | `crates/ollaya-runner/src/{lib,engine,server}.rs` |
| `DecideOptions`, `options` validation (`bool_type`, `extra_forbidden`), `parameters.resize`, 32 MiB body | `crates/ollaya-api` |
| Options to the runner (request, else the model's `resize` parameter); runner issues passed through as the 400's `detail`; `resize` merged into the `params` layer; `show` lists it | `crates/ollaya-server` |
| CLI `--image` (repeatable, in order) and `--no-resize`; `PARAMETER resize` in Modelfiles | `crates/ollaya/src/{run,modelfile,commands}.rs` |
| MCP `decide`: `images` (paths or data URLs) and `resize` | `crates/ollaya/src/mcp.rs` |
| Parity against the goldens | `crates/ollaya-runner/examples/parity_laya_vision.rs` |

`crates/ollaya-server/tests/http.rs` (`options_and_image_issues`) covers the daemon end to end with
a fake runner: `options.resize` reaches the runner from `/api/decide` but not from `/v1/*`, a
model's `PARAMETER resize` is the default and the request overrides it, and the runner's image
issue comes back as the request's own 400 body.

Not done: fp16 (unmeasured, so ship fp32 only); `ollaya show` capabilities; the `laya` router does not yet keep an image state away from its text models (it would read the data URL as text); and the step below.

## Remaining: library entry

`laya-vision` (`:latest` = `:201m`), with the manifest pointing at `thaitea/laya-vision` @
`8b318c99…` (`package.py`, as for the other families). Add a site page and a README table row.

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
cd ..
cargo run --release -p ollaya-runner --example parity_laya_vision -- \
    convert/out/laya-vision-wl convert/out/goldens-laya-vision.jsonl cpu --latency
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
