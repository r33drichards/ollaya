"""The request set for Laya Vision parity checks and goldens: synthetic images, generated deterministically.

Images travel the way they will over HTTP: as `data:` URLs (base64) under the state keys `"image"` and
`"images"`. `materialize` turns them into the encoded bytes upstream's `split_state` accepts, so the
reference decodes exactly the file the runtime will decode.

Every preprocessing path the Rust port must match has a case: PNG RGB / RGBA / greyscale / palette, a JPEG,
upscaling (smaller than 512), a single resize (between 512 and 2048), the two-hop resize (longest edge over
2048), extreme aspect ratios, odd sizes, two images in one state, no image at all, and an image-only state.
The sequence builder gets the same edge cases as the text families: state truncation, option-budget
truncation, long instructions, list and dict criteria, noul with and without criteria, 10-level scores.

`options` holds the request's native options; `{"resize": False}` skips image resizing, which needs every image
to be 512x512 already. `rejected()` lists requests the runtime must refuse (a 400 naming the image).
"""
import base64
import io
from typing import Any, Dict, List, Tuple

import numpy as np

Case = Tuple[str, Any, Dict[str, Dict[str, Any]], Dict[str, Any]]  # (id, state, questions, options)


def _png(img, **kw) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", **kw)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jpeg(img, quality=90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=2)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _scene(w: int, h: int, seed: int, shape: str = "circle", colour=(220, 30, 30), bg=(245, 245, 240)):
    """A plain background with light noise and one filled shape in the middle third."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    base = np.array(bg, dtype=np.float64)[None, None, :] + rng.normal(0, 6, (h, w, 3))
    img = Image.fromarray(base.clip(0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    box = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
    if shape == "circle":
        d.ellipse(box, fill=colour)
    elif shape == "square":
        d.rectangle(box, fill=colour)
    else:
        d.polygon([(w // 2, h // 3), (w // 3, 2 * h // 3), (2 * w // 3, 2 * h // 3)], fill=colour)
    return img


def _gradient(w: int, h: int):
    from PIL import Image

    x = np.linspace(0, 255, w)[None, :, None]
    y = np.linspace(0, 255, h)[:, None, None]
    arr = np.concatenate([np.broadcast_to(x, (h, w, 1)), np.broadcast_to(y, (h, w, 1)),
                          np.broadcast_to((x + y) / 2, (h, w, 1))], -1)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _checker(w: int, h: int, cell: int):
    """Hard edges everywhere: the worst case for a resampler mismatch."""
    from PIL import Image

    yy, xx = np.mgrid[0:h, 0:w]
    on = ((yy // cell + xx // cell) % 2).astype(np.uint8) * 255
    return Image.fromarray(np.stack([on, 255 - on, on // 2], -1), "RGB")


def _frame():
    """An Atari-sized frame (160x210): a paddle, a ball and a brick wall."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (160, 210), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for row, c in enumerate([(200, 72, 72), (198, 108, 58), (180, 122, 48), (162, 162, 42)]):
        d.rectangle((8, 40 + 6 * row, 151, 45 + 6 * row), fill=c)
    d.rectangle((60, 190, 75, 193), fill=(200, 72, 72))
    d.rectangle((110, 120, 111, 123), fill=(200, 72, 72))
    return img


COLOUR_Q = {"type": "choice", "instructions": "What colour is the shape in the middle of the image?",
            "criteria": {"red": "", "green": "", "blue": "", "yellow": ""}}
SHAPE_Q = {"type": "choice", "instructions": "Which shape is in the image?",
           "criteria": ["circle", "square", "triangle", "none"]}
HAS_SHAPE_Q = {"type": "noul", "instructions": "The image contains a single coloured shape on a plain background."}
CLUTTER_Q = {"type": "score", "instructions": "How cluttered is the image?",
             "criteria": ["empty", "a few objects", "busy", "chaotic"]}
BREAKOUT_Q = {"type": "choice", "instructions": "You are playing Breakout. Which button should you press?",
              "criteria": {"NOOP": "do nothing", "FIRE": "launch the ball", "RIGHT": "move the paddle right",
                           "LEFT": "move the paddle left"}}


def cases() -> List[Case]:
    from PIL import Image

    red_circle = _scene(640, 480, 1, "circle", (220, 30, 30))
    blue_square = _scene(512, 512, 2, "square", (30, 60, 210))
    green_tri = _scene(300, 900, 3, "triangle", (40, 170, 60))
    rgba = _scene(400, 300, 4, "circle", (30, 60, 210)).convert("RGBA")
    a = np.array(rgba)
    a[..., 3] = np.linspace(0, 255, a.shape[1]).astype(np.uint8)[None, :]
    rgba = Image.fromarray(a, "RGBA")
    grey = _scene(333, 517, 5, "square", (20, 20, 20)).convert("L")
    palette = _scene(250, 250, 6, "circle", (230, 200, 20)).convert("P", palette=Image.Palette.ADAPTIVE, colors=16)
    big = _gradient(2600, 1100)
    huge_tall = _checker(700, 3000, 7)
    tiny = _scene(48, 36, 8, "circle", (220, 30, 30))
    thin = _checker(1200, 40, 5)

    long_note = " ".join(["The courier left the parcel at the door and the customer says the box was crushed."] * 90)
    many = {"type": "choice", "instructions": "Pick the best label for the photo.",
            "criteria": {"label_%02d" % i: "a fairly long description of visual category number %d, with extra "
                         "words so the options exceed the head budget" % i for i in range(24)}}
    long_ins = {"type": "noul", "instructions": " ".join(["Consider every region of the picture carefully."] * 60),
                "criteria": {"true": "a shape is visible", "false": "nothing is visible"}}
    ten = {"type": "score", "instructions": "Rate the image quality.", "criteria": ["level %d" % i for i in range(10)]}

    frame512 = _checker(512, 512, 16)
    cases = [
        ("lv/red_circle_png", {"image": _png(red_circle)}, {"colour": COLOUR_Q, "shape": SHAPE_Q,
                                                            "has_shape": HAS_SHAPE_Q, "clutter": CLUTTER_Q}),
        ("lv/blue_square_exact512", {"image": _png(blue_square), "note": "customer photo of the part"},
         {"colour": COLOUR_Q, "shape": SHAPE_Q}),
        ("lv/green_triangle_jpeg_tall", {"image": _jpeg(green_tri)}, {"colour": COLOUR_Q, "shape": SHAPE_Q}),
        ("lv/rgba_png", {"image": _png(rgba)}, {"colour": COLOUR_Q, "has_shape": HAS_SHAPE_Q}),
        ("lv/greyscale_png", {"image": _png(grey)}, {"colour": COLOUR_Q, "shape": SHAPE_Q}),
        ("lv/palette_png", {"image": _png(palette)}, {"colour": COLOUR_Q}),
        ("lv/two_hop_wide", {"image": _png(big), "caption": "a colour gradient"}, {"clutter": CLUTTER_Q}),
        ("lv/two_hop_tall_checker", {"image": _png(huge_tall)}, {"clutter": CLUTTER_Q, "has_shape": HAS_SHAPE_Q}),
        ("lv/upscale_tiny", {"image": _png(tiny)}, {"colour": COLOUR_Q}),
        ("lv/thin_strip", {"image": _png(thin)}, {"shape": SHAPE_Q}),
        ("lv/two_images", {"images": [_png(red_circle), _png(blue_square)], "question": "compare the two photos"},
         {"colour": COLOUR_Q, "same": {"type": "noul", "instructions": "Both images show the same shape.",
                                       "criteria": {"true": "same shape", "false": "different shapes"}}}),
        ("lv/image_and_images", {"image": _png(tiny), "images": [_png(green_tri)]}, {"shape": SHAPE_Q}),
        ("lv/no_image_text", "The parcel arrived with a torn corner but the contents look fine.",
         {"damaged": {"type": "noul", "instructions": "The item is damaged."}, "clutter": CLUTTER_Q}),
        ("lv/no_image_dict", {"subject": "Broken lamp", "body": "Arrived shattered, want a refund."},
         {"refund": {"type": "noul", "instructions": "The customer asks for a refund."}}),
        ("lv/state_truncated", {"image": _png(red_circle), "note": long_note}, {"colour": COLOUR_Q}),
        ("lv/option_budget", {"image": _png(blue_square)}, {"label": many}),
        ("lv/long_instructions", {"image": _png(green_tri)}, {"visible": long_ins}),
        ("lv/ten_levels", {"image": _png(big)}, {"quality": ten}),
        ("lv/breakout_frame", {"image": _png(_frame())}, {"action": BREAKOUT_Q}),
        ("lv/non_ascii", {"image": _png(red_circle), "nota": "El cliente dice que llegó roto 😡 — ¿reembolso?"},
         {"colour": COLOUR_Q}),
    ]
    out = [(cid, st, qs, {}) for cid, st, qs in cases]
    no_resize = {"resize": False}
    out += [
        ("lv/noresize_blue_square_png", {"image": _png(blue_square), "note": "customer photo of the part"},
         {"colour": COLOUR_Q, "shape": SHAPE_Q}, no_resize),
        ("lv/noresize_checker_png", {"image": _png(frame512)}, {"clutter": CLUTTER_Q, "has_shape": HAS_SHAPE_Q},
         no_resize),
        ("lv/noresize_jpeg", {"image": _jpeg(_scene(512, 512, 9, "triangle", (40, 170, 60)))},
         {"colour": COLOUR_Q, "shape": SHAPE_Q}, no_resize),
        ("lv/noresize_rgba_png", {"image": _png(_scene(512, 512, 10, "circle", (220, 30, 30)).convert("RGBA"))},
         {"colour": COLOUR_Q}, no_resize),
        ("lv/noresize_two_images", {"images": [_png(blue_square), _png(frame512)]},
         {"colour": COLOUR_Q, "shape": SHAPE_Q}, no_resize),
        ("lv/noresize_no_image", "A photo of a red ball on a white table.", {"colour": COLOUR_Q}, no_resize),
    ]
    return out


def rejected() -> List[Tuple[str, Any, Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]]:
    """Requests the runtime refuses: (id, state, questions, options, expected validation issue)."""
    q = {"colour": COLOUR_Q}
    red = _png(_scene(640, 480, 1, "circle", (220, 30, 30)))
    sq = _png(_scene(512, 512, 2, "square", (30, 60, 210)))
    size = lambda loc, w, h: {"loc": ["body", "state"] + loc, "type": "image_size",  # noqa: E731
                              "ctx": {"width": 512, "height": 512, "actual_width": w, "actual_height": h}}
    return [
        ("lv/reject_noresize_640x480", {"image": red}, q, {"resize": False}, size(["image"], 640, 480)),
        ("lv/reject_noresize_second_image", {"images": [sq, red]}, q, {"resize": False}, size(["images", 1], 640, 480)),
    ]


def materialize(x: Any) -> Any:
    """Wire state -> what upstream's `split_state` takes: every image data URL decoded to its file bytes."""
    def dec(v):
        if isinstance(v, str) and v.startswith("data:") and ";base64," in v:
            return base64.b64decode(v.split(",", 1)[1])
        return v

    if not isinstance(x, dict):
        return x
    out = dict(x)
    if "image" in out and out["image"] is not None:
        out["image"] = dec(out["image"])
    if "images" in out and out["images"]:
        out["images"] = [dec(v) for v in out["images"]]
    return out
