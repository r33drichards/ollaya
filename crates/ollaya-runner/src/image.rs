//! Image preprocessing for image-input models, bit-exact with the checkpoint's processor.
//!
//! Laya Vision's processor (`Idefics3ImageProcessor`, torchvision backend, on the CPU) does, per
//! image: decode, convert to RGB, resize the longest edge to 2048 (Lanczos), resize to 512×512
//! (Lanczos), normalize `(v − 127.5) / 127.5`. Both resizes are torchvision's native uint8 path,
//! which ATen implements with Pillow-SIMD's fixed-point convolution:
//!
//! * separable, horizontal pass first, the intermediate rounded to uint8;
//! * an axis whose size does not change is not resampled;
//! * Lanczos-3 weights per output pixel, centred at `(i + 0.5) · scale`, support
//!   `3 · max(scale, 1)`, clipped at the borders and normalised, then quantized to integers at
//!   the largest precision (≤ 22 bits) that keeps the biggest weight within int16;
//! * `acc = 2^(p−1) + Σ px·w`, output `clamp(acc >> p, 0, 255)`.
//!
//! JPEG is decoded by libjpeg (see [`decode_jpeg`]); PNG, WebP and GIF by the pure-Rust decoders.
//!
//! `docs/families/laya-vision.md` has the measurements: this reproduces torch on random images at
//! every size tried, and every golden image's pixels by sha256.

use base64::Engine as _;
use ollaya_decision::laya_vision::{ImageConfig, ImageIssue, ImageRef};
use serde_json::{Value, json};

/// Refuse images over this many pixels before decoding them (decompression bombs).
pub const MAX_PIXELS: u64 = 50_000_000;

/// An 8-bit RGB image, rows top to bottom, pixels interleaved (HWC).
#[derive(Debug, Clone, PartialEq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// An image problem, boxed: it travels inside `ollaya_decision::Error::Image`.
pub type Issue = Box<ImageIssue>;

fn issue(loc: &[Value], kind: &str, msg: impl Into<String>) -> Issue {
    Box::new(ImageIssue::new(loc.to_vec(), kind, msg))
}

fn issue_ctx(loc: &[Value], kind: &str, msg: impl Into<String>, ctx: Value) -> Issue {
    Box::new(ImageIssue::new(loc.to_vec(), kind, msg).with_ctx(ctx))
}

/// The encoded bytes of a `data:` URL (`data:<type>;base64,<payload>`) or of bare base64.
pub fn payload(data: &str, loc: &[Value]) -> Result<Vec<u8>, Issue> {
    let b64 = match data.strip_prefix("data:") {
        Some(rest) => match rest.split_once(',') {
            Some((meta, body)) if meta.ends_with(";base64") => body,
            _ => {
                return Err(issue(
                    loc,
                    "image_data",
                    "a data: URL must be base64-encoded (data:<type>;base64,<data>)",
                ));
            }
        },
        None => data,
    };
    base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| issue(loc, "image_data", format!("not valid base64: {e}")))
}

/// Decode an image file and convert it to RGB as PIL's `convert("RGB")` does: alpha dropped,
/// greyscale replicated, palettes expanded. EXIF orientation is not applied (upstream does not).
pub fn decode(bytes: &[u8], loc: &[Value]) -> Result<Rgb, Issue> {
    use image::ImageFormat;
    let format = image::guess_format(bytes).ok().filter(|f| {
        matches!(
            f,
            ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif
        )
    });
    let Some(format) = format else {
        return Err(issue(
            loc,
            "image_type",
            "not a PNG, JPEG, WebP or GIF image",
        ));
    };
    if format == ImageFormat::Jpeg {
        return decode_jpeg(bytes, loc);
    }
    let reader = || image::ImageReader::with_format(std::io::Cursor::new(bytes), format);
    let (w, h) = reader()
        .into_dimensions()
        .map_err(|e| issue(loc, "image_decode", format!("cannot decode the image: {e}")))?;
    too_large(loc, u64::from(w), u64::from(h))?;
    let mut r = reader();
    r.no_limits();
    let img = r
        .decode()
        .map_err(|e| issue(loc, "image_decode", format!("cannot decode the image: {e}")))?;
    let rgb = img.to_rgb8();
    Ok(Rgb {
        width: rgb.width() as usize,
        height: rgb.height() as usize,
        data: rgb.into_raw(),
    })
}

fn too_large(loc: &[Value], w: u64, h: u64) -> Result<(), Issue> {
    if w * h > MAX_PIXELS {
        return Err(issue_ctx(
            loc,
            "image_too_large",
            format!("the image is {w}x{h}; at most {MAX_PIXELS} pixels are accepted"),
            json!({"max_pixels": MAX_PIXELS, "actual_width": w, "actual_height": h}),
        ));
    }
    Ok(())
}

/// JPEG through libjpeg (mozjpeg's build of libjpeg-turbo's decoder): the Rust decoders' IDCT and
/// chroma upsampling differ from libjpeg-turbo's by up to 3 levels, and upstream decodes with
/// libjpeg-turbo (PIL). Default libjpeg settings, as PIL uses them: ISLOW IDCT, fancy upsampling.
/// libjpeg reports a fatal error by unwinding, which is caught here and becomes an issue.
fn decode_jpeg(bytes: &[u8], loc: &[Value]) -> Result<Rgb, Issue> {
    let bad = |e: String| issue(loc, "image_decode", format!("cannot decode the image: {e}"));
    let run = || -> std::io::Result<Result<Rgb, Issue>> {
        let d = mozjpeg::Decompress::new_mem(bytes)?;
        let (w, h) = d.size();
        if let Err(e) = too_large(loc, w as u64, h as u64) {
            return Ok(Err(e));
        }
        if matches!(
            d.color_space(),
            mozjpeg::ColorSpace::JCS_CMYK | mozjpeg::ColorSpace::JCS_YCCK
        ) {
            return Ok(Err(issue(
                loc,
                "image_decode",
                "CMYK JPEG images are not supported; convert the image to RGB",
            )));
        }
        let mut started = d.rgb()?;
        let data = started.read_scanlines::<u8>()?;
        started.finish()?;
        Ok(Ok(Rgb {
            width: w,
            height: h,
            data,
        }))
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => Err(bad(e.to_string())),
        Err(payload) => Err(bad(payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| "libjpeg error".into())
            .trim_start_matches("libjpeg fatal error: ")
            .to_owned())),
    }
}

/// The processor's first hop: the longest edge to `longest`, the other edge `int()` of the
/// scaled size, rounded up to even (`_resize_output_size_rescale_to_max_len`).
pub fn stage1_size(height: usize, width: usize, longest: usize) -> (usize, usize) {
    let aspect = width as f64 / height as f64;
    let (h, w) = if width >= height {
        let h = (longest as f64 / aspect) as usize;
        (h + h % 2, longest)
    } else {
        let w = (longest as f64 * aspect) as usize;
        (longest, w + w % 2)
    };
    (h.max(1), w.max(1))
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

fn lanczos3(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Integer kernels for one axis: per output pixel, the first input index and its weights.
struct Kernels {
    starts: Vec<usize>,
    weights: Vec<Vec<i32>>,
    precision: u32,
}

fn kernels(in_size: usize, out_size: usize) -> Kernels {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 3.0 * filterscale;
    let ss = 1.0 / filterscale;
    let mut starts = Vec::with_capacity(out_size);
    let mut float = Vec::with_capacity(out_size);
    let mut max_w = 0.0f64;
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        // Python `int()` truncates toward zero; the bounds are then clamped to the image.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
        let mut k: Vec<f64> = (xmin..xmax)
            .map(|x| lanczos3((x as f64 - center + 0.5) * ss))
            .collect();
        let total: f64 = k.iter().sum();
        if total != 0.0 {
            for w in &mut k {
                *w /= total;
            }
        }
        for &w in &k {
            max_w = max_w.max(w);
        }
        starts.push(xmin);
        float.push(k);
    }
    // The largest precision whose biggest weight still fits in int16.
    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + max_w * f64::from(1u32 << (precision + 1))) as i64;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let one = f64::from(1u32 << precision);
    let weights = float
        .into_iter()
        .map(|k| {
            k.into_iter()
                .map(|w| {
                    if w < 0.0 {
                        (-0.5 + w * one) as i32
                    } else {
                        (0.5 + w * one) as i32
                    }
                })
                .collect()
        })
        .collect();
    Kernels {
        starts,
        weights,
        precision,
    }
}

#[inline]
fn clip(acc: i64, precision: u32) -> u8 {
    (acc >> precision).clamp(0, 255) as u8
}

fn horizontal(src: &Rgb, out_w: usize) -> Rgb {
    let k = kernels(src.width, out_w);
    let round = 1i64 << k.precision.saturating_sub(1);
    let mut data = vec![0u8; src.height * out_w * 3];
    for y in 0..src.height {
        let row = &src.data[y * src.width * 3..(y + 1) * src.width * 3];
        let out = &mut data[y * out_w * 3..(y + 1) * out_w * 3];
        for x in 0..out_w {
            let (start, w) = (k.starts[x], &k.weights[x]);
            let mut acc = [round; 3];
            for (j, &wj) in w.iter().enumerate() {
                let p = &row[(start + j) * 3..(start + j) * 3 + 3];
                for c in 0..3 {
                    acc[c] += i64::from(p[c]) * i64::from(wj);
                }
            }
            for c in 0..3 {
                out[x * 3 + c] = clip(acc[c], k.precision);
            }
        }
    }
    Rgb {
        width: out_w,
        height: src.height,
        data,
    }
}

fn vertical(src: &Rgb, out_h: usize) -> Rgb {
    let k = kernels(src.height, out_h);
    let round = 1i64 << k.precision.saturating_sub(1);
    let stride = src.width * 3;
    let mut data = vec![0u8; out_h * stride];
    let mut acc = vec![0i64; stride];
    for y in 0..out_h {
        let (start, w) = (k.starts[y], &k.weights[y]);
        acc.fill(round);
        for (j, &wj) in w.iter().enumerate() {
            let row = &src.data[(start + j) * stride..(start + j + 1) * stride];
            for (a, &p) in acc.iter_mut().zip(row) {
                *a += i64::from(p) * i64::from(wj);
            }
        }
        for (o, &a) in data[y * stride..(y + 1) * stride].iter_mut().zip(&acc) {
            *o = clip(a, k.precision);
        }
    }
    Rgb {
        width: src.width,
        height: out_h,
        data,
    }
}

/// torchvision's `resize(uint8, [out_h, out_w], LANCZOS, antialias=True)` on the CPU.
pub fn resize(src: &Rgb, out_h: usize, out_w: usize) -> Rgb {
    let mut img = if out_w != src.width {
        horizontal(src, out_w)
    } else {
        src.clone()
    };
    if out_h != img.height {
        img = vertical(&img, out_h);
    }
    img
}

/// A decoded image as the vision tower reads it: `size`×`size` uint8 pixels.
///
/// With `resize` the processor's two hops run; without it the image must already be
/// `size`×`size` and is used as it is (the processor with `do_resize=False`).
pub fn prepare(
    img: &Rgb,
    config: &ImageConfig,
    resize_images: bool,
    loc: &[Value],
) -> Result<Rgb, Issue> {
    let s = config.size;
    if !resize_images {
        if (img.width, img.height) != (s, s) {
            return Err(issue_ctx(
                loc,
                "image_size",
                format!(
                    "the image is {}x{}; with resize off it must be {s}x{s}",
                    img.width, img.height
                ),
                json!({"width": s, "height": s, "actual_width": img.width, "actual_height": img.height}),
            ));
        }
        return Ok(img.clone());
    }
    let (h, w) = stage1_size(img.height, img.width, config.stage1_longest_edge);
    Ok(resize(&resize(img, h, w), s, s))
}

/// Decode and prepare every image of a request, in order.
pub fn prepare_all(
    images: &[ImageRef<'_>],
    config: &ImageConfig,
    resize_images: bool,
) -> Result<Vec<Rgb>, Issue> {
    images
        .iter()
        .map(|r| {
            let bytes = payload(r.data, &r.loc)?;
            let img = decode(&bytes, &r.loc)?;
            prepare(&img, config, resize_images, &r.loc)
        })
        .collect()
}

/// `pixel_values [n, 3, size, size]` for the graph: `(v − 127.5) / 127.5` in f32, channels first.
/// No images gives one all-zero image, which no row reads.
pub fn pixel_values(images: &[Rgb], size: usize) -> ndarray::Array4<f32> {
    let n = images.len().max(1);
    let mut out = ndarray::Array4::<f32>::zeros((n, 3, size, size));
    for (i, img) in images.iter().enumerate() {
        for y in 0..size {
            for x in 0..size {
                let p = &img.data[(y * size + x) * 3..(y * size + x) * 3 + 3];
                for c in 0..3 {
                    out[[i, c, y, x]] = (f32::from(p[c]) - 127.5) / 127.5;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc() -> Vec<Value> {
        vec![json!("body"), json!("state"), json!("image")]
    }

    fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb(f(x, y)));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn stage1_sizes_match_the_processor() {
        // (h, w) -> processor output, from `_resize_output_size_rescale_to_max_len`.
        assert_eq!(stage1_size(480, 640, 2048), (1536, 2048));
        assert_eq!(stage1_size(900, 300, 2048), (2048, 682));
        assert_eq!(stage1_size(210, 160, 2048), (2048, 1560));
        assert_eq!(stage1_size(1100, 2600, 2048), (866, 2048));
        assert_eq!(stage1_size(40, 1200, 2048), (68, 2048));
        assert_eq!(stage1_size(512, 512, 2048), (2048, 2048));
    }

    #[test]
    fn resize_is_the_identity_at_the_same_size_and_keeps_flat_colour() {
        let flat = Rgb {
            width: 7,
            height: 5,
            data: [10u8, 200, 77].repeat(35),
        };
        assert_eq!(resize(&flat, 5, 7), flat);
        // Lanczos weights sum to one, so a flat image stays flat both ways.
        for (h, w) in [(20, 3), (2, 50), (512, 512)] {
            let r = resize(&flat, h, w);
            assert!(r.data.chunks(3).all(|p| p == [10, 200, 77]), "{h}x{w}");
        }
    }

    #[test]
    fn kernel_precision_fits_int16() {
        for (i, o) in [(64, 128), (100, 40), (2048, 512), (512, 2048), (3000, 2048)] {
            let k = kernels(i, o);
            let max = k.weights.iter().flatten().copied().max().unwrap();
            // The largest weight fits int16, and one more bit of precision would not.
            assert!(max < 1 << 15, "{i}->{o}");
            assert!(
                k.precision == 22 || i64::from(max) * 2 >= (1 << 15) - 1,
                "{i}->{o}"
            );
            assert!(k.precision > 10);
        }
    }

    #[test]
    fn decode_payloads_and_errors() {
        let bytes = png(3, 2, |x, y| [x as u8, y as u8, 9]);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        for data in [format!("data:image/png;base64,{b64}"), b64.clone()] {
            let raw = payload(&data, &loc()).unwrap();
            let img = decode(&raw, &loc()).unwrap();
            assert_eq!((img.width, img.height), (3, 2));
            assert_eq!(&img.data[..6], &[0, 0, 9, 1, 0, 9]);
        }
        let kind = |r: Result<Vec<u8>, Issue>| r.unwrap_err().kind;
        assert_eq!(kind(payload("data:image/png,raw", &loc())), "image_data");
        assert_eq!(kind(payload("not base64!", &loc())), "image_data");
        let e = decode(b"hello there", &loc()).unwrap_err();
        assert_eq!(e.kind, "image_type");
        assert_eq!(e.loc, loc());
        let e = decode(&bytes[..20], &loc()).unwrap_err();
        assert_eq!(e.kind, "image_decode");
        // A JPEG header with no image behind it is an issue, not a crash in libjpeg.
        let e = decode(
            &[0xff, 0xd8, 0xff, 0xe0, 0, 16, b'J', b'F', b'I', b'F'],
            &loc(),
        )
        .unwrap_err();
        assert_eq!(e.kind, "image_decode");
    }

    #[test]
    fn resize_off_needs_the_input_size() {
        let config: ImageConfig = serde_json::from_value(json!({
            "size": 4, "stage1_longest_edge": 16, "image_seq_len": 1,
            "tokens": {"fake": "f", "global": "g", "image": "i"}, "image_token_id": 1, "max_images": 2
        }))
        .unwrap();
        let four = Rgb {
            width: 4,
            height: 4,
            data: (0..48).map(|v| v as u8).collect(),
        };
        assert_eq!(prepare(&four, &config, false, &loc()).unwrap(), four);
        let wide = Rgb {
            width: 6,
            height: 4,
            data: vec![0; 72],
        };
        let e = prepare(&wide, &config, false, &loc()).unwrap_err();
        assert_eq!(e.kind, "image_size");
        assert_eq!(
            e.ctx,
            Some(json!({"width": 4, "height": 4, "actual_width": 6, "actual_height": 4}))
        );
        let r = prepare(&wide, &config, true, &loc()).unwrap();
        assert_eq!((r.width, r.height), (4, 4));
        // Normalisation is the processor's `(v - 127.5) / 127.5`, channels first.
        let pv = pixel_values(&[four], 4);
        assert_eq!(pv.shape(), &[1, 3, 4, 4]);
        assert_eq!(pv[[0, 0, 0, 0]], -1.0);
        assert_eq!(pv[[0, 1, 0, 0]], (1.0 - 127.5) / 127.5);
        assert_eq!(pv[[0, 0, 0, 1]], (3.0 - 127.5) / 127.5);
        assert_eq!(pixel_values(&[], 4).shape(), &[1, 3, 4, 4]);
    }
}
