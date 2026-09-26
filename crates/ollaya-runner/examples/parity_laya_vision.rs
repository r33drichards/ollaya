//! Compare the `laya-vision-terminator-v1` runtime against goldens from
//! `ollaya_convert.families.laya_vision.goldens` (upstream laya-vision with its backbone in
//! float64, one unpadded row at a time; see `docs/families/laya-vision.md`).
//!
//!     cargo run --release -p ollaya-runner --example parity_laya_vision -- <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]
//!
//! Every case is encoded first, before anything runs through the graph:
//! * the state's text (images removed) must match;
//! * every image, decoded and prepared (resized, or not with `options.resize = false`), must have
//!   the golden's pixels, bit for bit (sha256 of the 512×512×3 uint8 array);
//! * per question: token ids, marker positions and the option span must match exactly.
//!
//! Then every case runs through the graph, all rows of a request batched as the runtime runs them:
//! * option logits and act logits must be within `LOGIT_TOL`;
//! * calibrated probabilities must pick the reference's option (max difference reported).
//!
//! Requests the goldens mark as refused must fail with the golden's validation issue.
//!
//! `--latency` then times full requests (decode, resize, tokenize, forward) per case.

use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ollaya_decision::{Calibration, CalibrationFile, parse_questions};
use ollaya_runner::laya_vision::LayaVisionModel;
use ollaya_runner::{Device, Engine, RunOptions};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Largest raw-logit difference accepted (ONNX Runtime in fp32 vs the backbone in float64).
const LOGIT_TOL: f64 = 1e-3;

fn max_diff(a: &[f32], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).abs())
        .fold(0.0, f64::max)
}

fn argmax(p: &[f64]) -> usize {
    p.iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
        )
        .0
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: parity_laya_vision <model-dir> <goldens.jsonl> [cpu|cuda] [--latency]");
    }
    let device = match args.get(3).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let latency = args.iter().any(|a| a == "--latency");
    let dir = PathBuf::from(&args[1]);
    let t = Instant::now();
    let model = LayaVisionModel::load(&dir, device, None)?;
    println!("load {:.1}s on {device:?}", t.elapsed().as_secs_f64());
    let calibration: CalibrationFile =
        serde_json::from_str(&std::fs::read_to_string(dir.join("calibration.json"))?)?;
    let calibration = Calibration::from_file(&calibration);

    let file = std::fs::File::open(&args[2]).with_context(|| args[2].clone())?;
    let mut records = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        records.push(serde_json::from_str::<Value>(&line?)?);
    }

    let (mut text_bad, mut pixels_bad, mut pixels_total, mut rows_bad, mut rows_total) =
        (0, 0, 0, 0, 0);
    let (mut rejected_ok, mut rejected_total) = (0, 0);
    let (mut logit_max, mut act_max, mut prob_max) = (0.0f64, 0.0f64, 0.0f64);
    let (mut agree, mut logit_bad) = (0, 0);
    let mut timed = Vec::new();
    for rec in &records {
        let id = rec["id"].as_str().unwrap_or("?");
        let state = &rec["state"];
        let questions = parse_questions(&rec["questions"]).with_context(|| id.to_owned())?;
        let options: RunOptions = serde_json::from_value(rec["options"].clone())?;

        if let Some(err) = rec.get("error") {
            rejected_total += 1;
            let want = &err["detail"][0];
            match model.encode(state, &questions, &options) {
                Err(ollaya_runner::Error::Decision(ollaya_decision::Error::Image(issue))) => {
                    let got = serde_json::to_value(&issue)?;
                    let same = got["loc"] == want["loc"]
                        && got["type"] == want["type"]
                        && got["ctx"] == want["ctx"];
                    rejected_ok += usize::from(same);
                    if !same {
                        println!("REJECT {id}: got {got}, want {want}");
                    }
                }
                other => println!("REJECT {id}: not refused as expected: {:?}", other.err()),
            }
            continue;
        }

        let enc = model
            .encode(state, &questions, &options)
            .with_context(|| format!("{id}: the runtime rejects a request upstream accepts"))?;
        let (_, text) = ollaya_decision::laya_vision::split_state(state, 16)?;
        if text != rec["state_text"].as_str().unwrap_or_default() {
            text_bad += 1;
            println!("TEXT {id}");
        }
        let gold_images = rec["images"].as_array().context("images")?;
        if gold_images.len() != enc.images.len() {
            bail!(
                "{id}: {} images vs {} in the goldens",
                enc.images.len(),
                gold_images.len()
            );
        }
        for (i, (img, gold)) in enc.images.iter().zip(gold_images).enumerate() {
            pixels_total += 1;
            let sha = hex::encode(Sha256::digest(&img.data));
            if Some(sha.as_str()) != gold["sha256"].as_str() {
                pixels_bad += 1;
                let mean: Vec<f64> = (0..3)
                    .map(|c| {
                        img.data
                            .iter()
                            .skip(c)
                            .step_by(3)
                            .map(|&v| f64::from(v))
                            .sum::<f64>()
                            / (img.data.len() / 3) as f64
                    })
                    .collect();
                println!(
                    "PIXELS {id} image {i}: channel means {mean:.4?} vs {}",
                    gold["mean"]
                );
            }
        }
        let items = rec["items"].as_array().context("items")?;
        for (row, item) in enc.rows.iter().zip(items) {
            rows_total += 1;
            let ids: Vec<u32> = serde_json::from_value(item["ids"].clone())?;
            let markers: Vec<usize> = serde_json::from_value(item["markers"].clone())?;
            let span: (usize, usize) = serde_json::from_value(item["option_span"].clone())?;
            if row.ids != ids || row.markers != markers || row.option_span != span {
                rows_bad += 1;
                println!(
                    "ROW {id} {}: len {} vs {}, first differing token {:?}",
                    item["qid"],
                    row.ids.len(),
                    ids.len(),
                    row.ids.iter().zip(&ids).position(|(a, b)| a != b)
                );
            }
        }

        let t = Instant::now();
        let out = model.run_encoded(&enc, &questions)?;
        timed.push((id.to_owned(), t.elapsed()));
        for ((q, got), item) in questions.values().zip(&out.questions).zip(items) {
            let gold: Vec<f64> = serde_json::from_value(item["logits"].clone())?;
            let gold_act: Vec<f64> = serde_json::from_value(item["act_logits"].clone())?;
            let d = max_diff(&got.logits, &gold);
            let da = max_diff(got.act_logits.as_deref().unwrap_or_default(), &gold_act);
            logit_max = logit_max.max(d);
            act_max = act_max.max(da);
            if d > LOGIT_TOL || da > LOGIT_TOL {
                logit_bad += 1;
                println!("LOGITS {id} {}: {d:.2e} / act {da:.2e}", item["qid"]);
            }
            let p = calibration.probabilities(q.qtype, &got.logits, out.state_tokens);
            let gold_p: Vec<f64> = serde_json::from_value(item["probabilities"].clone())?;
            let dp = p
                .iter()
                .zip(&gold_p)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            prob_max = prob_max.max(dp);
            if argmax(&p) == argmax(&gold_p) {
                agree += 1;
            } else {
                println!("ARGMAX {id} {}", item["qid"]);
            }
        }
    }

    println!(
        "state text:            {}/{} match",
        records.len() - rejected_total - text_bad,
        records.len() - rejected_total
    );
    println!(
        "image pixels (sha256): {}/{pixels_total} bit-exact",
        pixels_total - pixels_bad
    );
    println!(
        "rows (ids, markers, span): {}/{rows_total} identical",
        rows_total - rows_bad
    );
    println!("argmax agreement:      {agree}/{rows_total}");
    println!(
        "option logits max |d|: {logit_max:.2e} (tolerance {LOGIT_TOL:.0e}), act logits {act_max:.2e}"
    );
    println!("probabilities max |d|: {prob_max:.2e}");
    println!("refused as expected:   {rejected_ok}/{rejected_total}");

    if latency {
        println!("latency (encode + forward, one request each):");
        for rec in records.iter().filter(|r| r.get("error").is_none()) {
            let questions = parse_questions(&rec["questions"])?;
            let options: RunOptions = serde_json::from_value(rec["options"].clone())?;
            let t = Instant::now();
            model.run_with(&rec["state"], &questions, &options)?;
            let pre = Instant::now();
            model.encode(&rec["state"], &questions, &options)?;
            println!(
                "  {:32} {:7.1} ms (of which encode {:6.1} ms)",
                rec["id"].as_str().unwrap_or("?"),
                t.elapsed().as_secs_f64() * 1e3 - pre.elapsed().as_secs_f64() * 1e3,
                pre.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    if text_bad + pixels_bad + rows_bad + logit_bad > 0
        || agree != rows_total
        || rejected_ok != rejected_total
    {
        bail!("parity FAILED");
    }
    println!("parity OK");
    Ok(())
}
