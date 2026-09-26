//! ONNX Runtime engine for `laya-vision-terminator-v1` (Laya Vision, image + text).
//!
//! Graph contract V (`docs/families/laya-vision.md`): `input_ids`, `attention_mask`,
//! `option_span`, `marker_pos`, `marker_mask`, `qtype` per row, and the request's images as
//! `pixel_values [i, 3, 512, 512]`; out come `logits` and `act_logits`. Every row carries every
//! image, so the graph runs the vision tower once per image and shares the features.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array2, Array4};
use ollaya_decision::laya_vision::{LayaVisionLayout, VisionEncoded, split_state};
use ollaya_decision::{Questions, TokenEncoder};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use serde_json::Value;

use crate::engine::{self, Engine};
use crate::image;
use crate::onnx::{Device, ModelFiles, load_tokenizer, session};
use crate::{Error, Output, QuestionOutput, RunOptions};

#[derive(Debug, Deserialize)]
struct Config {
    engine: String,
    layout: String,
    #[serde(flatten)]
    layout_config: LayaVisionLayout,
    #[serde(default = "two")]
    min_markers: usize,
}

fn two() -> usize {
    2
}

struct Tokenizer(tokenizers::Tokenizer);

impl TokenEncoder for Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, ollaya_decision::Error> {
        self.0
            .encode_fast(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
    }
}

/// A request, encoded: one row per question plus the images' pixels.
#[derive(Debug, Clone)]
pub struct Encoding {
    pub rows: Vec<VisionEncoded>,
    /// The request's images, prepared (`size`×`size` uint8), in state order.
    pub images: Vec<image::Rgb>,
    /// Tokens in the state's text, before any truncation.
    pub state_tokens: usize,
}

pub struct LayaVisionModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pub layout: LayaVisionLayout,
    min_markers: usize,
    pub device: Device,
}

impl LayaVisionModel {
    pub fn load(dir: &Path, device: Device, threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        threads: Option<usize>,
    ) -> Result<Self, Error> {
        let text = std::fs::read_to_string(&files.decision)
            .map_err(|e| Error::Model(format!("{}: {e}", files.decision.display())))?;
        let config: Config = serde_json::from_str(&text)
            .map_err(|e| Error::Model(format!("{}: {e}", files.decision.display())))?;
        if config.engine != "onnx" || config.layout != "laya-vision-terminator-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}",
                config.engine, config.layout
            )));
        }
        let tokenizer = Tokenizer(load_tokenizer(&files.tokenizer)?);
        config
            .layout_config
            .validate(&tokenizer)
            .map_err(|e| Error::Model(format!("{}: {e}", files.decision.display())))?;
        Ok(LayaVisionModel {
            session: Mutex::new(session(&files.graph, device, threads)?),
            tokenizer,
            layout: config.layout_config,
            min_markers: config.min_markers,
            device,
        })
    }

    /// Split the state, decode and prepare its images, and build every question's row.
    pub fn encode(
        &self,
        state: &Value,
        questions: &Questions,
        options: &RunOptions,
    ) -> Result<Encoding, Error> {
        let (refs, text) = split_state(state, self.layout.image.max_images)?;
        let resize = options.resize.unwrap_or(self.layout.image.resize_default);
        let images = image::prepare_all(&refs, &self.layout.image, resize)
            .map_err(ollaya_decision::Error::Image)?;
        let prefix = self.layout.prefix_ids(&self.tokenizer, images.len())?;
        let state_ids = if text.is_empty() {
            Vec::new()
        } else {
            self.tokenizer.encode(&text)?
        };
        let rows = questions
            .iter()
            .map(|(qid, q)| {
                self.layout
                    .encode(&self.tokenizer, &prefix, &state_ids, q)
                    .map_err(|e| match e {
                        ollaya_decision::Error::Invalid(m) => Error::Decision(
                            ollaya_decision::Error::Invalid(format!("question {qid:?}: {m}")),
                        ),
                        e => Error::Decision(e.for_question(qid)),
                    })
            })
            .collect::<Result<_, _>>()?;
        Ok(Encoding {
            rows,
            images,
            state_tokens: state_ids.len(),
        })
    }

    pub fn run_encoded(&self, enc: &Encoding, questions: &Questions) -> Result<Output, Error> {
        let qtypes: Vec<i64> = questions.values().map(|q| q.qtype.index() as i64).collect();
        let lens: Vec<usize> = enc.rows.iter().map(|r| r.ids.len()).collect();
        let pixels = image::pixel_values(&enc.images, self.layout.image.size);
        let mut outputs = Vec::with_capacity(enc.rows.len());
        for range in engine::batches(&lens, engine::TOKEN_BUDGET, usize::MAX) {
            outputs.extend(self.run_batch(&enc.rows[range.clone()], &qtypes[range], &pixels)?);
        }
        Ok(Output {
            questions: outputs,
            input_tokens: lens.iter().sum(),
            state_tokens: enc.state_tokens,
            state_truncated: enc.rows.iter().any(|r| r.state_truncated),
        })
    }

    fn run_batch(
        &self,
        rows: &[VisionEncoded],
        qtypes: &[i64],
        pixels: &Array4<f32>,
    ) -> Result<Vec<QuestionOutput>, Error> {
        let n = rows.len();
        let seq = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let k = rows
            .iter()
            .map(|r| r.markers.len())
            .max()
            .unwrap_or(0)
            .max(self.min_markers);
        let pad = i64::from(self.layout.special_tokens.pad);
        let mut input_ids = Array2::<i64>::from_elem((n, seq), pad);
        let mut attention = Array2::<i64>::zeros((n, seq));
        let mut span = Array2::<i64>::zeros((n, 2));
        let mut marker_pos = Array2::<i64>::zeros((n, k));
        let mut marker_mask = Array2::<bool>::from_elem((n, k), false);
        for (r, row) in rows.iter().enumerate() {
            for (c, &id) in row.ids.iter().enumerate() {
                input_ids[[r, c]] = i64::from(id);
                attention[[r, c]] = 1;
            }
            span[[r, 0]] = row.option_span.0 as i64;
            span[[r, 1]] = row.option_span.1 as i64;
            for (c, &m) in row.markers.iter().enumerate() {
                marker_pos[[r, c]] = m as i64;
                marker_mask[[r, c]] = true;
            }
        }
        let mut session = self.session.lock().expect("session mutex poisoned");
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array(input_ids)?,
            "attention_mask" => Tensor::from_array(attention)?,
            "option_span" => Tensor::from_array(span)?,
            "marker_pos" => Tensor::from_array(marker_pos)?,
            "marker_mask" => Tensor::from_array(marker_mask)?,
            "qtype" => Tensor::from_array(([n], qtypes.to_vec()))?,
            "pixel_values" => Tensor::from_array(pixels.clone())?,
        ])?;
        let logits = outputs["logits"].try_extract_array::<f32>()?;
        let act = outputs["act_logits"].try_extract_array::<f32>()?;
        Ok(rows
            .iter()
            .enumerate()
            .map(|(r, row)| QuestionOutput {
                logits: (0..row.markers.len()).map(|c| logits[[r, c]]).collect(),
                act_logits: Some(act.slice(ndarray::s![r, ..]).to_vec()),
            })
            .collect())
    }
}

impl Engine for LayaVisionModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        self.run_with(state, questions, &RunOptions::default())
    }

    fn run_with(
        &self,
        state: &Value,
        questions: &Questions,
        options: &RunOptions,
    ) -> Result<Output, Error> {
        let enc = self.encode(state, questions, options)?;
        self.run_encoded(&enc, questions)
    }
}
