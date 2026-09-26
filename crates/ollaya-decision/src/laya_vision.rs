//! `laya-vision-terminator-v1`: Laya Vision (`thaitea/laya-vision`), SmolVLM-256M cut to 20
//! language layers with Laya's decision head. Rows follow upstream `laya.vlm.build_vlm_inputs`
//! and `ollaya_convert.families.laya_vision.layout`; `docs/families/laya-vision.md` is the spec.
//!
//! One row per question, each piece tokenized on its own (no special tokens) and concatenated:
//!
//! ```text
//! prefix  <|im_start|>User:  + per image  <fake_token_around_image><global-img><image>×64<fake_token_around_image>
//! state   the state's text (images removed), cut from the right to what max_len leaves
//! tail    \n<type> question: <instructions><end_of_utterance>\nAssistant: Options:\n
//!         per option: "- " + option text (inner "\n" -> " "), cut to 48 tokens, then the "\n" token
//! ```
//!
//! Each option is read at its terminating `\n`; the option span (first option token to the end of
//! the row) attends to itself both ways inside the graph.
//!
//! Images travel inside the state: a JSON object's `"image"` (one) and `"images"` (a list), in that
//! order, as `data:` URLs or bare base64. Decoding them is the runner's job; this module only
//! finds them, and says where each one is for error messages.
//!
//! Ollaya deviations, on inputs upstream renders differently or rejects:
//! 1. Non-string criterion values render as JSON (Laya's rule, shared with every Laya layout);
//!    upstream laya-vision uses Python `str()`.
//! 2. noul criteria keys match case-insensitively, as the shared question parser does.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Error;
use crate::layout::TokenEncoder;
use crate::pyjson;
use crate::question::Question;

/// A validation problem with a request's images, in the API's validation-issue shape
/// (`{loc, msg, type, ctx}`), so the daemon can pass it through as it is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageIssue {
    /// Path to the bad value: `["body", "state", "image"]` or `["body", "state", "images", n]`.
    pub loc: Vec<Value>,
    pub msg: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<Value>,
}

impl ImageIssue {
    pub fn new(loc: Vec<Value>, kind: &str, msg: impl Into<String>) -> Self {
        ImageIssue {
            loc,
            msg: msg.into(),
            kind: kind.to_owned(),
            ctx: None,
        }
    }

    pub fn with_ctx(mut self, ctx: Value) -> Self {
        self.ctx = Some(ctx);
        self
    }
}

impl std::fmt::Display for ImageIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path: Vec<String> = self
            .loc
            .iter()
            .skip(1)
            .map(|v| match v {
                Value::String(s) => s.clone(),
                v => v.to_string(),
            })
            .collect();
        write!(f, "{}: {}", path.join("."), self.msg)
    }
}

/// One image found in a state: its location and its encoded payload (a `data:` URL or base64).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageRef<'a> {
    pub loc: Vec<Value>,
    pub data: &'a str,
}

/// Split a state into its images (in order: `"image"`, then `"images"`) and its text.
///
/// Only a JSON object carries images. Its other keys are the text, serialized as a Laya state
/// (`json.dumps(ensure_ascii=False)`); with no other key the text is empty. A string state is its
/// own text and any other state is `json.dumps` of it, as for the text-only Laya models.
pub fn split_state(state: &Value, max_images: usize) -> Result<(Vec<ImageRef<'_>>, String), Error> {
    let obj = match state {
        Value::String(s) => return Ok((Vec::new(), s.clone())),
        Value::Object(m) => m,
        v => return Ok((Vec::new(), pyjson::dumps(v, false))),
    };
    let base = || vec![Value::from("body"), Value::from("state")];
    let mut images = Vec::new();
    match obj.get("image") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => images.push(ImageRef {
            loc: [base(), vec![Value::from("image")]].concat(),
            data: s,
        }),
        Some(_) => {
            return Err(Error::Image(Box::new(ImageIssue::new(
                [base(), vec![Value::from("image")]].concat(),
                "string_type",
                "Input should be a valid string: an image as a data: URL or base64",
            ))));
        }
    }
    match obj.get("images") {
        None | Some(Value::Null) => {}
        Some(Value::Array(items)) => {
            for (i, item) in items.iter().enumerate() {
                let loc = [base(), vec![Value::from("images"), Value::from(i)]].concat();
                match item {
                    Value::String(s) => images.push(ImageRef { loc, data: s }),
                    _ => {
                        return Err(Error::Image(Box::new(ImageIssue::new(
                            loc,
                            "string_type",
                            "Input should be a valid string: an image as a data: URL or base64",
                        ))));
                    }
                }
            }
        }
        Some(_) => {
            return Err(Error::Image(Box::new(ImageIssue::new(
                [base(), vec![Value::from("images")]].concat(),
                "list_type",
                "Input should be a valid list of images",
            ))));
        }
    }
    if images.len() > max_images {
        return Err(Error::Image(Box::new(
            ImageIssue::new(
                base(),
                "too_many_images",
                format!(
                    "{} images; this model takes at most {max_images} per request",
                    images.len()
                ),
            )
            .with_ctx(serde_json::json!({"max_images": max_images, "images": images.len()})),
        )));
    }
    let rest: Map<String, Value> = obj
        .iter()
        .filter(|(k, _)| k.as_str() != "image" && k.as_str() != "images")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let text = if rest.is_empty() {
        String::new()
    } else {
        pyjson::dumps(&Value::Object(rest), false)
    };
    Ok((images, text))
}

/// Prompt strings, as `decision.json` records upstream's constants.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptText {
    /// `<|im_start|>User:`
    pub prefix: String,
    /// Python format string with two `%s`: the question type, then the instructions.
    pub question: String,
    pub option_bullet: String,
    pub option_end: String,
    pub end_of_utterance: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageTokens {
    pub fake: String,
    pub global: String,
    pub image: String,
}

/// The `image` block of `decision.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageConfig {
    /// Side of the square the vision tower reads (512).
    pub size: usize,
    /// First resize hop: longest edge to this (2048).
    pub stage1_longest_edge: usize,
    /// `<image>` tokens per image (64).
    pub image_seq_len: usize,
    pub tokens: ImageTokens,
    pub image_token_id: u32,
    pub max_images: usize,
    /// Resize images unless the request says otherwise.
    #[serde(default = "yes")]
    pub resize_default: bool,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionTokens {
    pub pad: u32,
    pub option_end: u32,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct LayaVisionLayout {
    pub max_len: usize,
    pub head_max_len: usize,
    pub option_max_tokens: usize,
    pub text: PromptText,
    pub image: ImageConfig,
    pub special_tokens: VisionTokens,
}

/// One question's row.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionEncoded {
    pub ids: Vec<u32>,
    /// Position of each option's terminating `\n`, in option order.
    pub markers: Vec<usize>,
    /// `[first option token, end of row)`.
    pub option_span: (usize, usize),
    /// The state's text did not fit and was cut (its beginning is kept).
    pub state_truncated: bool,
}

/// Options leave at least this many tokens of `head_max_len` to the question.
const MIN_HEAD_ROOM: usize = 16;
/// The question is never cut below this many tokens.
const MIN_QUESTION_TOKENS: usize = 8;
/// When options overflow the budget, each keeps at least this many tokens.
const MIN_TOKENS_PER_OPTION: usize = 4;

impl LayaVisionLayout {
    /// Check the declared strings are the ones this code was written against.
    pub fn validate(&self, enc: &dyn TokenEncoder) -> Result<(), Error> {
        if self.text.question.matches("%s").count() != 2 {
            return Err(Error::invalid(
                "decision.json: text.question must hold two %s (type, instructions)",
            ));
        }
        if enc.encode(&self.text.option_end)? != [self.special_tokens.option_end] {
            return Err(Error::invalid(
                "decision.json: the option terminator must be one token",
            ));
        }
        Ok(())
    }

    /// The row prefix for `n_images` images, tokenized once as one string.
    pub fn prefix_ids(&self, enc: &dyn TokenEncoder, n_images: usize) -> Result<Vec<u32>, Error> {
        let t = &self.image.tokens;
        let run = format!(
            "{}{}{}{}",
            t.fake,
            t.global,
            t.image.repeat(self.image.image_seq_len),
            t.fake
        );
        enc.encode(&format!("{}{}", self.text.prefix, run.repeat(n_images)))
    }

    fn question_text(&self, q: &Question) -> String {
        let mut parts = self.text.question.splitn(3, "%s");
        let (a, b, c) = (
            parts.next().unwrap_or_default(),
            parts.next().unwrap_or_default(),
            parts.next().unwrap_or_default(),
        );
        let ins = q.instructions.replace(&self.text.end_of_utterance, " ");
        format!("{a}{}{b}{ins}{c}", q.qtype.name())
    }

    /// One question's row: `prefix` from [`Self::prefix_ids`], `state_ids` the state's text
    /// tokenized once for every question.
    pub fn encode(
        &self,
        enc: &dyn TokenEncoder,
        prefix: &[u32],
        state_ids: &[u32],
        q: &Question,
    ) -> Result<VisionEncoded, Error> {
        let options = q.render_options();
        let end = self.special_tokens.option_end;
        let mut opt_ids = Vec::with_capacity(options.len());
        for o in &options {
            let text = format!(
                "{}{}",
                self.text.option_bullet,
                o.replace(&self.text.option_end, " ")
            );
            let mut ids = enc.encode(&text)?;
            ids.truncate(self.option_max_tokens);
            opt_ids.push(ids);
        }
        let mut head = enc.encode(&self.question_text(q))?;

        // Budgets are signed: options can overflow head_max_len before they are cut.
        let head_max = self.head_max_len as isize;
        let used = |o: &[Vec<u32>]| o.iter().map(|x| x.len() as isize + 1).sum::<isize>();
        let mut budget = head_max - used(&opt_ids);
        if budget < MIN_HEAD_ROOM as isize {
            let per = ((self.head_max_len.saturating_sub(MIN_HEAD_ROOM)) / opt_ids.len().max(1))
                .saturating_sub(1)
                .max(MIN_TOKENS_PER_OPTION);
            for o in &mut opt_ids {
                o.truncate(per);
            }
            budget = head_max - used(&opt_ids);
        }
        let keep = budget.max(MIN_QUESTION_TOKENS as isize) as usize;
        if head.len() > keep {
            // Keep both ends: the tail carries the "Assistant: Options:" cue.
            let front = keep / 2;
            let back = keep - front;
            let tail = head[head.len() - back..].to_vec();
            head.truncate(front);
            head.extend(tail);
        }

        let mut tail = head;
        let span_start = tail.len();
        let mut markers = Vec::with_capacity(opt_ids.len());
        for o in &opt_ids {
            tail.extend_from_slice(o);
            tail.push(end);
            markers.push(tail.len() - 1);
        }
        if prefix.len() + tail.len() > self.max_len {
            return Err(Error::invalid(format!(
                "the question and its options do not fit next to the images in the model's {}-token context ({} tokens of images and prompt)",
                self.max_len,
                prefix.len()
            )));
        }
        let room = self.max_len - prefix.len() - tail.len();
        let state = &state_ids[..room.min(state_ids.len())];
        let off = prefix.len() + state.len();
        let mut ids = Vec::with_capacity(off + tail.len());
        ids.extend_from_slice(prefix);
        ids.extend_from_slice(state);
        let tail_len = tail.len();
        ids.extend(tail);
        Ok(VisionEncoded {
            ids,
            markers: markers.into_iter().map(|m| m + off).collect(),
            option_span: (span_start + off, tail_len + off),
            state_truncated: state.len() < state_ids.len(),
        })
    }

    /// Tokens one image adds to a row: the `<image>` run and its framing tokens.
    pub fn image_tokens(&self) -> usize {
        self.image.image_seq_len + 3
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// One token per character, `<...>` special tokens as one token each.
    struct Chars;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            let mut out = Vec::new();
            let mut rest = text;
            while let Some(c) = rest.chars().next() {
                if c == '<'
                    && let Some(end) = rest.find('>')
                {
                    let tok = &rest[..=end];
                    out.push(match tok {
                        "<image>" => 49190,
                        "<fake_token_around_image>" => 49189,
                        "<global-img>" => 49152,
                        "<end_of_utterance>" => 49279,
                        "<|im_start|>" => 1,
                        _ => 7,
                    });
                    rest = &rest[end + 1..];
                    continue;
                }
                out.push(c as u32);
                rest = &rest[c.len_utf8()..];
            }
            Ok(out)
        }
    }

    fn layout() -> LayaVisionLayout {
        serde_json::from_value(json!({
            "max_len": 300,
            "head_max_len": 256,
            "option_max_tokens": 48,
            "text": {"prefix": "<|im_start|>User:", "question": "\n%s question: %s<end_of_utterance>\nAssistant: Options:\n",
                     "option_bullet": "- ", "option_end": "\n", "end_of_utterance": "<end_of_utterance>"},
            "image": {"size": 512, "stage1_longest_edge": 2048, "image_seq_len": 4,
                      "tokens": {"fake": "<fake_token_around_image>", "global": "<global-img>", "image": "<image>"},
                      "image_token_id": 49190, "max_images": 3},
            "special_tokens": {"pad": 2, "option_end": 10}
        }))
        .unwrap()
    }

    fn q(v: Value) -> Question {
        Question::parse("q", &v).unwrap()
    }

    #[test]
    fn splits_images_in_order() {
        let state =
            json!({"note": "héllo", "image": "data:a", "images": ["data:b", "data:c"], "n": 1});
        let (imgs, text) = split_state(&state, 3).unwrap();
        assert_eq!(
            imgs.iter().map(|i| i.data).collect::<Vec<_>>(),
            ["data:a", "data:b", "data:c"]
        );
        assert_eq!(
            imgs[2].loc,
            json!(["body", "state", "images", 1])
                .as_array()
                .unwrap()
                .clone()
        );
        assert_eq!(text, "{\"note\": \"héllo\", \"n\": 1}");
        // Images alone: empty text. null image keys are no images.
        let count = |s: Value| {
            let (imgs, text) = split_state(&s, 3).unwrap();
            (imgs.len(), text)
        };
        assert_eq!(count(json!({"image": "x"})), (1, String::new()));
        assert_eq!(
            count(json!({"image": null, "images": null, "a": "b"})),
            (0, "{\"a\": \"b\"}".into())
        );
        // Strings and arrays carry no images.
        assert_eq!(count(json!("plain")), (0, "plain".into()));
        assert_eq!(
            count(json!([{"image": "x"}])),
            (0, "[{\"image\": \"x\"}]".into())
        );
    }

    #[test]
    fn bad_images_are_issues() {
        let issue = |s: Value| match split_state(&s, 2) {
            Err(Error::Image(i)) => i,
            other => panic!("{other:?}"),
        };
        let i = issue(json!({"image": 3}));
        assert_eq!((i.kind.as_str(), i.loc.len()), ("string_type", 3));
        let i = issue(json!({"images": ["ok", {"x": 1}]}));
        assert_eq!(i.kind, "string_type");
        assert_eq!(
            i.loc,
            json!(["body", "state", "images", 1])
                .as_array()
                .unwrap()
                .clone()
        );
        assert_eq!(issue(json!({"images": "one"})).kind, "list_type");
        let i = issue(json!({"image": "a", "images": ["b", "c"]}));
        assert_eq!(i.kind, "too_many_images");
        assert_eq!(
            i.to_string(),
            "state: 3 images; this model takes at most 2 per request"
        );
    }

    #[test]
    fn row_layout() {
        let l = layout();
        let prefix = l.prefix_ids(&Chars, 2).unwrap();
        // "<|im_start|>" + "User:" + 2 × (fake, global, 4 × image, fake)
        assert_eq!(prefix.len(), 1 + 5 + 2 * 7);
        assert_eq!(prefix.iter().filter(|&&t| t == 49190).count(), 8);
        let state = Chars.encode("ab").unwrap();
        let e = l
            .encode(
                &Chars,
                &prefix,
                &state,
                &q(json!({"type": "choice", "instructions": "Pick<end_of_utterance>", "criteria": {"x": "", "y": "why\nnot"}})),
            )
            .unwrap();
        let text: String = e.ids[prefix.len()..]
            .iter()
            .map(|&t| char::from_u32(t).filter(|_| t < 49000).unwrap_or('@'))
            .collect();
        assert_eq!(
            text,
            "ab\nchoice question: Pick @\nAssistant: Options:\n- x\n- y: why not\n"
        );
        let rel: Vec<usize> = e.markers.iter().map(|m| m - prefix.len() - 2).collect();
        let head = "\nchoice question: Pick @\nAssistant: Options:\n"
            .chars()
            .count();
        // "- x" then its "\n"; "- y: why not" then its "\n".
        assert_eq!(rel, [head + 3, head + 3 + 1 + 12]);
        assert_eq!(e.option_span, (prefix.len() + 2 + head, e.ids.len()));
        assert!(e.markers.iter().all(|&m| e.ids[m] == 10));
        assert!(!e.state_truncated);
    }

    #[test]
    fn budgets_and_truncation() {
        let l = layout();
        let prefix = l.prefix_ids(&Chars, 1).unwrap();
        // The state keeps its beginning; the row fills max_len exactly.
        let state = Chars.encode(&"s".repeat(500)).unwrap();
        let noul = q(json!({"type": "noul", "instructions": "ok?"}));
        let e = l.encode(&Chars, &prefix, &state, &noul).unwrap();
        assert_eq!(e.ids.len(), l.max_len);
        assert!(e.state_truncated);
        // Options over the budget are all cut to the same length, then the question keeps its ends.
        let many: Map<String, Value> = (0..20)
            .map(|i| (format!("option-{i:02}-{}", "z".repeat(40)), Value::from("")))
            .collect();
        let long = format!("{}END", "w".repeat(400));
        let e = l
            .encode(
                &Chars,
                &prefix,
                &[],
                &q(json!({"type": "choice", "instructions": long, "criteria": many})),
            )
            .unwrap();
        let per = (256 - 16) / 20 - 1;
        let lens: Vec<usize> = e.markers.windows(2).map(|w| w[1] - w[0] - 1).collect();
        assert!(lens.iter().all(|&n| n == per), "{lens:?}");
        let head = &e.ids[prefix.len()..e.option_span.0];
        // 20 options of `per` tokens plus a terminator each leave 256 - 20 * (per + 1) = 16.
        assert_eq!(head.len(), 256 - 20 * (per + 1));
        assert_eq!(char::from_u32(*head.last().unwrap()), Some('\n'));
        // A question that cannot fit next to the images is an error.
        let prefix3 = l.prefix_ids(&Chars, 3).unwrap();
        let tiny = LayaVisionLayout { max_len: 60, ..l };
        assert!(tiny.encode(&Chars, &prefix3, &[], &noul).is_err());
    }
}
