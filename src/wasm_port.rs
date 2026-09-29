use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::audio::decode_audio_bytes;
use crate::inference::run_inference;

/// Payload for [`info`]. Typed structs serialize to plain JS objects, whereas
/// `serde_json::Value` would come out as a JS `Map` (which is awkward to use
/// from the JS side: `result.label` vs `result.get("label")`).
#[derive(Serialize)]
struct Info {
    status: &'static str,
    count: u32,
}

#[wasm_bindgen]
pub fn info() -> JsValue {
    serde_wasm_bindgen::to_value(&Info {
        status: "ok",
        count: 3,
    })
    .expect("failed to serialize info payload")
}

/// Result payload for [`detect_mgm`] — mirrors the Python dict shape
/// (`type` / `filename` / `label` / `confidence` / `sample_rate`).
#[derive(Serialize)]
struct Detection {
    r#type: &'static str,
    filename: String,
    label: &'static str,
    confidence: f64,
    sample_rate: u32,
}

/// Detect whether an audio file is AI-generated or human ("MGM" = Made by
/// Generative Music) — the JS counterpart of the Python `detect_mgm()`.
///
/// `bytes` is the raw file content (`.wav`, `.mp3` or `.m4a`/`.mp4`); there is
/// no filesystem or libcurl in the browser, so fetch/read the data on the JS
/// side and pass it in. `filename` picks the decoder from its extension and
/// is echoed back (without extension, like the Python version) in the result.
///
/// Returns `{ type, filename, label, confidence, sample_rate }` where
/// `confidence` is `P(AI-generated)` in `[0.0, 1.0]` and `label` is `"AI"`
/// when `confidence > 0.5`, else `"Human"`.
#[wasm_bindgen]
pub fn detect_mgm(bytes: &[u8], filename: &str) -> Result<JsValue, JsValue> {
    let ext = filename
        .rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default();

    let (samples, sample_rate) =
        decode_audio_bytes(bytes, &ext).map_err(|e| JsValue::from_str(&e))?;
    let prob = run_inference(&samples, sample_rate).map_err(|e| JsValue::from_str(&e))?;

    let file_name = match filename.rfind('.') {
        Some(i) => &filename[..i],
        None => filename,
    };
    serde_wasm_bindgen::to_value(&Detection {
        r#type: "file",
        filename: file_name.to_string(),
        label: if prob > 0.5 { "AI" } else { "Human" },
        confidence: prob,
        sample_rate,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}
