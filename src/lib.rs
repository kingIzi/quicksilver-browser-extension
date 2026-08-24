pub mod audio;
pub mod core;
pub mod inference;

pub use inference::run_inference;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyString};

use crate::audio::decode_audio_source;
use serde_json::json;

#[pyfunction]
fn info(py: Python) -> PyObject {
    let data = json!({
        "status": "ok",
        "count": 3
    });

    // Convert serde_json::Value → Python dict
    let dict = PyDict::new(py);
    for (k, v) in data.as_object().unwrap() {
        dict.set_item(k, v.to_string()).unwrap();
    }

    dict.into()
}

/// Detect whether an audio file is AI-generated or human ("MGM" = Made by
/// Generative Music).
///
/// `file` may be a local path (`*.wav`, `*.mp3`, `*.m4a`/`*.mp4`) or an
/// `http(s)://` URL ending in one of those extensions. The
/// model is embedded directly in the compiled extension via `include_bytes!`,
/// so no external model file needs to ship alongside the Python package.
///
/// Returns a dict with:
///   - `"prob"`:    float in `[0.0, 1.0]` — `P(AI-generated)`
///   - `"verdict"`: `"AI"` if `prob > 0.5`, else `"Human"`
#[pyfunction]
fn detect_mgm(file: &str, py: Python) -> PyResult<PyObject> {
    // Run the (potentially long) decode + inference outside the GIL so that
    // other Python threads can proceed and KeyboardInterrupt can interrupt.
    let (prob, source, sample_rate) = py.allow_threads(|| {
        let decoded = decode_audio_source(file).map_err(PyValueError::new_err)?;
        let prob =
            run_inference(&decoded.samples, decoded.sample_rate).map_err(PyValueError::new_err)?;
        let source = decoded.source.clone();
        let sample_rate = decoded.sample_rate;
        drop(decoded);
        Ok::<_, PyErr>((prob, source, sample_rate))
    })?;

    let dict = PyDict::new(py);
    let file_name = file[0..file.rfind(".").unwrap_or(file.len())].to_string();
    dict.set_item("type", "file")?;
    dict.set_item("filename", file_name)?;
    dict.set_item("confidence", PyFloat::new(py, prob))?;
    dict.set_item(
        "label",
        PyString::new(py, if prob > 0.5 { "AI" } else { "Human" }),
    )?;
    // Provenance metadata, useful for debugging without a separate call.
    dict.set_item("source", PyString::new(py, &source))?;
    dict.set_item("sample_rate", sample_rate)?;
    Ok(dict.into())
}

// "type": "file",
// "filename": file.filename,
// "label": label,
// "confidence": confidence,

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_function(wrap_pyfunction!(detect_mgm, m)?)?;
    Ok(())
}
