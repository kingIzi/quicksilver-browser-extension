pub mod audio;
pub mod core;
pub mod inference;

pub use inference::run_inference;

#[cfg(not(target_family = "wasm"))]
pub mod pyo3_port;

#[cfg(target_family = "wasm")]
pub mod wasm_port;
