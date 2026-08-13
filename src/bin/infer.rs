use quicksilver::{audio::decode_audio, run_inference};

fn main() {
    let input_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --release --bin infer -- <input.wav|input.mp3>");
        std::process::exit(2);
    });

    let (samples, sample_rate) =
        decode_audio(std::path::Path::new(&input_path)).expect("Failed to decode audio file");

    let prob = run_inference(&samples, sample_rate).expect("Inference failed");
    let verdict = if prob > 0.5 { "AI-generated" } else { "human" };
    println!("Input:       {}", input_path);
    println!("Sample rate: {} Hz", sample_rate);
    println!("P(AI-generated) = {:.6}", prob);
    println!("Verdict: {} (threshold 0.5)", verdict);
}
