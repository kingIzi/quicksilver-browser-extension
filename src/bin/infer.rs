use quicksilver::{audio::decode_audio_source, run_inference};

fn main() {
    let input = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --release --bin infer -- <input.wav|input.mp3|http(s)://...>");
        std::process::exit(2);
    });

    let decoded = decode_audio_source(&input).expect("Failed to decode audio");
    let prob = run_inference(&decoded.samples, decoded.sample_rate).expect("Inference failed");
    // Drop the decoded audio now so any backing temp file (for URL sources) is
    // removed before we print the verdict.
    let source = decoded.source.clone();
    let sample_rate = decoded.sample_rate;
    drop(decoded);

    let verdict = if prob > 0.5 { "AI-generated" } else { "human" };
    println!("Input:       {}", source);
    println!("Sample rate: {} Hz", sample_rate);
    println!("P(AI-generated) = {:.6}", prob);
    println!("Verdict: {} (threshold 0.5)", verdict);
}
