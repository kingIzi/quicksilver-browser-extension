use hound;
use quicksilver::run_inference;

fn main() {
    let input_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --release --bin infer -- <input.wav>");
        std::process::exit(2);
    });

    let mut reader = hound::WavReader::open(&input_path).expect("Failed to open WAV file");
    let spec = reader.spec();
    assert!(
        spec.channels == 2,
        "must have 2 channels, got {} instead",
        spec.channels
    );
    assert!(
        spec.bits_per_sample == 16,
        "must be 16-bit audio, got {} bits per sample instead",
        spec.bits_per_sample
    );
    assert!(
        spec.sample_format == hound::SampleFormat::Int,
        "must be PCM audio, got {:?} instead",
        spec.sample_format
    );

    let samples = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect::<Vec<f32>>();

    let prob = run_inference(&samples, spec.sample_rate).expect("Inference failed");
    let verdict = if prob > 0.5 { "AI-generated" } else { "human" };
    println!("Input:       {}", input_path);
    println!("Sample rate: {} Hz", spec.sample_rate);
    println!("P(AI-generated) = {:.6}", prob);
    println!("Verdict: {} (threshold 0.5)", verdict);
}
