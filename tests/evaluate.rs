use hound;
use quicksilver::run_inference;
use std::path::{Path, PathBuf};

/// Read a 16-bit stereo PCM WAV file into interleaved f32 samples in [-1.0, 1.0].
fn read_wav(path: &Path) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file");
    let spec = reader.spec();
    assert!(
        spec.channels == 2,
        "{}: must have 2 channels, got {} instead",
        path.display(),
        spec.channels
    );
    assert!(
        spec.bits_per_sample == 16,
        "{}: must be 16-bit audio, got {} bits per sample instead",
        path.display(),
        spec.bits_per_sample
    );
    assert!(
        spec.sample_format == hound::SampleFormat::Int,
        "{}: must be PCM audio, got {:?} instead",
        path.display(),
        spec.sample_format
    );
    let samples = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect::<Vec<f32>>();
    (samples, spec.sample_rate)
}

/// Collect and sort all .wav files in a directory.
fn collect_wavs(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", dir.display(), e))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |e| e == "wav"))
        .collect();
    files.sort();
    files
}

struct SampleResult {
    name: String,
    prob: f64,
    predicted_ai: bool,
    correct: bool,
}

fn evaluate_file(path: &Path, expected_ai: bool) -> SampleResult {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let (samples, sample_rate) = read_wav(path);
    let prob = run_inference(&samples, sample_rate).expect("Inference failed");
    let predicted_ai = prob > 0.5;
    let correct = predicted_ai == expected_ai;
    SampleResult {
        name,
        prob,
        predicted_ai,
        correct,
    }
}

fn evaluate_group(label: &str, dir: &Path, expected_ai: bool) -> Vec<SampleResult> {
    println!(
        "\n {} samples (expected: {}):",
        label,
        if expected_ai { "AI-generated" } else { "human" }
    );
    println!(" {}", "-".repeat(78));

    let files = collect_wavs(dir);
    let mut results = Vec::with_capacity(files.len());
    for file in &files {
        let r = evaluate_file(file, expected_ai);
        let mark = if r.correct { "OK " } else { "XX " };
        let verdict = if r.predicted_ai { "AI" } else { "human" };
        println!(" {} {:<50} P={:8.6}  -> {}", mark, r.name, r.prob, verdict);
        results.push(r);
    }
    results
}

#[test]
fn evaluate_samples() {
    let ai_dir = Path::new("samples/AI");
    let human_dir = Path::new("samples/Human");

    if !ai_dir.exists() || !human_dir.exists() {
        eprintln!(
            "Skipping evaluation: samples/AI or samples/Human not found \
             (looked in {})",
            std::env::current_dir().unwrap().display()
        );
        return;
    }

    println!("\n{}", "=".repeat(78));
    println!(" Quicksilver Inference Evaluation");
    println!("{}", "=".repeat(78));

    let ai_results = evaluate_group("AI", ai_dir, true);
    let human_results = evaluate_group("Human", human_dir, false);

    let ai_correct = ai_results.iter().filter(|r| r.correct).count();
    let human_correct = human_results.iter().filter(|r| r.correct).count();
    let total_correct = ai_correct + human_correct;
    let total = ai_results.len() + human_results.len();

    let ai_acc = if ai_results.is_empty() {
        0.0
    } else {
        ai_correct as f64 / ai_results.len() as f64 * 100.0
    };
    let human_acc = if human_results.is_empty() {
        0.0
    } else {
        human_correct as f64 / human_results.len() as f64 * 100.0
    };
    let overall_acc = if total == 0 {
        0.0
    } else {
        total_correct as f64 / total as f64 * 100.0
    };

    println!("\n{}", "=".repeat(78));
    println!(" Summary");
    println!("{}", "=".repeat(78));
    println!(
        " AI:      {}/{} correct  ({:.1}%)",
        ai_correct,
        ai_results.len(),
        ai_acc
    );
    println!(
        " Human:   {}/{} correct  ({:.1}%)",
        human_correct,
        human_results.len(),
        human_acc
    );
    println!(
        " Overall: {}/{} correct ({:.1}%)",
        total_correct, total, overall_acc
    );
    println!("{}\n", "=".repeat(78));

    assert!(total > 0, "No samples were evaluated");
    // Looser-than-training threshold: just sanity-check it beats random.
    assert!(
        overall_acc > 50.0,
        "Overall accuracy {overall_acc:.1}% is not better than random"
    );
}
