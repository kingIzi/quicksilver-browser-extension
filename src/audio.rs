use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// The number of channels the inference pipeline expects.
const EXPECTED_CHANNELS: u16 = 2;

/// Returns `true` if `src` is an HTTP(S) URL that should be fetched remotely
/// rather than treated as a local file path.
pub fn is_url_source(src: &str) -> bool {
    src.starts_with("http://") || src.starts_with("https://")
}

/// The result of decoding an audio source.
///
/// `source` is a human-readable label for the input (the local path or the
/// URL) for display purposes.
#[derive(Debug)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub source: String,
    /// Backing temp file for URL sources; deleted when this is dropped.
    /// `None` for local files (which are not ours to remove).
    _temp: Option<TempGuard>,
}

/// Decode audio from a local path **or** an `http(s)://` URL into interleaved
/// f32 samples in `[-1.0, 1.0]`.
///
/// For URL sources the content is downloaded to a temporary file, decoded,
/// and the file is deleted once the returned [`DecodedAudio`] is dropped (so
/// it is cleaned up even if inference later panics). `.wav`, `.mp3` and
/// `.m4a`/`.mp4` are supported.
pub fn decode_audio_source(src: &str) -> Result<DecodedAudio, String> {
    if is_url_source(src) {
        let guard = download_to_temp(src)?;
        let (samples, sample_rate) = decode_audio(&guard.path)?;
        Ok(DecodedAudio {
            samples,
            sample_rate,
            source: src.to_string(),
            _temp: Some(guard),
        })
    } else {
        let path = PathBuf::from(src);
        let (samples, sample_rate) = decode_audio(&path)?;
        Ok(DecodedAudio {
            samples,
            sample_rate,
            source: src.to_string(),
            _temp: None,
        })
    }
}

/// Decode a local audio file into interleaved f32 samples in `[-1.0, 1.0]`.
///
/// `.wav` (via `hound`), `.mp3` and `.m4a`/`.mp4` (AAC, via `symphonia`) are
/// supported. Mono files are upmixed to stereo and files with more than two
/// channels are downmixed to the first two channels, so inference always
/// receives the interleaved stereo PCM that
/// [`run_inference`](crate::run_inference) requires.
///
/// Returns `(samples, sample_rate)` on success, or an error string describing
/// what went wrong (the caller decides whether to panic or propagate).
pub fn decode_audio(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "wav" => decode_wav(path),
        "mp3" => decode_symphonia(path, "mp3"),
        // AAC in an MP4 container (e.g. Mux `audio.m4a` static renditions).
        "m4a" | "m4b" | "mp4" => decode_symphonia(path, "mp4"),
        other => Err(format!(
            "Unsupported audio format '.{other}': only .wav, .mp3 and .m4a/.mp4 are supported"
        )),
    }
}

/// Download a URL to a temporary file on disk, returning a guard whose `Drop`
/// deletes the file. The extension is inferred from the URL path (defaulting
/// to `.bin`) so that [`decode_audio`] can dispatch on it.
fn download_to_temp(url: &str) -> Result<TempGuard, String> {
    // Decide a sensible extension from the URL's path component.
    let ext = url
        .split(['?', '#'])
        .next()
        .unwrap_or(url)
        .rsplit('/')
        .next()
        .and_then(|name| name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()))
        .filter(|e| matches!(e.as_str(), "wav" | "mp3" | "m4a" | "m4b" | "mp4"))
        .unwrap_or_else(|| "bin".to_string());

    let mut easy = curl::easy::Easy::new();
    easy.url(url)
        .map_err(|e| format!("Invalid URL '{url}': {e}"))?;
    // Follow HTTP redirects (e.g. short links / CDN hops).
    easy.follow_location(true)
        .map_err(|e| format!("Failed to enable redirects: {e}"))?;
    // A conservative cap so a bad URL can't hang the process forever.
    easy.connect_timeout(std::time::Duration::from_secs(30))
        .ok();
    easy.timeout(std::time::Duration::from_secs(300))
        .ok();

    let path = temp_path(&ext);
    let mut file = std::fs::File::create(&path).map_err(|e| {
        format!("Failed to create temp file {}: {e}", path.display())
    })?;

    {
        use std::io::Write;
        let mut transfer = easy.transfer();
        transfer
            .write_function(move |data: &[u8]| {
                if file.write_all(data).is_err() {
                    // Signal write failure back to libcurl.
                    Ok(0)
                } else {
                    Ok(data.len())
                }
            })
            .map_err(|e| format!("Failed to set up download: {e}"))?;
        transfer
            .perform()
            .map_err(|e| format!("Failed to download '{url}': {e}"))?;
    }

    let status = easy
        .response_code()
        .map_err(|e| format!("Failed to read HTTP status: {e}"))?;
    if !(200..300).contains(&status) {
        // Best-effort cleanup before surfacing the error.
        let _ = std::fs::remove_file(&path);
        return Err(format!("HTTP {status} when downloading '{url}'"));
    }

    Ok(TempGuard { path })
}

/// Allocate a unique path under the system temp dir.
fn temp_path(ext: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut name = format!("quicksilver-{pid}-{ts}-{id}.{ext}");
    // Replace any chars that are illegal in file names on some platforms.
    name = name.replace(|c: char| matches!(c, '/' | '\\'), "-");
    std::env::temp_dir().join(name)
}

/// RAII guard for a temp file: deletes the file on drop.
struct TempGuard {
    path: PathBuf,
}

impl std::fmt::Debug for TempGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TempGuard")
            .field("path", &self.path)
            .finish()
    }
}

impl Drop for TempGuard {
    fn drop(&mut self) {
        // Best-effort; ignore errors (file may already be gone).
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Decode a PCM WAV file via `hound`.
fn decode_wav(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open WAV file {}: {e}", path.display()))?;
    let spec = reader.spec();

    let samples = reader
        .samples::<i16>()
        .map(|s| match s {
            Ok(v) => v as f32 / i16::MAX as f32,
            Err(e) => panic!("Failed to read WAV sample from {}: {e}", path.display()),
        })
        .collect::<Vec<f32>>();

    let samples = coerce_to_stereo(&samples, spec.channels);
    Ok((samples, spec.sample_rate))
}

/// Decode a compressed audio file via `symphonia` (MP3, or AAC inside an MP4
/// container such as `.m4a`), copying the samples out as interleaved f32 in
/// `[-1.0, 1.0]`.
///
/// `hint_ext` names the container/extension to hint to the format probe so
/// the right demuxer is selected without content sniffing.
fn decode_symphonia(path: &Path, hint_ext: &str) -> Result<(Vec<f32>, u32), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Hint the extension so the format probe picks the demuxer directly.
    let mut hint = Hint::new();
    hint.with_extension(hint_ext);

    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| format!("Failed to probe {}: {e}", path.display()))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| format!("{}: no decodable audio track found", path.display()))?;

    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| format!("{}: audio codec parameters missing", path.display()))?;

    let sample_rate = audio_params
        .sample_rate
        .ok_or_else(|| format!("{}: unknown sample rate", path.display()))?;
    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .ok_or_else(|| format!("{}: unknown channel layout", path.display()))? as u16;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| format!("Failed to create decoder for {}: {e}", path.display()))?;

    let track_id = track.id;
    let mut samples: Vec<f32> = Vec::new();
    let mut packet_buf: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(symphonia::core::errors::Error::ResetRequired) => continue,
            Err(e) => {
                return Err(format!("Failed to read packet from {}: {e}", path.display()))
            }
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => {
                return Err(format!("Failed to decode packet from {}: {e}", path.display()))
            }
        };

        // Copy out as interleaved f32; symphonia handles sample-format conversion.
        packet_buf.clear();
        packet_buf.resize(decoded.samples_interleaved(), f32::MID);
        decoded.copy_to_slice_interleaved(&mut packet_buf);
        samples.extend_from_slice(&packet_buf);
    }

    Ok((coerce_to_stereo(&samples, channels), sample_rate))
}

/// Reshape interleaved audio to exactly two channels.
///
/// Mono is duplicated into both channels; layouts with more than two channels
/// keep only the first two (a rough downmix that is sufficient for this
/// model's feature extraction).
fn coerce_to_stereo(interleaved: &[f32], channels: u16) -> Vec<f32> {
    match channels {
        0 => panic!("audio has 0 channels"),
        1 => {
            let mut out = Vec::with_capacity(interleaved.len() * 2);
            for &s in interleaved {
                out.push(s);
                out.push(s);
            }
            out
        }
        2 => interleaved.to_vec(),
        n => {
            let n = n as usize;
            let mut out = Vec::with_capacity((interleaved.len() / n) * EXPECTED_CHANNELS as usize);
            for chunk in interleaved.chunks_exact(n) {
                out.push(chunk[0]);
                out.push(chunk[1]);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_extension_returns_error() {
        let err = decode_audio(Path::new("foo.flac")).unwrap_err();
        assert!(
            err.contains("Unsupported audio format '.flac'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_file_returns_open_error() {
        let err = decode_audio(Path::new("nonexistent.mp3")).unwrap_err();
        assert!(err.contains("Failed to open"), "unexpected error: {err}");
    }

    #[test]
    fn mono_is_upmixed_to_stereo() {
        let mono = vec![0.1, 0.2, 0.3];
        let stereo = coerce_to_stereo(&mono, 1);
        assert_eq!(stereo, vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
    }

    #[test]
    fn stereo_passes_through_unchanged() {
        let stereo_in = vec![0.1, 0.2, 0.3, 0.4];
        let stereo_out = coerce_to_stereo(&stereo_in, 2);
        assert_eq!(stereo_out, stereo_in);
    }

    #[test]
    fn five_channel_is_downmixed_to_first_two() {
        // Two frames of 5-channel audio.
        let surround = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stereo = coerce_to_stereo(&surround, 5);
        assert_eq!(stereo, vec![1.0, 2.0, 6.0, 7.0]);
    }

    #[test]
    fn is_url_source_detects_http_and_https() {
        assert!(is_url_source("http://example.com/a.mp3"));
        assert!(is_url_source("https://example.com/a.wav"));
        assert!(!is_url_source("ftp://example.com/a.mp3"));
        assert!(!is_url_source("samples/AI/track-1.wav"));
        assert!(!is_url_source("/abs/path/to/file.mp3"));
        assert!(!is_url_source("file.mp3"));
    }

    #[test]
    fn url_with_unknown_scheme_is_treated_as_local_path() {
        // ftp/ file paths flow into the local decoder branch, which surfaces a
        // clean error rather than a network attempt.
        let err = decode_audio_source("not-a-real-file.wav").unwrap_err();
        assert!(err.contains("No such file") || err.contains("failed to open") || err.contains("Failed to open"),
            "unexpected error: {err}");
    }
}
