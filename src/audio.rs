use std::path::Path;
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// The number of channels the inference pipeline expects.
const EXPECTED_CHANNELS: u16 = 2;

/// Decode an audio file into interleaved f32 samples in `[-1.0, 1.0]`.
///
/// Both `.wav` and `.mp3` are supported. Mono files are upmixed to stereo and
/// files with more than two channels are downmixed to the first two channels,
/// so inference always receives the interleaved stereo PCM that
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
        "mp3" => decode_mp3(path),
        other => Err(format!(
            "Unsupported audio format '.{other}': only .wav and .mp3 are supported"
        )),
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

/// Decode an MP3 file via `symphonia`, copying the samples out as interleaved
/// f32 in `[-1.0, 1.0]`.
fn decode_mp3(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Hint the extension so the format probe picks the MP3 demuxer directly.
    let mut hint = Hint::new();
    hint.with_extension("mp3");

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
}
