use anyhow::{Context, Result, bail};
use clap::Parser;
use dirs::home_dir;
use hound;
use regex::Regex;
use std::{
    collections::HashSet,
    ffi::OsStr,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::{Builder as TempBuilder, NamedTempFile};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

const ROOTS: &[&str] = &[
    "fuck", "shit", "bitch", "bastard", "cunt", "dick", "ass", "asshole", "jackass", "piss",
    "slut", "whore", "fag", "faggot", "damn", "goddamn", "hell", "motherfucker", "mf", "douche",
    "douchebag", "prick", "twat", "bollocks",
];

const SUFFIXES: &[&str] = &[
    "", "s", "es", "ed", "ing", "er", "ers", "y", "ies", "head", "hole", "face",
];

fn build_badwords() -> HashSet<String> {
    let mut set = HashSet::new();
    for root in ROOTS {
        for suf in SUFFIXES {
            set.insert(format!("{}{}", root, suf));
        }
    }
    set
}

fn is_swear(word: &str, badwords: &HashSet<String>, re_strip: &Regex) -> bool {
    // Use the pre-compiled regex passed as an argument
    // .to_ascii_lowercase() returns a String
    let tok: String = re_strip.replace_all(word, "").to_ascii_lowercase();
    // Pass tok.as_str() to contains, which expects &Q where String: Borrow<Q>
    // String implements Borrow<str>, so Q can be str.
    !tok.is_empty() && badwords.contains(tok.as_str())
}

#[derive(Debug, Clone)]
struct Word {
    start: f32,
    end: f32,
    text: String,
}

#[derive(Parser, Debug)]
#[command(version, about = "Batch-mute profanity in MKV/MP4 with Whisper", long_about = None)]
struct Args {
    /// Input files
    inputs: Vec<PathBuf>,

    /// Silence padding before/after swear (ms)
    #[arg(long = "pad-ms", default_value_t = 200)]
    pad_ms: u32,

    /// Replace original instead of creating *_censored file
    #[arg(long)]
    overwrite: bool,

    /// Export auto-generated subtitles (SRT)
    #[arg(long)]
    srt: bool,

    /// Path to GGML model file (e.g. ggml-large-v3.bin)
    #[arg(long, default_value = "ggml-large-v3.bin")]
    model: String,

    /// Beam search depth
    #[arg(long, default_value_t = 5)]
    beam_size: u32,

    /// ffmpeg threads
    #[arg(long = "ffmpeg-threads", default_value_t = 8)]
    ffmpeg_threads: u32,
}

fn timestamp(t: f32) -> String {
    let ms_total = (t * 1000.0).round() as u64;
    let (ms, s_total) = (ms_total % 1000, ms_total / 1000);
    let (s, m_total) = (s_total % 60, s_total / 60);
    let (m, h) = (m_total % 60, m_total / 60);
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

// Pass the pre-compiled regex as an argument
fn collect_ranges(
    words: &[Word],
    pad: f32,
    badwords: &HashSet<String>,
    re_strip: &Regex,
) -> Vec<(f32, f32, String)> {
    let mut raw: Vec<(f32, f32, String)> = words
        .iter()
        .filter(|w| is_swear(&w.text, badwords, re_strip)) // Use passed regex
        .map(|w| (f32::max(0.0, w.start - pad), w.end + pad, w.text.clone()))
        .collect();
    if raw.is_empty() {
        return vec![];
    }
    // Sorting f32: Assuming timestamps are not NaN.
    raw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged = vec![raw[0].clone()];
    for (s, e, w) in raw.into_iter().skip(1) {
        let last = merged.last_mut().unwrap(); // Should be safe due to non-empty raw
        if s <= last.1 + 0.05 { // Merge if start is within 50ms of last's end
            last.1 = last.1.max(e);
            last.2.push_str(&format!(",{}", w));
        } else {
            merged.push((s, e, w));
        }
    }
    merged
}

fn vol_filter(ranges: &[(f32, f32, String)]) -> Option<String> {
    if ranges.is_empty() {
        None
    } else {
        Some(
            ranges
                .iter()
                .map(|(s, e, _)| format!("volume=enable='between(t,{:.3},{:.3})':volume=0", s, e))
                .collect::<Vec<_>>()
                .join(","),
        )
    }
}

fn write_logs(
    out_dir: &Path,
    stem: &str, // Expecting a valid UTF-8 stem
    ranges: &[(f32, f32, String)],
) -> Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(out_dir).context(format!("Failed to create log directory {:?}", out_dir))?;
    let tsv_name = format!("{}_censored.tsv", stem);
    let csv_name = format!("{}_censored.csv", stem);
    let tsv_path = out_dir.join(tsv_name);
    let csv_path = out_dir.join(csv_name);

    let mut ft = File::create(&tsv_path)
        .context(format!("Failed to create TSV file {:?}", tsv_path))?;
    let mut wtr = csv::Writer::from_path(&csv_path)
        .context(format!("Failed to create CSV writer for {:?}", csv_path))?;

    for (s, e, w) in ranges {
        writeln!(ft, "{:.3}\t{:.3}\t{}", s, e, w)
            .context(format!("Failed to write to TSV file {:?}", tsv_path))?;
        wtr.write_record(&[format!("{:.3}", s), format!("{:.3}", e), w.clone()])
            .context(format!("Failed to write record to CSV file {:?}", csv_path))?;
    }
    wtr.flush().context(format!("Failed to flush CSV writer for {:?}", csv_path))?;
    Ok((tsv_path, csv_path))
}

fn transcribe(wav: &Path, ctx: &WhisperContext, beam_size: u32) -> Result<Vec<Word>> {
    let samples_i16: Vec<i16> = {
        // rdr does not need to be mutable as .spec() takes &self and .into_samples() takes self
        let rdr = hound::WavReader::open(wav)
            .with_context(|| format!("Failed to open WAV file: {:?}", wav))?;
        let spec = rdr.spec();
        if spec.channels != 1 || spec.sample_rate != 16_000 {
            bail!(
                "Expected 16-kHz mono WAV, got {} Hz / {} ch from {:?}",
                spec.sample_rate,
                spec.channels,
                wav
            );
        }
        rdr.into_samples::<i16>()
            .collect::<Result<Vec<i16>, hound::Error>>()
            .with_context(|| format!("Failed to read samples from WAV file: {:?}", wav))?
    };

    // Convert i16 samples to f32 samples
    let samples: Vec<f32> = samples_i16.iter().map(|&s| s as f32 / 32768.0).collect();


    let mut state = ctx.create_state().context("Failed to create whisper state")?;
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: beam_size as i32, // u32 to i32 is safe for typical beam sizes
        patience: 1.0, // Default patience from whisper.cpp, can be configurable
    });
    params.set_token_timestamps(true);
    params.set_print_special(false);
    params.set_print_progress(true); // Enable progress printing
    params.set_print_realtime(true); // Enable realtime (live timestamp) printing
    // Consider adding language detection or allowing language specification if needed
    // params.set_language(Some("en"));


    state.full(params, &samples).context("Whisper inference failed")?;

    // The live timestamps are printed by whisper_rs itself when print_realtime is true.
    // The following block prints the *final* segments after the whole inference is done.
    // You might want to keep it for a final summary or remove/comment it if the live output is sufficient.
    let num_segments = state.full_n_segments().context("Failed to get segment count")?;
    println!("\n--- Final Transcription Segments ---"); // Added a newline for better separation from live output
    for i in 0..num_segments {
        let segment_text = state.full_get_segment_text(i).context(format!("Failed to get text for segment {}", i))?;
        let t0 = state.full_get_segment_t0(i).context(format!("Failed to get t0 for segment {}", i))? as f32 * 0.01;
        let t1 = state.full_get_segment_t1(i).context(format!("Failed to get t1 for segment {}", i))? as f32 * 0.01;
        println!("[{} --> {}] {}", timestamp(t0), timestamp(t1), segment_text.trim());
    }
    println!("----------------------------------");


    let mut words = Vec::new();
    for i in 0..num_segments {
        let num_tokens = state.full_n_tokens(i).context(format!("Failed to get token count for segment {}", i))?;
        for j in 0..num_tokens {
            if let Ok(token_data) = state.full_get_token_data(i, j) {
                 // Skip special tokens like <|nospeech|> etc.
                if token_data.id >= ctx.token_eot() { // Assuming special tokens are at/after EOT. Check whisper-rs docs for specifics.
                    continue;
                }
                let text = state.full_get_token_text(i, j).context(format!("Failed to get text for token {} in segment {}",j,i))?;
                // Filter out tokens that are just spaces or start with special characters like Whisper's timestamp tokens
                let trimmed_text = text.trim();
                if trimmed_text.is_empty() || trimmed_text.starts_with("<|") {
                    continue;
                }

                let start_time = token_data.t0 as f32 * 0.01;
                let end_time = token_data.t1 as f32 * 0.01;
                
                words.push(Word {
                    start: start_time,
                    end: end_time,
                    text: trimmed_text.to_string(),
                });
            } else {
                 eprintln!("Warning: Could not retrieve token data for segment {}, token {}", i, j);
            }
        }
    }
    Ok(words)
}

fn extract_wav(src: &Path) -> Result<(NamedTempFile, PathBuf)> {
    let tmp_wav_file = TempBuilder::new()
        .prefix("audio_extract_")
        .suffix(".wav")
        .tempfile()
        .context("Failed to create temporary WAV file")?;
    let wav_path = tmp_wav_file.path().to_path_buf();

    // Use chained .arg() for clarity and type safety with paths
    let status = Command::new("ffmpeg")
        .arg("-loglevel").arg("error") // Be less verbose by default
        .arg("-y") // Overwrite output files without asking
        .arg("-i").arg(src) // Input file
        .arg("-vn") // No video output
        .arg("-ac").arg("1") // Mono audio
        .arg("-ar").arg("16000") // 16kHz sample rate
        .arg("-c:a").arg("pcm_s16le") // PCM signed 16-bit little-endian audio codec
        .arg(&wav_path) // Output to the temporary WAV path
        .status()
        .context(format!("ffmpeg command failed during audio extraction from {:?}", src))?;

    if !status.success() {
        bail!("ffmpeg failed to extract audio from {:?}. Exit status: {}", src, status);
    }
    Ok((tmp_wav_file, wav_path))
}

fn mute_video(src: &Path, dst: &Path, filt: Option<String>, threads: u32) -> Result<()> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-loglevel").arg("error")
       .arg("-y")
       .arg("-i").arg(src);

    if let Some(f_val) = filt {
        // Using -filter_complex for volume filter to avoid issues with simple -af
        // This also allows for more complex filter graphs in the future if needed.
        cmd.arg("-filter_complex").arg(&f_val)
           .arg("-c:v").arg("copy"); // Copy video stream
           // If audio codec needs to be specified, e.g., AAC:
           // .arg("-c:a").arg("aac")
           // .arg("-b:a").arg("192k"); // Example bitrate
           // For now, let ffmpeg choose or copy if possible. If issues arise, specify codec.
    } else {
        // If no filter, just copy everything
        cmd.arg("-c").arg("copy");
    }

    cmd.arg("-threads").arg(threads.to_string())
       .arg(dst); // Output file

    let status = cmd.status().context(format!("ffmpeg command failed during muting of {:?}", src))?;
    if !status.success() {
        bail!("ffmpeg failed while muting {:?}. Outputting to {:?}. Exit status: {}", src, dst, status);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let pad_s = args.pad_ms as f32 / 1000.0;

    // Compile regex once
    let re_strip = Regex::new(r"[^\w']").context("Failed to compile word stripping regex")?;

    let log_dir_base = home_dir().context("Could not determine home directory")?;
    let log_dir = log_dir_base.join("profanity_muter_logs"); // More descriptive log folder name
    fs::create_dir_all(&log_dir).context(format!("Failed to create log directory at {:?}", log_dir))?;


    let badwords = build_badwords();

    println!("Loading Whisper model from: {}", args.model);
    let whisper_params = whisper_rs::WhisperContextParameters::default(); // Use default params for now
    let ctx = WhisperContext::new_with_params(&args.model, whisper_params)
        .with_context(|| format!("Failed to load Whisper model from '{}'", args.model))?;
    println!("Whisper model loaded successfully.");


    for vid_path in &args.inputs {
        println!(
            "\n→ Processing: {}",
            vid_path.file_name().and_then(OsStr::to_str).unwrap_or("<invalid_filename>")
        );

        let (_wav_tmp_file, wav_path) = extract_wav(vid_path)
            .with_context(|| format!("Failed extracting WAV from {:?}", vid_path))?;
        
        println!("  Transcribing audio from: {:?}", wav_path);
        let words = transcribe(&wav_path, &ctx, args.beam_size)
            .with_context(|| format!("Transcription failed for {:?}", wav_path))?;
        println!("  Found {} words in transcription.", words.len());

        let ranges_to_mute = collect_ranges(&words, pad_s, &badwords, &re_strip);

        let current_file_stem_os = vid_path.file_stem()
            .ok_or_else(|| anyhow::anyhow!("Input file {:?} must have a stem.", vid_path))?;
        let current_file_stem_str = current_file_stem_os.to_string_lossy();

        let current_file_ext_str = vid_path.extension()
            .and_then(OsStr::to_str) // Keep it as &str if valid UTF-8
            .map(|e| format!(".{}", e))
            .unwrap_or_else(String::new); // Empty string if no extension or not UTF-8

        let (output_path_final, output_path_temp_processing) = if args.overwrite {
            // When overwriting, process to a temporary file, then rename.
            let temp_file_name = format!("{}.tmp_censored{}", current_file_stem_str, current_file_ext_str);
            (vid_path.clone(), vid_path.with_file_name(temp_file_name))
        } else {
            // When not overwriting, create a new file with _censored suffix.
            let censored_file_name = format!("{}_censored{}", current_file_stem_str, current_file_ext_str);
            let final_target_path = vid_path.with_file_name(censored_file_name);
            // ffmpeg will write directly to this path.
            (final_target_path.clone(), final_target_path)
        };
        
        if ranges_to_mute.is_empty() {
            println!("  No profanity found; copying original stream if not overwriting to same file.");
            // If not overwriting and no swears, we might not need to do anything unless the output path is different.
            // The current mute_video logic with None filter handles stream copy.
            // If args.overwrite is true and ranges_to_mute is empty, output_path_temp_processing is different from vid_path.
            // We'd be copying vid_path to output_path_temp_processing, then renaming. This is redundant.
            if args.overwrite && output_path_temp_processing.file_name() == vid_path.file_name() {
                 println!("  Overwrite specified and no swears, no action needed for {:?}.", vid_path);
            } else if !args.overwrite && output_path_final == *vid_path {
                 println!("  No overwrite, no swears, and output is same as input. No action needed for {:?}.", vid_path);
            }
            else {
                 mute_video(vid_path, &output_path_temp_processing, None, args.ffmpeg_threads)
                    .with_context(|| format!("Failed to copy video stream for {:?}", vid_path))?;
            }
        } else {
            println!("  Muting {} profanity instances:", ranges_to_mute.len());
            for (s, e, w) in &ranges_to_mute {
                println!("    {:.2}s – {:.2}s ({})", s, e, w);
            }
            let filter_string = vol_filter(&ranges_to_mute);
            mute_video(vid_path, &output_path_temp_processing, filter_string, args.ffmpeg_threads)
                .with_context(|| format!("Failed to mute video {:?}", vid_path))?;
        }

        if args.overwrite && output_path_temp_processing != *vid_path {
            // Only rename if a temporary file was actually used for processing and is different from original
             if output_path_temp_processing.exists() { // Ensure temp file was created
                fs::rename(&output_path_temp_processing, &output_path_final)
                    .with_context(|| format!("Failed to rename temp file {:?} to {:?}", output_path_temp_processing, output_path_final))?;
                println!("  ✔ Overwritten original video: {:?}", output_path_final);
             } else if !ranges_to_mute.is_empty() { // If ranges were supposed to be muted but temp file doesn't exist
                eprintln!("  Warning: Temporary processing file {:?} not found after muting attempt. Original file {:?} not overwritten.", output_path_temp_processing, vid_path);
             }
        } else if !args.overwrite {
             println!("  ✔ Censored video saved to: {:?}", output_path_final);
        }


        let log_file_stem = current_file_stem_str.as_ref(); // .as_ref() because current_file_stem_str is Cow<str>
        let (tsv_log_path, csv_log_path) = write_logs(&log_dir, log_file_stem, &ranges_to_mute)
            .context("Failed to write log files")?;
        println!("  ✔ Log files saved: {:?} & {:?}", tsv_log_path, csv_log_path);

        if args.srt {
            let srt_file_name = format!("{}_auto.srt", current_file_stem_str);
            let srt_output_path = vid_path.with_file_name(srt_file_name);
            let mut srt_file = File::create(&srt_output_path)
                .with_context(|| format!("Failed to create SRT file at {:?}", srt_output_path))?;
            
            if ranges_to_mute.is_empty() {
                 writeln!(srt_file, "1\n00:00:00,000 --> 00:00:00,000\n(No profanity detected)")
                    .context("Failed to write 'no profanity' message to SRT file")?;
            } else {
                for (idx, (s, e, _original_words)) in ranges_to_mute.iter().enumerate() {
                    // Using a generic placeholder for censored text in SRT
                    let censored_text_display = "[censored]";
                    writeln!(
                        srt_file,
                        "{}\n{} --> {}\n{}\n",
                        idx + 1,
                        timestamp(*s),
                        timestamp(*e),
                        censored_text_display
                    )
                    .context("Failed to write segment to SRT file")?;
                }
            }
            println!("  ✔ SRT subtitles saved: {:?}", srt_output_path);
        }
        // Clean up the temporary WAV file explicitly, though NamedTempFile handles it on drop.
        // _wav_tmp_file.close()?; // This consumes the file, drop is usually enough.
    }
    println!("\nProcessing complete.");
    Ok(())
}

