"""
Audio Note Detector and Splitter

This script takes an audio file, detects musical notes, and splits it into
separate files named after each detected note. Splits occur at zero crossings.
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import csv
from collections import defaultdict


def detect_zero_crossings(audio, sr):
    """
    Detect zero crossings in the audio signal.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        
    Returns:
        Array of zero crossing indices
    """
    # Find zero crossings
    zero_crossings = librosa.zero_crossings(audio, pad=False)
    zero_crossing_indices = np.where(zero_crossings)[0]
    return zero_crossing_indices


def calculate_rms_volume(audio_segment):
    """
    Calculate the RMS (Root Mean Square) volume of an audio segment.
    
    Args:
        audio_segment: Audio signal array
        
    Returns:
        RMS volume value (0.0 to 1.0+ for normalized audio)
    """
    if len(audio_segment) == 0:
        return 0.0
    
    # Calculate RMS (Root Mean Square)
    rms = np.sqrt(np.mean(audio_segment ** 2))
    return float(rms)


def detect_pitch_librosa(audio_segment, sr, min_freq=80, max_freq=2000):
    """
    Detect pitch using librosa's pyin method (more robust).
    
    Args:
        audio_segment: Audio signal segment
        sr: Sample rate
        min_freq: Minimum frequency to detect (Hz)
        max_freq: Maximum frequency to detect (Hz)
        
    Returns:
        Detected frequency in Hz, or None if no clear pitch
    """
    if len(audio_segment) < sr * 0.05:  # Need at least 50ms for pyin
        return None
    
    try:
        # Calculate appropriate frame length (should be power of 2 and long enough)
        frame_length = 2048
        if len(audio_segment) < frame_length:
            # Use next power of 2 that fits
            frame_length = 2 ** int(np.ceil(np.log2(len(audio_segment))))
            if frame_length < 512:
                frame_length = 512
        
        hop_length = frame_length // 4
        
        # Use librosa's pyin for robust pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_segment,
            fmin=min_freq,
            fmax=max_freq,
            frame_length=frame_length,
            hop_length=hop_length,
            threshold=0.1  # Lower threshold for more lenient detection
        )
        
        # Get median of voiced frames (more robust than mean)
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) > 0:
            # Use median to avoid outliers
            median_f0 = np.median(voiced_f0)
            if not np.isnan(median_f0) and min_freq <= median_f0 <= max_freq:
                return float(median_f0)
    except Exception as e:
        # Silently fall back to other methods
        pass
    
    return None


def detect_pitch_autocorrelation(audio_segment, sr, min_freq=80, max_freq=2000):
    """
    Detect the dominant pitch using autocorrelation (fallback method).
    
    Args:
        audio_segment: Audio signal segment
        sr: Sample rate
        min_freq: Minimum frequency to detect (Hz)
        max_freq: Maximum frequency to detect (Hz)
        
    Returns:
        Detected frequency in Hz, or None if no clear pitch
    """
    if len(audio_segment) < sr * 0.02:  # Need at least 20ms
        return None
    
    # Normalize the segment
    audio_segment = audio_segment - np.mean(audio_segment)
    if np.std(audio_segment) < 1e-6:
        return None
    audio_segment = audio_segment / np.std(audio_segment)
    
    # Calculate autocorrelation
    autocorr = np.correlate(audio_segment, audio_segment, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find the range of lags corresponding to min_freq to max_freq
    min_lag = int(sr / max_freq)
    max_lag = int(sr / min_freq)
    
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    if min_lag < 1:
        min_lag = 1
    
    if max_lag <= min_lag:
        return None
    
    # Find the peak in the autocorrelation within the valid range
    search_range = autocorr[min_lag:max_lag]
    if len(search_range) == 0:
        return None
    
    peak_index = np.argmax(search_range)
    lag = peak_index + min_lag
    
    if lag == 0:
        return None
    
    # Calculate frequency from lag
    frequency = sr / lag
    
    # Check if the peak is significant - LOWERED from 0.3 to 0.1 for more lenient detection
    peak_value = autocorr[lag]
    max_autocorr = np.max(autocorr[min_lag:max_lag])
    
    if peak_value < max_autocorr * 0.1:  # More lenient threshold
        return None
    
    return frequency


def detect_pitch_fft(audio_segment, sr, min_freq=80, max_freq=2000):
    """
    Detect pitch using FFT-based method (good for piano).
    
    Args:
        audio_segment: Audio signal segment
        sr: Sample rate
        min_freq: Minimum frequency to detect (Hz)
        max_freq: Maximum frequency to detect (Hz)
        
    Returns:
        Detected frequency in Hz, or None if no clear pitch
    """
    if len(audio_segment) < sr * 0.02:
        return None
    
    # Apply window and compute FFT
    windowed = audio_segment * np.hanning(len(audio_segment))
    fft = np.fft.rfft(windowed)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(windowed), 1/sr)
    
    # Find the range of frequencies we're interested in
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(freq_mask):
        return None
    
    # Find peaks in the magnitude spectrum
    magnitude_range = magnitude[freq_mask]
    freqs_range = freqs[freq_mask]
    
    # Find the dominant peak
    peak_idx = np.argmax(magnitude_range)
    peak_freq = freqs_range[peak_idx]
    peak_magnitude = magnitude_range[peak_idx]
    
    # Check if the peak is significant (at least 10% of max)
    max_magnitude = np.max(magnitude_range)
    if peak_magnitude < max_magnitude * 0.1:
        return None
    
    return float(peak_freq)


def detect_pitch(audio_segment, sr, min_freq=80, max_freq=2000):
    """
    Detect pitch using multiple methods with fallback.
    
    Args:
        audio_segment: Audio signal segment
        sr: Sample rate
        min_freq: Minimum frequency to detect (Hz)
        max_freq: Maximum frequency to detect (Hz)
        
    Returns:
        Detected frequency in Hz, or None if no clear pitch
    """
    # Try librosa's pyin first (most robust)
    frequency = detect_pitch_librosa(audio_segment, sr, min_freq, max_freq)
    if frequency is not None:
        return frequency
    
    # Try FFT-based method (good for piano)
    frequency = detect_pitch_fft(audio_segment, sr, min_freq, max_freq)
    if frequency is not None:
        return frequency
    
    # Fallback to autocorrelation
    frequency = detect_pitch_autocorrelation(audio_segment, sr, min_freq, max_freq)
    return frequency


def frequency_to_note(frequency):
    """
    Convert frequency in Hz to musical note name.
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        Tuple of (note_name, octave) or (None, None) if invalid
    """
    if frequency is None or frequency <= 0:
        return None, None
    
    # A4 = 440 Hz
    A4 = 440.0
    
    # Calculate semitones from A4
    semitones = 12 * np.log2(frequency / A4)
    
    # Round to nearest semitone
    semitones = round(semitones)
    
    # Calculate octave and note
    # A4 is note 9 (A) in octave 4
    note_index = (9 + semitones) % 12
    octave = 4 + (9 + semitones) // 12
    
    # Note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    note_name = note_names[note_index]
    
    return note_name, octave


def format_note_name(note_name, octave):
    """
    Format note name with octave for filename.
    
    Args:
        note_name: Note name (e.g., 'C', 'C#')
        octave: Octave number
        
    Returns:
        Formatted string (e.g., 'C4', 'C#4')
    """
    # Replace '#' with 'sharp' for filename compatibility
    note_name_safe = note_name.replace('#', 'sharp')
    return f"{note_name_safe}{octave}"


def split_audio_by_notes(audio_file, output_dir=None, min_segment_duration=0.05, 
                         window_size=0.15, hop_size=0.03, min_volume=0.01):
    """
    Split audio file into separate files based on detected notes.
    
    Args:
        audio_file: Path to input audio file
        output_dir: Directory to save output files (default: same as input)
        min_segment_duration: Minimum duration for a note segment (seconds)
        window_size: Size of window for pitch detection (seconds)
        hop_size: Hop size for pitch detection (seconds)
        min_volume: Minimum RMS volume threshold (0.0 to 1.0+, default: 0.01)
    """
    # Load audio file
    print(f"Loading audio file: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    print(f"Audio loaded: {len(audio)/sr:.2f}s duration, sample rate: {sr} Hz")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(audio_file).parent / f"{Path(audio_file).stem}_notes"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Settings: min_duration={min_segment_duration}s, min_volume={min_volume:.4f}, window={window_size}s, hop={hop_size}s")
    
    # Detect zero crossings
    print("Detecting zero crossings...")
    zero_crossings = detect_zero_crossings(audio, sr)
    
    if len(zero_crossings) == 0:
        print("No zero crossings found!")
        return
    
    # Convert zero crossings to sample indices
    # We'll use zero crossings as potential split points
    print(f"Found {len(zero_crossings)} zero crossings")
    
    # Analyze audio in windows to detect notes
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    # Store detected notes with their time ranges
    note_segments = []
    current_note = None
    current_octave = None
    current_start = 0
    current_start_sample = 0
    no_pitch_count = 0
    max_no_pitch_frames = 5  # Allow several frames without pitch before ending note
    
    # First pass: collect all pitch detections
    print("Analyzing pitch in audio windows...")
    print("-" * 80)
    detected_data = []  # Store (position, frequency, note_name, octave)
    
    for i in range(0, len(audio) - window_samples, hop_samples):
        window = audio[i:i + window_samples]
        time_sec = i / sr
        frequency = detect_pitch(window, sr)
        
        if frequency is not None:
            note_name, octave = frequency_to_note(frequency)
            if note_name is not None:
                detected_data.append((i, frequency, note_name, octave))
                print(f"  [{time_sec:6.2f}s] Frequency: {frequency:7.2f} Hz -> {note_name}{octave}")
            else:
                detected_data.append((i, None, None, None))
                print(f"  [{time_sec:6.2f}s] Frequency: {frequency:7.2f} Hz -> (invalid note)")
        else:
            detected_data.append((i, None, None, None))
            print(f"  [{time_sec:6.2f}s] No pitch detected")
    
    print("-" * 80)
    print(f"Detected {sum(1 for d in detected_data if d[1] is not None)} windows with pitch out of {len(detected_data)} total")
    print()
    
    # Second pass: group into note segments with tolerance for gaps
    print("Grouping detections into note segments...")
    print("-" * 80)
    for idx, (i, frequency, note_name, octave) in enumerate(detected_data):
        time_sec = i / sr
        # Find nearest zero crossing to window start
        zc_before = zero_crossings[zero_crossings <= i]
        
        if len(zc_before) > 0:
            window_start_zc = zc_before[-1]
        else:
            window_start_zc = i
        
        if frequency is None or note_name is None:
            # No pitch detected - increment counter
            no_pitch_count += 1
            # Only end note if we've had many consecutive frames without pitch
            if current_note is not None and no_pitch_count >= max_no_pitch_frames:
                # Find nearest zero crossing for end
                zc_end = zero_crossings[zero_crossings >= current_start_sample]
                if len(zc_end) > 0:
                    end_sample = zc_end[0]
                else:
                    end_sample = i
                
                duration = (end_sample - current_start_sample) / sr
                
                # Check volume of the segment
                segment_audio = audio[current_start_sample:end_sample]
                volume = calculate_rms_volume(segment_audio)
                
                if duration < min_segment_duration:
                    print(f"  [{time_sec:6.2f}s] ✗ Discarding short segment: {current_note}{current_octave} (duration: {duration:.3f}s < {min_segment_duration}s)")
                elif volume < min_volume:
                    print(f"  [{time_sec:6.2f}s] ✗ Discarding low volume segment: {current_note}{current_octave} (volume: {volume:.4f} < {min_volume:.4f})")
                else:
                    print(f"  [{time_sec:6.2f}s] ✓ Ending note segment: {current_note}{current_octave} (duration: {duration:.3f}s, volume: {volume:.4f})")
                    note_segments.append({
                        'note': current_note,
                        'octave': current_octave,
                        'start_sample': current_start_sample,
                        'end_sample': end_sample,
                        'duration': duration
                    })
                current_note = None
                current_octave = None
            continue
        
        # Reset no-pitch counter when we detect pitch
        no_pitch_count = 0
        
        # Check if note changed
        if current_note is None:
            # Start new note
            print(f"  [{time_sec:6.2f}s] Starting note: {note_name}{octave} (freq: {frequency:.2f} Hz)")
            current_note = note_name
            current_octave = octave
            current_start = i / sr
            current_start_sample = window_start_zc
        elif current_note != note_name or current_octave != octave:
            # Note changed - save previous note
            print(f"  [{time_sec:6.2f}s] Note changed: {current_note}{current_octave} -> {note_name}{octave} (freq: {frequency:.2f} Hz)")
            # Find nearest zero crossing for end - use the one just before the current window
            # This ensures we capture the full note before the transition
            zc_before_current = zero_crossings[zero_crossings < i]
            if len(zc_before_current) > 0:
                # Use the last zero crossing before current position
                end_sample = zc_before_current[-1]
            else:
                # Fallback: find zero crossing after start but before current
                zc_end = zero_crossings[(zero_crossings >= current_start_sample) & (zero_crossings < i)]
                if len(zc_end) > 0:
                    end_sample = zc_end[-1]  # Use last zero crossing before current
                else:
                    end_sample = i  # Use current position as fallback
            
            # Ensure end_sample is after start_sample
            if end_sample <= current_start_sample:
                # Find first zero crossing after start
                zc_after_start = zero_crossings[zero_crossings > current_start_sample]
                if len(zc_after_start) > 0:
                    end_sample = zc_after_start[0]
                else:
                    end_sample = current_start_sample + hop_samples  # At least one hop
            
            duration = (end_sample - current_start_sample) / sr
            print(f"  [{time_sec:6.2f}s] Calculating segment: start={current_start_sample/sr:.3f}s, end={end_sample/sr:.3f}s, duration={duration:.3f}s")
            
            # Check volume of the segment
            segment_audio = audio[current_start_sample:end_sample]
            volume = calculate_rms_volume(segment_audio)
            
            if duration < min_segment_duration:
                print(f"  [{time_sec:6.2f}s] ✗ Discarding short segment: {current_note}{current_octave} (duration: {duration:.3f}s < {min_segment_duration}s)")
            elif volume < min_volume:
                print(f"  [{time_sec:6.2f}s] ✗ Discarding low volume segment: {current_note}{current_octave} (volume: {volume:.4f} < {min_volume:.4f})")
            else:
                print(f"  [{time_sec:6.2f}s] ✓ Saving note segment: {current_note}{current_octave} (duration: {duration:.3f}s, volume: {volume:.4f})")
                note_segments.append({
                    'note': current_note,
                    'octave': current_octave,
                    'start_sample': current_start_sample,
                    'end_sample': end_sample,
                    'duration': duration
                })
            
            # Start new note
            print(f"  [{time_sec:6.2f}s] Starting new note: {note_name}{octave} (freq: {frequency:.2f} Hz)")
            current_note = note_name
            current_octave = octave
            current_start = i / sr
            current_start_sample = window_start_zc
    
    # Handle last note
    if current_note is not None:
        # Find the last zero crossing that's >= current_start_sample and < len(audio)
        zc_valid = zero_crossings[(zero_crossings >= current_start_sample) & (zero_crossings < len(audio))]
        if len(zc_valid) > 0:
            end_sample = zc_valid[-1]
        else:
            # If no valid zero crossing, use last sample
            end_sample = len(audio) - 1
        
        duration = (end_sample - current_start_sample) / sr
        
        # Check volume of the segment
        segment_audio = audio[current_start_sample:end_sample]
        volume = calculate_rms_volume(segment_audio)
        
        if duration < min_segment_duration:
            print(f"  [End] ✗ Discarding short final segment: {current_note}{current_octave} (duration: {duration:.3f}s < {min_segment_duration}s)")
        elif volume < min_volume:
            print(f"  [End] ✗ Discarding low volume final segment: {current_note}{current_octave} (volume: {volume:.4f} < {min_volume:.4f})")
        else:
            print(f"  [End] ✓ Saving final note segment: {current_note}{current_octave} (duration: {duration:.3f}s, volume: {volume:.4f})")
            note_segments.append({
                'note': current_note,
                'octave': current_octave,
                'start_sample': current_start_sample,
                'end_sample': end_sample,
                'duration': duration
            })
    
    print("-" * 80)
    print(f"\nDetected {len(note_segments)} note segments")
    
    if len(note_segments) == 0:
        print("WARNING: No note segments were saved! Check the debug output above.")
        return
    
    print("\nNote segments to be saved:")
    for idx, segment in enumerate(note_segments):
        print(f"  {idx+1}. {segment['note']}{segment['octave']} - "
              f"start: {segment['start_sample']/sr:.3f}s, "
              f"end: {segment['end_sample']/sr:.3f}s, "
              f"duration: {segment['duration']:.3f}s")
    
    # Save note lengths to CSV
    csv_file = output_dir / "note_lengths.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Note', 'Octave', 'Start Time (s)', 'End Time (s)', 'Duration (s)', 'Filename'])
        
        # Group notes by name to avoid filename conflicts
        note_counts = defaultdict(int)
        
        for i, segment in enumerate(note_segments):
            note_full = format_note_name(segment['note'], segment['octave'])
            note_counts[note_full] += 1
            
            # Create unique filename
            if note_counts[note_full] > 1:
                filename = f"{note_full}_{note_counts[note_full]}.wav"
            else:
                filename = f"{note_full}.wav"
            
            start_time = segment['start_sample'] / sr
            end_time = segment['end_sample'] / sr
            
            writer.writerow([
                segment['note'],
                segment['octave'],
                f"{start_time:.4f}",
                f"{end_time:.4f}",
                f"{segment['duration']:.4f}",
                filename
            ])
            
            # Extract and save audio segment
            # Ensure we start and end at zero crossings
            # Find exact zero crossing at start
            zc_start = zero_crossings[zero_crossings <= segment['start_sample']]
            if len(zc_start) > 0:
                actual_start = zc_start[-1]
            else:
                actual_start = segment['start_sample']
            
            # Find exact zero crossing at end
            zc_end = zero_crossings[zero_crossings >= segment['end_sample']]
            if len(zc_end) > 0:
                actual_end = zc_end[0]
            else:
                actual_end = segment['end_sample']
            
            # Extract with zero crossings
            audio_segment = audio[actual_start:actual_end]
            
            output_file = output_dir / filename
            sf.write(str(output_file), audio_segment, sr)
            
            print(f"  {i+1}. {segment['note']}{segment['octave']} - "
                  f"{segment['duration']:.3f}s -> {filename}")
    
    print(f"\nNote lengths saved to: {csv_file}")
    print(f"All note files saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python note_detector.py <audio_file> [output_dir]")
        print("\nExample:")
        print("  python note_detector.py audio.wav")
        print("  python note_detector.py audio.wav output/")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(audio_file).exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    split_audio_by_notes(audio_file, output_dir)

