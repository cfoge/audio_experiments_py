"""
Audio Mosaic - Recreate an audio file using snippets from another audio file

This script analyzes a target audio file and attempts to recreate it by finding
the most spectrally similar sound clips from a source audio file. Snippets can
be repeated or reused as needed.
"""

import librosa
import librosa.feature
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy import signal
import argparse


def extract_features(audio, sr, hop_length=512, n_mfcc=13):
    """
    Extract spectral features from audio for matching.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        hop_length: Hop length for analysis
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Dictionary containing various feature arrays
    """
    # MFCC features (good for timbral similarity)
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    
    # Chroma features (good for pitch/harmonic content)
    # Use chroma_stft which is the standard chroma function in librosa
    chroma_features = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
    
    # Spectral centroid (brightness)
    spectral_centroid_features = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)
    
    # Spectral rolloff
    spectral_rolloff_features = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
    
    # Combine features into a single feature vector per frame
    # Normalize each feature type
    mfcc_norm = (mfcc_features - np.mean(mfcc_features, axis=1, keepdims=True)) / (np.std(mfcc_features, axis=1, keepdims=True) + 1e-6)
    chroma_norm = (chroma_features - np.mean(chroma_features, axis=1, keepdims=True)) / (np.std(chroma_features, axis=1, keepdims=True) + 1e-6)
    spectral_centroid_norm = (spectral_centroid_features - np.mean(spectral_centroid_features)) / (np.std(spectral_centroid_features) + 1e-6)
    spectral_rolloff_norm = (spectral_rolloff_features - np.mean(spectral_rolloff_features)) / (np.std(spectral_rolloff_features) + 1e-6)
    zcr_norm = (zcr - np.mean(zcr)) / (np.std(zcr) + 1e-6)
    
    # Stack features horizontally
    features = np.vstack([
        mfcc_norm,
        chroma_norm,
        spectral_centroid_norm,
        spectral_rolloff_norm,
        zcr_norm
    ])
    
    return {
        'features': features.T,  # Shape: (n_frames, n_features)
        'mfcc': mfcc_features,
        'chroma': chroma_features,
        'hop_length': hop_length,
        'sr': sr
    }


def compute_segment_features(audio, sr, start_sample, end_sample, hop_length=512):
    """
    Compute features for a specific audio segment.
    
    Args:
        audio: Full audio signal
        sr: Sample rate
        start_sample: Start sample index
        end_sample: End sample index
        hop_length: Hop length for analysis
        
    Returns:
        Feature vector for the segment
    """
    segment = audio[start_sample:end_sample]
    if len(segment) == 0:
        return None
    
    features_dict = extract_features(segment, sr, hop_length=hop_length)
    # Average features across frames to get a single feature vector
    avg_features = np.mean(features_dict['features'], axis=0)
    return avg_features


def find_best_matches(target_features, source_features_dict, segment_length_samples, 
                      source_audio, target_sr, n_matches=1):
    """
    Find the best matching snippets from source for each target segment.
    
    Args:
        target_features: List of feature vectors for target segments
        source_features_dict: Dictionary with source features
        segment_length_samples: Length of segments in samples
        source_audio: Source audio array
        target_sr: Target sample rate
        n_matches: Number of best matches to consider
        
    Returns:
        List of tuples (start_sample, end_sample, similarity_score) for each target segment
    """
    source_features = source_features_dict['features']
    source_hop = source_features_dict['hop_length']
    source_sr = source_features_dict['sr']
    
    # Convert segment length to source frames
    segment_frames = int(segment_length_samples * source_sr / source_hop / target_sr)
    
    matches = []
    
    print(f"Finding matches for {len(target_features)} target segments...")
    print(f"Source has {len(source_features)} frames available")
    
    for i, target_feat in enumerate(target_features):
        if target_feat is None or len(target_feat) == 0:
            matches.append((0, segment_length_samples, 1.0))
            continue
        
        # Calculate distance to all source segments
        # Use sliding window approach
        best_match_idx = 0
        best_distance = float('inf')
        
        # Slide through source features
        for start_frame in range(0, len(source_features) - segment_frames + 1, max(1, segment_frames // 4)):
            end_frame = start_frame + segment_frames
            source_segment_feat = source_features[start_frame:end_frame]
            
            # Average the source segment features
            source_avg = np.mean(source_segment_feat, axis=0)
            
            # Compute cosine distance (1 - cosine similarity)
            # Normalize vectors
            target_norm = target_feat / (np.linalg.norm(target_feat) + 1e-6)
            source_norm = source_avg / (np.linalg.norm(source_avg) + 1e-6)
            
            # Cosine distance
            distance = 1 - np.dot(target_norm, source_norm)
            
            # Also consider Euclidean distance
            euclidean_dist = np.linalg.norm(target_feat - source_avg)
            
            # Combine distances (weighted)
            combined_distance = 0.7 * distance + 0.3 * (euclidean_dist / (len(target_feat) + 1e-6))
            
            if combined_distance < best_distance:
                best_distance = combined_distance
                best_match_idx = start_frame
        
        # Convert frame index to sample index
        best_start_sample = best_match_idx * source_hop
        best_end_sample = best_start_sample + segment_length_samples
        
        # Ensure we don't exceed source audio length
        if best_end_sample > len(source_audio):
            best_end_sample = len(source_audio)
            best_start_sample = best_end_sample - segment_length_samples
            if best_start_sample < 0:
                best_start_sample = 0
        
        similarity = 1.0 - best_distance  # Convert distance to similarity
        matches.append((best_start_sample, best_end_sample, similarity))
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(target_features)} segments...")
    
    return matches


def compute_volume_envelope(audio, window_size=512):
    """
    Compute volume envelope (RMS) for an audio segment.
    
    Args:
        audio: Audio signal array
        window_size: Window size for RMS calculation
        
    Returns:
        Volume envelope array with same length as audio
    """
    if len(audio) == 0:
        return np.array([])
    
    # Use vectorized RMS calculation with convolution
    # Square the audio
    audio_squared = audio ** 2
    
    # Create a window for moving average
    window = np.ones(window_size) / window_size
    
    # Pad audio to handle boundaries
    padded = np.pad(audio_squared, (window_size // 2, window_size // 2), mode='edge')
    
    # Compute moving average using convolution
    from scipy import signal as scipy_signal
    envelope_squared = scipy_signal.convolve(padded, window, mode='valid')
    
    # Take square root to get RMS
    envelope = np.sqrt(envelope_squared)
    
    # Ensure envelope matches audio length
    if len(envelope) != len(audio):
        # Trim or pad as needed
        if len(envelope) > len(audio):
            envelope = envelope[:len(audio)]
        else:
            envelope = np.pad(envelope, (0, len(audio) - len(envelope)), mode='edge')
    
    # Smooth the envelope to avoid abrupt changes
    if len(envelope) > 3:
        # Use Savitzky-Golay filter for smoothing
        filter_length = min(len(envelope), 101)
        if filter_length % 2 == 0:
            filter_length -= 1
        if filter_length >= 3:
            envelope = scipy_signal.savgol_filter(envelope, filter_length, 3)
    
    return envelope


def apply_volume_envelope(audio, target_envelope):
    """
    Apply volume envelope from target to source audio.
    
    Args:
        audio: Source audio to apply envelope to
        target_envelope: Volume envelope from target audio
        
    Returns:
        Audio with target volume envelope applied
    """
    if len(audio) == 0 or len(target_envelope) == 0:
        return audio
    
    # Normalize target envelope to match audio length
    if len(target_envelope) != len(audio):
        # Interpolate envelope to match audio length
        from scipy.interpolate import interp1d
        target_indices = np.linspace(0, len(target_envelope) - 1, len(audio))
        interp_func = interp1d(np.arange(len(target_envelope)), target_envelope, 
                              kind='linear', fill_value='extrapolate')
        target_envelope = interp_func(target_indices)
    
    # Compute source envelope
    source_envelope = compute_volume_envelope(audio)
    
    # Avoid division by zero
    source_envelope = np.maximum(source_envelope, 1e-8)
    target_envelope = np.maximum(target_envelope, 1e-8)
    
    # Compute gain factor
    gain_factor = target_envelope / source_envelope
    
    # Limit gain to avoid extreme values (e.g., 0.1x to 10x)
    gain_factor = np.clip(gain_factor, 0.1, 10.0)
    
    # Apply gain
    result = audio * gain_factor
    
    return result


def crossfade(audio1, audio2, fade_length_samples):
    """
    Crossfade between two audio segments.
    
    Args:
        audio1: First audio segment
        audio2: Second audio segment
        fade_length_samples: Length of fade in samples
        
    Returns:
        Crossfaded audio
    """
    fade_length_samples = min(fade_length_samples, len(audio1), len(audio2))
    
    if fade_length_samples == 0:
        return np.concatenate([audio1, audio2])
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, fade_length_samples)
    fade_in = np.linspace(0.0, 1.0, fade_length_samples)
    
    # Apply fades
    audio1_faded = audio1.copy()
    audio2_faded = audio2.copy()
    
    audio1_faded[-fade_length_samples:] *= fade_out
    audio2_faded[:fade_length_samples] *= fade_in
    
    # Overlap and add
    overlapped = audio1_faded[-fade_length_samples:] + audio2_faded[:fade_length_samples]
    
    # Combine
    result = np.concatenate([
        audio1_faded[:-fade_length_samples],
        overlapped,
        audio2_faded[fade_length_samples:]
    ])
    
    return result


def create_audio_mosaic(target_file, source_file, output_file, 
                       segment_length=0.1, hop_length=512, 
                       crossfade_length=0.01, min_similarity=0.0,
                       unmatched_action='silence', volume_threshold=0.001):
    """
    Create an audio mosaic by recreating target audio using source snippets.
    
    Args:
        target_file: Path to target audio file to recreate
        source_file: Path to source audio file with snippets
        output_file: Path to output mosaic file
        segment_length: Length of each segment in seconds
        hop_length: Hop length for analysis
        crossfade_length: Length of crossfade between segments in seconds
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
        unmatched_action: What to do with unmatched segments - 'silence' or 'original' (default: 'silence')
        volume_threshold: RMS volume threshold below which segments are treated as silence (default: 0.001)
    """
    print("=" * 80)
    print("Audio Mosaic Generator")
    print("=" * 80)
    
    # Load audio files
    print(f"\nLoading target audio: {target_file}")
    target_audio, target_sr = librosa.load(target_file, sr=None, mono=True)
    print(f"  Duration: {len(target_audio)/target_sr:.2f}s, Sample rate: {target_sr} Hz")
    
    print(f"\nLoading source audio: {source_file}")
    source_audio, source_sr = librosa.load(source_file, sr=None, mono=True)
    print(f"  Duration: {len(source_audio)/source_sr:.2f}s, Sample rate: {source_sr} Hz")
    
    # Resample if needed to match sample rates
    if target_sr != source_sr:
        print(f"\nResampling source from {source_sr} Hz to {target_sr} Hz...")
        source_audio = librosa.resample(source_audio, orig_sr=source_sr, target_sr=target_sr)
        source_sr = target_sr
    
    # Extract features
    print("\nExtracting features from source audio...")
    source_features = extract_features(source_audio, source_sr, hop_length=hop_length)
    
    print("\nExtracting features from target audio...")
    target_features_dict = extract_features(target_audio, target_sr, hop_length=hop_length)
    
    # Divide target into segments
    segment_length_samples = int(segment_length * target_sr)
    hop_segment_samples = segment_length_samples  # Can overlap segments if desired
    
    print(f"\nDividing target into segments...")
    print(f"  Segment length: {segment_length:.3f}s ({segment_length_samples} samples)")
    
    target_segments = []
    target_segment_features = []
    
    for start_sample in range(0, len(target_audio) - segment_length_samples + 1, hop_segment_samples):
        end_sample = start_sample + segment_length_samples
        segment = target_audio[start_sample:end_sample]
        
        # Compute features for this segment
        features = compute_segment_features(target_audio, target_sr, start_sample, end_sample, hop_length)
        
        target_segments.append((start_sample, end_sample))
        target_segment_features.append(features)
    
    # Handle remaining audio if any
    if len(target_audio) % hop_segment_samples != 0:
        start_sample = len(target_audio) - segment_length_samples
        end_sample = len(target_audio)
        features = compute_segment_features(target_audio, target_sr, start_sample, end_sample, hop_length)
        target_segments.append((start_sample, end_sample))
        target_segment_features.append(features)
    
    print(f"  Created {len(target_segments)} target segments")
    
    # Find best matches
    print("\n" + "-" * 80)
    matches = find_best_matches(
        target_segment_features, 
        source_features, 
        segment_length_samples,
        source_audio,
        target_sr
    )
    
    # Build output audio
    print("\n" + "-" * 80)
    print("Constructing output audio with volume envelope preservation...")
    output_audio = []
    crossfade_samples = int(crossfade_length * target_sr)
    
    avg_similarity = 0.0
    valid_matches = 0
    
    for i, ((target_start, target_end), (source_start, source_end, similarity)) in enumerate(zip(target_segments, matches)):
        # Extract target segment to check volume
        target_segment = target_audio[target_start:target_end]
        
        # Check volume threshold - skip matching if volume is too low
        segment_rms = np.sqrt(np.mean(target_segment ** 2))
        if segment_rms < volume_threshold:
            # Handle low volume segment based on unmatched_action
            if unmatched_action == 'original' or unmatched_action == 'target':
                # Use original target audio for this segment
                target_seg = target_audio[target_start:target_end]
                
                # Ensure segment has correct length
                if len(target_seg) < segment_length_samples:
                    target_seg = np.pad(target_seg, (0, segment_length_samples - len(target_seg)), mode='constant')
                elif len(target_seg) > segment_length_samples:
                    target_seg = target_seg[:segment_length_samples]
                
                # Add original target segment to output
                if len(output_audio) > 0 and crossfade_samples > 0:
                    output_audio = crossfade(output_audio, target_seg, crossfade_samples)
                else:
                    output_audio = np.concatenate([output_audio, target_seg]) if len(output_audio) > 0 else target_seg
                
                print(f"  Segment {i+1}: Using original target audio (volume {segment_rms:.6f} < {volume_threshold:.6f})")
            else:
                # Insert silence (default behavior)
                silence = np.zeros(segment_length_samples)
                
                if len(output_audio) > 0 and crossfade_samples > 0:
                    output_audio = crossfade(output_audio, silence, crossfade_samples)
                else:
                    output_audio = np.concatenate([output_audio, silence]) if len(output_audio) > 0 else silence
                
                print(f"  Segment {i+1}: Inserting silence (volume {segment_rms:.6f} < {volume_threshold:.6f})")
            continue
        
        # Check similarity threshold
        if similarity < min_similarity:
            # Handle unmatched segment based on unmatched_action
            if unmatched_action == 'original' or unmatched_action == 'target':
                # Use original target audio for this segment
                target_seg = target_audio[target_start:target_end]
                
                # Ensure segment has correct length
                if len(target_seg) < segment_length_samples:
                    target_seg = np.pad(target_seg, (0, segment_length_samples - len(target_seg)), mode='constant')
                elif len(target_seg) > segment_length_samples:
                    target_seg = target_seg[:segment_length_samples]
                
                # Add original target segment to output
                if len(output_audio) > 0 and crossfade_samples > 0:
                    output_audio = crossfade(output_audio, target_seg, crossfade_samples)
                else:
                    output_audio = np.concatenate([output_audio, target_seg]) if len(output_audio) > 0 else target_seg
                
                print(f"  Segment {i+1}: Using original target audio (similarity {similarity:.3f} < {min_similarity:.3f})")
            else:
                # Insert silence (default behavior)
                silence = np.zeros(segment_length_samples)
                
                if len(output_audio) > 0 and crossfade_samples > 0:
                    output_audio = crossfade(output_audio, silence, crossfade_samples)
                else:
                    output_audio = np.concatenate([output_audio, silence]) if len(output_audio) > 0 else silence
                
                print(f"  Segment {i+1}: Inserting silence (similarity {similarity:.3f} < {min_similarity:.3f})")
            continue
        
        avg_similarity += similarity
        valid_matches += 1
        
        # Target segment already extracted above for volume check
        
        # Extract source snippet
        source_snippet = source_audio[source_start:source_end]
        
        # Ensure snippet has correct length (pad if needed)
        if len(source_snippet) < segment_length_samples:
            source_snippet = np.pad(source_snippet, (0, segment_length_samples - len(source_snippet)), mode='constant')
        elif len(source_snippet) > segment_length_samples:
            source_snippet = source_snippet[:segment_length_samples]
        
        # Extract volume envelope from target segment
        target_envelope = compute_volume_envelope(target_segment)
        
        # Apply target volume envelope to source snippet
        source_snippet = apply_volume_envelope(source_snippet, target_envelope)
        
        # Add to output with crossfade
        if len(output_audio) > 0 and crossfade_samples > 0:
            output_audio = crossfade(output_audio, source_snippet, crossfade_samples)
        else:
            output_audio = np.concatenate([output_audio, source_snippet]) if len(output_audio) > 0 else source_snippet
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(target_segments)} segments...")
    
    if valid_matches > 0:
        avg_similarity /= valid_matches
        print(f"\nAverage similarity: {avg_similarity:.3f}")
        print(f"Used {valid_matches}/{len(target_segments)} segments")
        unmatched_count = len(target_segments) - valid_matches
        if unmatched_count > 0:
            print(f"Unmatched segments: {unmatched_count} (handled with '{unmatched_action}')")
    else:
        print("\nWARNING: No valid matches found! Try lowering min_similarity.")
        if unmatched_action == 'original' or unmatched_action == 'target':
            print("Output will contain original target audio for all segments.")
        else:
            print("Output will contain silence for all segments.")
    
    # Ensure output length matches target (pad if shorter)
    if len(output_audio) < len(target_audio):
        padding = np.zeros(len(target_audio) - len(output_audio))
        output_audio = np.concatenate([output_audio, padding])
    elif len(output_audio) > len(target_audio):
        output_audio = output_audio[:len(target_audio)]
    
    # Save output
    print(f"\nSaving output to: {output_file}")
    sf.write(str(output_file), output_audio, target_sr)
    
    print(f"\nOutput duration: {len(output_audio)/target_sr:.2f}s")
    print(f"Target duration: {len(target_audio)/target_sr:.2f}s")
    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an audio mosaic by recreating target audio using source snippets"
    )
    parser.add_argument("target", help="Path to target audio file to recreate")
    parser.add_argument("source", help="Path to source audio file with snippets")
    parser.add_argument("output", help="Path to output mosaic file")
    parser.add_argument("--segment-length", type=float, default=0.3,
                       help="Length of each segment in seconds (default: 0.1)")
    parser.add_argument("--hop-length", type=int, default=512,
                       help="Hop length for analysis (default: 512)")
    parser.add_argument("--crossfade", type=float, default=0.02,
                       help="Crossfade length between segments in seconds (default: 0.04)")
    parser.add_argument("--min-similarity", type=float, default=0.1,
                       help="Minimum similarity threshold 0.0-1.0 (default: 0.1)")
    parser.add_argument("--unmatched-action", type=str, default='silence',
                       choices=['silence', 'original', 'target'],
                       help="What to do with unmatched segments: 'silence' or 'original'/'target' (default: 'silence')")
    parser.add_argument("--volume-threshold", type=float, default=0.0005,
                       help="RMS volume threshold below which segments are treated as silence (default: 0.001)")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.target).exists():
        print(f"Error: Target file not found: {args.target}")
        exit(1)
    
    if not Path(args.source).exists():
        print(f"Error: Source file not found: {args.source}")
        exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_audio_mosaic(
        args.target,
        args.source,
        args.output,
        segment_length=args.segment_length,
        hop_length=args.hop_length,
        crossfade_length=args.crossfade,
        min_similarity=args.min_similarity,
        unmatched_action=args.unmatched_action,
        volume_threshold=args.volume_threshold
    )

