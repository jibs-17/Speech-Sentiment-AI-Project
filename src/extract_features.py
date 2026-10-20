import os
import sys
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_features_from_audio(y, sr=22050):
    """
    Extracts a 478-dimensional acoustic feature vector from an audio waveform y:
    - Zero Crossing Rate: mean, std, max (3)
    - RMS Energy: mean, std, max (3)
    - Spectral Centroid: mean, std (2)
    - Spectral Bandwidth: mean, std (2)
    - Spectral Rolloff: mean, std (2)
    - Mel Spectrogram (128 bands): mean, std (256)
    - MFCCs (40 coefficients): mean, std (80)
    - MFCC Delta: mean, std (80)
    - Chroma STFT (12 bands): mean, std (24)
    - Spectral Contrast (7 bands): mean, std (14)
    - Tonnetz (6 bands): mean, std (12)
    Total: 478 features
    """
    # Ensure minimum audio length (at least 0.5 sec)
    if len(y) < sr * 0.5:
        y = np.pad(y, (0, int(sr * 0.5) - len(y)), mode='constant')

    # Compute STFT once to accelerate all downstream spectral features
    D = np.abs(librosa.stft(y))
    S = D ** 2

    features = []

    # 1. ZCR
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.extend([float(np.mean(zcr)), float(np.std(zcr)), float(np.max(zcr))])

    # 2. RMS Energy
    rms = librosa.feature.rms(S=D)
    features.extend([float(np.mean(rms)), float(np.std(rms)), float(np.max(rms))])

    # 3. Spectral Centroid
    cent = librosa.feature.spectral_centroid(S=D, sr=sr)
    features.extend([float(np.mean(cent)), float(np.std(cent))])

    # 4. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=D, sr=sr)
    features.extend([float(np.mean(bandwidth)), float(np.std(bandwidth))])

    # 5. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(S=D, sr=sr)
    features.extend([float(np.mean(rolloff)), float(np.std(rolloff))])

    # 6. Mel Spectrogram (128 bands)
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1).tolist())
    features.extend(np.std(mel_db, axis=1).tolist())

    # 7. MFCCs (40 coefficients)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1).tolist())
    features.extend(np.std(mfcc, axis=1).tolist())

    # 8. MFCC Delta
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1).tolist())
    features.extend(np.std(mfcc_delta, axis=1).tolist())

    # 9. Chroma STFT
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    features.extend(np.mean(chroma, axis=1).tolist())
    features.extend(np.std(chroma, axis=1).tolist())

    # 10. Spectral Contrast
    try:
        contrast = librosa.feature.spectral_contrast(S=D, sr=sr)
        features.extend(np.mean(contrast, axis=1).tolist())
        features.extend(np.std(contrast, axis=1).tolist())
    except Exception:
        features.extend([0.0] * 14)

    # 11. Tonnetz
    try:
        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        features.extend(np.mean(tonnetz, axis=1).tolist())
        features.extend(np.std(tonnetz, axis=1).tolist())
    except Exception:
        features.extend([0.0] * 12)

    feat_array = np.array(features, dtype=np.float32)
    # Replace any potential NaNs or Infs
    feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_array

def augment_audio(y, sr):
    """
    Generates realistic augmented variants for robust real-world microphone performance:
    1. White noise injection (SNR ~ 25 dB)
    2. Pitch shift up (+1.5 semitones)
    """
    augmented = []
    
    # 1. Subtle white noise
    noise_amp = 0.005 * np.amax(np.abs(y)) if np.amax(np.abs(y)) > 0 else 0.001
    y_noise = y + noise_amp * np.random.normal(size=len(y)).astype(np.float32)
    augmented.append(('noise', y_noise))
    
    # 2. Pitch shift (+1.5 semitones)
    try:
        y_pitch_up = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=1.5)
        augmented.append(('pitch_up', y_pitch_up))
    except Exception:
        pass

    return augmented

def process_single_file(args):
    file_path, emotion, sentiment, actor_id, gender, augment = args
    results = []
    
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)
        
        # 1. Original features
        feat_orig = extract_features_from_audio(y, sr)
        results.append({
            'features': feat_orig,
            'emotion': emotion,
            'sentiment': sentiment,
            'actor_id': actor_id,
            'gender': gender,
            'augmented': 'none',
            'file_path': file_path
        })
        
        # 2. Augmented features
        if augment:
            variants = augment_audio(y, sr)
            for aug_name, y_aug in variants:
                feat_aug = extract_features_from_audio(y_aug, sr)
                results.append({
                    'features': feat_aug,
                    'emotion': emotion,
                    'sentiment': sentiment,
                    'actor_id': actor_id,
                    'gender': gender,
                    'augmented': aug_name,
                    'file_path': file_path
                })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return results

def extract_all(metadata_df, output_dir="data", augment=True, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    for _, row in metadata_df.iterrows():
        tasks.append((
            row['file_path'],
            row['emotion'],
            row['sentiment'],
            row['actor_id'],
            row['gender'],
            augment
        ))
        
    all_data = []
    print(f"Extracting features from {len(tasks)} files (augment={augment}, workers={max_workers})...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
            res = f.result()
            if res:
                all_data.extend(res)
                
    print(f"\nTotal feature vectors extracted: {len(all_data)}")
    
    X = np.array([item['features'] for item in all_data], dtype=np.float32)
    emotions = np.array([item['emotion'] for item in all_data])
    sentiments = np.array([item['sentiment'] for item in all_data])
    actors = np.array([item['actor_id'] for item in all_data])
    genders = np.array([item['gender'] for item in all_data])
    aug_types = np.array([item['augmented'] for item in all_data])
    file_paths = np.array([item['file_path'] for item in all_data])
    
    output_npz = os.path.join(output_dir, "ravdess_features.npz")
    np.savez_compressed(
        output_npz,
        X=X,
        emotions=emotions,
        sentiments=sentiments,
        actors=actors,
        genders=genders,
        aug_types=aug_types,
        file_paths=file_paths
    )
    print(f"Saved extracted features to: {output_npz}")
    print(f"Feature matrix shape: {X.shape}")
    return output_npz

if __name__ == "__main__":
    from dataset import load_ravdess_metadata
    df = load_ravdess_metadata()
    extract_all(df, augment=True, max_workers=8)
