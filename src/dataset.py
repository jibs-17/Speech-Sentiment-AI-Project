import os
import glob
import pandas as pd
from pathlib import Path

# RAVDESS emotion codes
EMOTION_DICT = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fearful',
    7: 'disgust',
    8: 'surprised'
}

# Emotion to Sentiment mapping:
# Positive: happy, calm
# Negative: angry, sad, fearful, disgust
# Neutral: neutral, surprised (surprised is valence-neutral/ambiguous in SER standard benchmarks)
SENTIMENT_DICT = {
    'neutral': 'Neutral',
    'calm': 'Positive',
    'happy': 'Positive',
    'sad': 'Negative',
    'angry': 'Negative',
    'fearful': 'Negative',
    'disgust': 'Negative',
    'surprised': 'Neutral'
}

INTENSITY_DICT = {
    1: 'normal',
    2: 'strong'
}

STATEMENT_DICT = {
    1: 'Kids are talking by the door',
    2: 'Dogs are sitting by the door'
}

def load_ravdess_metadata(base_dir: str = ".") -> pd.DataFrame:
    """
    Scans the directory for RAVDESS .wav files and parses their metadata from filenames.
    Filename format: 03-01-01-01-01-01-01.wav
    [Modality]-[VocalChannel]-[Emotion]-[Intensity]-[Statement]-[Repetition]-[Actor].wav
    """
    # Look first in audio_speech_actors_01-24, or fallback to Actor_* folders
    speech_dir = os.path.join(base_dir, "audio_speech_actors_01-24")
    if os.path.exists(speech_dir):
        pattern = os.path.join(speech_dir, "Actor_*", "*.wav")
        files = glob.glob(pattern)
    else:
        pattern = os.path.join(base_dir, "Actor_*", "*.wav")
        files = glob.glob(pattern)
        
    records = []
    for f in sorted(files):
        fname = Path(f).stem
        parts = fname.split('-')
        if len(parts) != 7:
            continue
            
        try:
            modality = int(parts[0])
            vocal_channel = int(parts[1])
            emotion_code = int(parts[2])
            intensity_code = int(parts[3])
            statement_code = int(parts[4])
            repetition_code = int(parts[5])
            actor_id = int(parts[6])
        except ValueError:
            continue
            
        emotion = EMOTION_DICT.get(emotion_code, 'unknown')
        sentiment = SENTIMENT_DICT.get(emotion, 'unknown')
        intensity = INTENSITY_DICT.get(intensity_code, 'normal')
        statement = STATEMENT_DICT.get(statement_code, 'unknown')
        gender = 'female' if (actor_id % 2 == 0) else 'male'
        
        records.append({
            'file_path': os.path.abspath(f),
            'filename': os.path.basename(f),
            'modality': modality,
            'vocal_channel': vocal_channel,
            'emotion_code': emotion_code,
            'emotion': emotion,
            'sentiment': sentiment,
            'intensity_code': intensity_code,
            'intensity': intensity,
            'statement_code': statement_code,
            'statement': statement,
            'repetition': repetition_code,
            'actor_id': actor_id,
            'gender': gender
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = load_ravdess_metadata()
    print(f"Total audio files found: {len(df)}")
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    print("\nGender distribution:")
    print(df['gender'].value_counts())
