import json

def create_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎙️ Speech Sentiment & Emotion Recognition System\n",
                "### 6th Semester Engineering Capstone Project\n",
                "**Dataset:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)\n",
                "\n",
                "---\n",
                "\n",
                "## 📌 Project Overview\n",
                "This project implements an end-to-end Machine Learning and Acoustic Signal Processing pipeline for **Speech Sentiment Analysis** and **Fine-Grained Emotion Recognition**.\n",
                "\n",
                "### Key Objectives:\n",
                "1. **Dataset Ingestion & Metadata Parsing:** Parse RAVDESS filenames containing modality, vocal channel, emotion, intensity, statement, repetition, and actor IDs.\n",
                "2. **Acoustic Signal Processing:** Extract 478 high-dimensional acoustic features including:\n",
                "   - Zero Crossing Rate (ZCR)\n",
                "   - Root-Mean-Square (RMS) Energy\n",
                "   - Spectral Centroid, Bandwidth, and Rolloff\n",
                "   - 128 Mel-frequency Spectrogram bands (mean + std)\n",
                "   - 40 Mel-Frequency Cepstral Coefficients (MFCCs) + First-order Deltas\n",
                "   - Chroma STFT (12 semitone pitch classes)\n",
                "   - Spectral Contrast (7 octave subbands)\n",
                "   - Tonnetz (6 tonal centroid features)\n",
                "3. **Data Augmentation:** White noise injection and pitch modulation to ensure real-world microphone robustness.\n",
                "4. **Comparative Model Training:** Benchmark **LightGBM**, **XGBoost**, **ExtraTrees**, and **Deep Multi-Layer Perceptron (MLP)** alongside a **Soft-Voting Ensemble**.\n",
                "5. **Dual-Task Prediction:** 3-class Sentiment Polarity (*Positive*, *Negative*, *Neutral*) and 8-class Emotion Classification (*Happy*, *Calm*, *Sad*, *Angry*, *Fearful*, *Disgust*, *Surprised*, *Neutral*).\n",
                "6. **Production Deployment:** Interactive Web Application powered by FastAPI, HTML5 AudioContext, Chart.js, and Tailwind CSS."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Imports and Environment Setup"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import glob\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import librosa\n",
                "import librosa.display\n",
                "import soundfile as sf\n",
                "import joblib\n",
                "from pathlib import Path\n",
                "\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score\n",
                "from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier\n",
                "from sklearn.neural_network import MLPClassifier\n",
                "import xgboost as xgb\n",
                "import lightgbm as lgb\n",
                "\n",
                "sns.set_theme(style='darkgrid')\n",
                "print('Environment and libraries loaded successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Loading & Exploratory Data Analysis (EDA)\n",
                "The RAVDESS audio dataset contains 1,440 speech files spoken by 24 professional actors (12 male, 12 female) across 8 emotional expressions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# RAVDESS Filename Convention:\n",
                "# Modality (01=full-av, 02=video-only, 03=audio-only)\n",
                "# Vocal channel (01=speech, 02=song)\n",
                "# Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)\n",
                "# Emotional intensity (01=normal, 02=strong)\n",
                "# Statement (01='Kids are talking by the door', 02='Dogs are sitting by the door')\n",
                "# Repetition (01=1st repetition, 02=2nd repetition)\n",
                "# Actor (01 to 24. Odd=male, Even=female)\n",
                "\n",
                "EMOTION_DICT = {\n",
                "    1: 'neutral',\n",
                "    2: 'calm',\n",
                "    3: 'happy',\n",
                "    4: 'sad',\n",
                "    5: 'angry',\n",
                "    6: 'fearful',\n",
                "    7: 'disgust',\n",
                "    8: 'surprised'\n",
                "}\n",
                "\n",
                "SENTIMENT_DICT = {\n",
                "    'neutral': 'Neutral',\n",
                "    'calm': 'Positive',\n",
                "    'happy': 'Positive',\n",
                "    'sad': 'Negative',\n",
                "    'angry': 'Negative',\n",
                "    'fearful': 'Negative',\n",
                "    'disgust': 'Negative',\n",
                "    'surprised': 'Neutral'\n",
                "}\n",
                "\n",
                "def parse_dataset(data_dir=\"../audio_speech_actors_01-24\"):\n",
                "    files = glob.glob(f\"{data_dir}/Actor_*/*.wav\")\n",
                "    if not files:\n",
                "        files = glob.glob(\"Actor_*/*.wav\")\n",
                "    \n",
                "    records = []\n",
                "    for f in sorted(files):\n",
                "        parts = Path(f).stem.split('-')\n",
                "        if len(parts) == 7:\n",
                "            emo_code = int(parts[2])\n",
                "            actor_id = int(parts[6])\n",
                "            emo = EMOTION_DICT.get(emo_code, 'unknown')\n",
                "            sent = SENTIMENT_DICT.get(emo, 'unknown')\n",
                "            records.append({\n",
                "                'filepath': f,\n",
                "                'emotion': emo,\n",
                "                'sentiment': sent,\n",
                "                'actor_id': actor_id,\n",
                "                'gender': 'female' if actor_id % 2 == 0 else 'male'\n",
                "            })\n",
                "    return pd.DataFrame(records)\n",
                "\n",
                "df = parse_dataset()\n",
                "print(f\"Total Audio Samples Found: {len(df)}\")\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.1 Distribution Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "sns.countplot(data=df, x='emotion', palette='viridis', ax=axes[0],\n",
                "              order=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'])\n",
                "axes[0].set_title('Distribution of Emotions in RAVDESS', fontsize=12, fontweight='bold')\n",
                "axes[0].tick_params(axis='x', rotation=45)\n",
                "\n",
                "sns.countplot(data=df, x='sentiment', palette=['#f43f5e', '#6366f1', '#10b981'], ax=axes[1],\n",
                "              order=['Negative', 'Neutral', 'Positive'])\n",
                "axes[1].set_title('Aggregated Sentiment Distribution', fontsize=12, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Audio Signal & Spectral Visualizations\n",
                "Comparing the acoustic profiles of high-energy vs somber emotions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_emotions = ['happy', 'angry', 'sad', 'calm']\n",
                "fig, axes = plt.subplots(len(sample_emotions), 2, figsize=(15, 10))\n",
                "\n",
                "for idx, emo in enumerate(sample_emotions):\n",
                "    sample_row = df[df['emotion'] == emo].iloc[0]\n",
                "    y, sr = librosa.load(sample_row['filepath'], sr=22050)\n",
                "    \n",
                "    # Waveform\n",
                "    librosa.display.waveshow(y, sr=sr, ax=axes[idx, 0], color='#6366f1')\n",
                "    axes[idx, 0].set_title(f\"Waveform: {emo.upper()} (Actor {sample_row['actor_id']})\", fontweight='bold')\n",
                "    axes[idx, 0].set_ylabel('Amplitude')\n",
                "    \n",
                "    # Mel Spectrogram\n",
                "    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)\n",
                "    S_dB = librosa.power_to_db(S, ref=np.max)\n",
                "    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=axes[idx, 1], cmap='magma')\n",
                "    axes[idx, 1].set_title(f\"Mel-Spectrogram: {emo.upper()}\", fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Acoustic Feature Extraction Pipeline (478 Dimensions)\n",
                "We compute statistical moments (mean, standard deviation, and peak) across temporal, spectral, and harmonic domains."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def extract_features(y, sr=22050):\n",
                "    if len(y) < sr * 0.5:\n",
                "        y = np.pad(y, (0, int(sr * 0.5) - len(y)), mode='constant')\n",
                "        \n",
                "    D = np.abs(librosa.stft(y))\n",
                "    S = D ** 2\n",
                "    \n",
                "    feats = []\n",
                "    \n",
                "    # 1. ZCR (3)\n",
                "    zcr = librosa.feature.zero_crossing_rate(y=y)\n",
                "    feats.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])\n",
                "    \n",
                "    # 2. RMS Energy (3)\n",
                "    rms = librosa.feature.rms(S=D)\n",
                "    feats.extend([np.mean(rms), np.std(rms), np.max(rms)])\n",
                "    \n",
                "    # 3. Spectral Centroid, Bandwidth, Rolloff (6)\n",
                "    cent = librosa.feature.spectral_centroid(S=D, sr=sr)\n",
                "    feats.extend([np.mean(cent), np.std(cent)])\n",
                "    bw = librosa.feature.spectral_bandwidth(S=D, sr=sr)\n",
                "    feats.extend([np.mean(bw), np.std(bw)])\n",
                "    ro = librosa.feature.spectral_rolloff(S=D, sr=sr)\n",
                "    feats.extend([np.mean(ro), np.std(ro)])\n",
                "    \n",
                "    # 4. Mel-Spectrogram 128 bands (256)\n",
                "    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=128)\n",
                "    mel_db = librosa.power_to_db(mel, ref=np.max)\n",
                "    feats.extend(np.mean(mel_db, axis=1))\n",
                "    feats.extend(np.std(mel_db, axis=1))\n",
                "    \n",
                "    # 5. MFCCs 40 coeff + Deltas (160)\n",
                "    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=40)\n",
                "    feats.extend(np.mean(mfcc, axis=1))\n",
                "    feats.extend(np.std(mfcc, axis=1))\n",
                "    mfcc_delta = librosa.feature.delta(mfcc)\n",
                "    feats.extend(np.mean(mfcc_delta, axis=1))\n",
                "    feats.extend(np.std(mfcc_delta, axis=1))\n",
                "    \n",
                "    # 6. Chroma STFT (24)\n",
                "    chroma = librosa.feature.chroma_stft(S=S, sr=sr)\n",
                "    feats.extend(np.mean(chroma, axis=1))\n",
                "    feats.extend(np.std(chroma, axis=1))\n",
                "    \n",
                "    # 7. Spectral Contrast (14)\n",
                "    try:\n",
                "        contrast = librosa.feature.spectral_contrast(S=D, sr=sr)\n",
                "        feats.extend(np.mean(contrast, axis=1))\n",
                "        feats.extend(np.std(contrast, axis=1))\n",
                "    except Exception:\n",
                "        feats.extend([0.0] * 14)\n",
                "        \n",
                "    # 8. Tonnetz (12)\n",
                "    try:\n",
                "        tonnetz = librosa.feature.tonnetz(chroma=chroma)\n",
                "        feats.extend(np.mean(tonnetz, axis=1))\n",
                "        feats.extend(np.std(tonnetz, axis=1))\n",
                "    except Exception:\n",
                "        feats.extend([0.0] * 12)\n",
                "        \n",
                "    arr = np.nan_to_num(np.array(feats, dtype=np.float32))\n",
                "    return arr\n",
                "\n",
                "print(f\"Feature dimension per sample: {len(extract_features(np.zeros(22050)))}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Load Pre-extracted Augmented Feature Matrix"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_npz = \"../data/ravdess_features.npz\"\n",
                "if not os.path.exists(feature_npz):\n",
                "    feature_npz = \"data/ravdess_features.npz\"\n",
                "\n",
                "data = np.load(feature_npz, allow_pickle=True)\n",
                "X = data['X']\n",
                "emotions = data['emotions']\n",
                "sentiments = data['sentiments']\n",
                "\n",
                "print(f\"Feature Matrix X Shape: {X.shape}\")\n",
                "print(f\"Total Augmented Instances: {len(emotions)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Model Training & Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load trained models and evaluation metrics\n",
                "models_dir = \"../models\" if os.path.exists(\"../models\") else \"models\"\n",
                "with open(f\"{models_dir}/metrics.json\", \"r\") as f:\n",
                "    metrics = json.load(f)\n",
                "\n",
                "print(f\"=== BEST SENTIMENT MODEL: {metrics['sentiment_model']} ===\")\n",
                "print(f\"Sentiment Accuracy: {metrics['sentiment_accuracy']*100:.2f}%\")\n",
                "print(f\"\\n=== BEST EMOTION MODEL: {metrics['emotion_model']} ===\")\n",
                "print(f\"Emotion Accuracy: {metrics['emotion_accuracy']*100:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6.1 Classification Reports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sent_report_df = pd.DataFrame(metrics['sentiment_report']).transpose()\n",
                "print(\"Sentiment Classification Report:\")\n",
                "display(sent_report_df)\n",
                "\n",
                "emo_report_df = pd.DataFrame(metrics['emotion_report']).transpose()\n",
                "print(\"\\nEmotion Classification Report:\")\n",
                "display(emo_report_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6.2 Confusion Matrices"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image, display\n",
                "display(Image(filename=f\"{models_dir}/confusion_matrix_sentiment.png\"))\n",
                "display(Image(filename=f\"{models_dir}/confusion_matrix_emotion.png\"))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Interactive Inference & Web App Integration\n",
                "Demonstrating end-to-end prediction from raw audio input using the SpeechSentimentPredictor class."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.append('..')\n",
                "from src.predict import SpeechSentimentPredictor\n",
                "\n",
                "predictor = SpeechSentimentPredictor(models_dir=models_dir)\n",
                "sample_test_file = glob.glob(f\"{models_dir}/../audio_speech_actors_01-24/Actor_01/*.wav\")[0]\n",
                "\n",
                "result = predictor.predict_from_file(sample_test_file)\n",
                "print(\"Sample Voice Prediction Result:\")\n",
                "print(json.dumps(result, indent=2))"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("c:/Users/jibra/Downloads/Project 6th sem/codefile/make1.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("make1.ipynb populated successfully.")

if __name__ == "__main__":
    create_notebook()
