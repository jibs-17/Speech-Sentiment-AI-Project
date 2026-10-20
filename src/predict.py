import os
import io
import joblib
import numpy as np
import librosa
import soundfile as sf
from src.extract_features import extract_features_from_audio

class SpeechSentimentPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.sent_model_path = os.path.join(models_dir, "sentiment_model.joblib")
        self.emo_model_path = os.path.join(models_dir, "emotion_model.joblib")
        self.scaler_path = os.path.join(models_dir, "scaler.joblib")
        self.encoders_path = os.path.join(models_dir, "encoders.joblib")
        
        self.sent_model = None
        self.emo_model = None
        self.scaler = None
        self.encoders = None
        self._load_models()
        
    def _load_models(self):
        # Prefer full ensemble models, fallback to lightweight models (for Vercel deployment)
        if not os.path.exists(self.sent_model_path):
            self.sent_model_path = os.path.join(self.models_dir, "sentiment_model_light.joblib")
        if not os.path.exists(self.emo_model_path):
            self.emo_model_path = os.path.join(self.models_dir, "emotion_model_light.joblib")

        if not all(os.path.exists(p) for p in [self.sent_model_path, self.emo_model_path, self.scaler_path, self.encoders_path]):
            raise FileNotFoundError(f"One or more model files missing in {self.models_dir}. Train models first.")
            
        self.sent_model = joblib.load(self.sent_model_path)
        self.emo_model = joblib.load(self.emo_model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.encoders = joblib.load(self.encoders_path)
        
        self.emotion_classes = list(self.encoders['emotion_classes'])
        self.sentiment_classes = list(self.encoders['sentiment_classes'])
        print(f"SpeechSentimentPredictor loaded successfully ({os.path.basename(self.sent_model_path)}).")
        
    def predict_from_audio(self, y, sr=22050):
        # Guarantee minimum length (at least 1.0 sec = 22050 samples) to prevent framing errors
        if len(y) < sr * 1.0:
            pad_len = int(sr * 1.0) - len(y)
            y = np.pad(y, (0, pad_len), mode='constant')

        # Trim silence safely
        try:
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            if len(y_trimmed) < sr * 0.5:
                y_trimmed = y
        except Exception:
            y_trimmed = y
            
        duration = float(len(y) / sr)
        
        # Extract features
        features = extract_features_from_audio(y_trimmed, sr=sr)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Sentiment prediction
        sent_preds = self.sent_model.predict(features_scaled)[0]
        sent_top_label = str(self.sentiment_classes[sent_preds] if isinstance(sent_preds, (int, np.integer)) else sent_preds)
        
        sent_probs = {}
        if hasattr(self.sent_model, "predict_proba"):
            probs = self.sent_model.predict_proba(features_scaled)[0]
            for cls_idx, p in enumerate(probs):
                cls_name = str(self.sentiment_classes[cls_idx])
                sent_probs[cls_name] = round(float(p) * 100, 2)
            sent_conf = float(np.max(probs)) * 100
        else:
            sent_probs[sent_top_label] = 100.0
            sent_conf = 100.0
            
        # Emotion prediction
        emo_preds = self.emo_model.predict(features_scaled)[0]
        emo_top_label = str(self.emotion_classes[emo_preds] if isinstance(emo_preds, (int, np.integer)) else emo_preds)
        
        emo_probs = {}
        if hasattr(self.emo_model, "predict_proba"):
            probs = self.emo_model.predict_proba(features_scaled)[0]
            for cls_idx, p in enumerate(probs):
                cls_name = str(self.emotion_classes[cls_idx])
                emo_probs[cls_name] = round(float(p) * 100, 2)
            emo_conf = float(np.max(probs)) * 100
        else:
            emo_probs[emo_top_label] = 100.0
            emo_conf = 100.0
            
        # Audio acoustic indicators for UI
        rms_val = float(np.mean(librosa.feature.rms(y=y_trimmed)))
        zcr_val = float(np.mean(librosa.feature.zero_crossing_rate(y=y_trimmed)))
        
        return {
            'sentiment': {
                'label': str(sent_top_label),
                'confidence': round(sent_conf, 2),
                'probabilities': sent_probs
            },
            'emotion': {
                'label': str(emo_top_label),
                'confidence': round(emo_conf, 2),
                'probabilities': emo_probs
            },
            'audio_stats': {
                'duration_seconds': round(duration, 2),
                'energy_level': "High" if rms_val > 0.04 else ("Moderate" if rms_val > 0.015 else "Low"),
                'rms': round(rms_val, 4),
                'zcr': round(zcr_val, 4)
            }
        }
        
    def predict_from_file(self, file_path_or_bytes):
        target_sr = 22050
        y = None
        sr = target_sr
        
        # 1. First attempt: librosa standard load
        try:
            if isinstance(file_path_or_bytes, (str, os.PathLike)):
                y, sr = librosa.load(file_path_or_bytes, sr=target_sr, mono=True)
            else:
                bio = io.BytesIO(file_path_or_bytes)
                y, sr = librosa.load(bio, sr=target_sr, mono=True)
        except Exception:
            pass

        # 2. Second attempt: soundfile standard read
        if y is None:
            try:
                if isinstance(file_path_or_bytes, (str, os.PathLike)):
                    data, in_sr = sf.read(file_path_or_bytes, dtype='float32')
                else:
                    bio = io.BytesIO(file_path_or_bytes)
                    data, in_sr = sf.read(bio, dtype='float32')
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if in_sr != target_sr:
                    data = librosa.resample(data, orig_sr=in_sr, target_sr=target_sr)
                y = data
                sr = target_sr
            except Exception:
                pass

        # 3. Third attempt: ffmpeg universal decoder (handles M4A, AAC, WEBM, OPUS, 3GP, MP3, etc.)
        if y is None:
            try:
                import imageio_ffmpeg
                import subprocess
                import tempfile
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

                in_created = False
                if not isinstance(file_path_or_bytes, (str, os.PathLike)):
                    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as in_tmp:
                        in_tmp.write(file_path_or_bytes)
                        in_path = in_tmp.name
                    in_created = True
                else:
                    in_path = file_path_or_bytes

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
                    out_path = out_tmp.name

                cmd = [
                    ffmpeg_exe, "-y",
                    "-i", str(in_path),
                    "-ac", "1",
                    "-ar", str(target_sr),
                    out_path
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                data, _ = sf.read(out_path, dtype='float32')
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                y = data
                sr = target_sr

                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                if in_created and os.path.exists(in_path):
                    try:
                        os.remove(in_path)
                    except Exception:
                        pass
            except Exception as ffmpeg_err:
                raise RuntimeError(f"Unable to read audio format. Please provide WAV or MP3 audio. ({ffmpeg_err})")

        if y is None or len(y) == 0:
            raise ValueError("Audio recording is empty or could not be decoded.")

        return self.predict_from_audio(y, sr)

if __name__ == "__main__":
    predictor = SpeechSentimentPredictor()
    sample_file = "audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
    result = predictor.predict_from_file(sample_file)
    print("Sample Prediction Result:")
    import json
    print(json.dumps(result, indent=2))
