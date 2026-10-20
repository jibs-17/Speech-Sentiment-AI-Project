import os
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import socket
import qrcode
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.predict import SpeechSentimentPredictor

app = FastAPI(
    title="Speech Sentiment & Emotion AI",
    description="Deep Acoustic Sentiment Analysis on RAVDESS Speech Dataset",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "audio_speech_actors_01-24"

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount models directory so confusion matrix images can be viewed
if MODELS_DIR.exists():
    app.mount("/model-assets", StaticFiles(directory=str(MODELS_DIR)), name="model-assets")

# Initialize predictor lazily
predictor: Optional[SpeechSentimentPredictor] = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = SpeechSentimentPredictor(models_dir=str(MODELS_DIR))
        except Exception as e:
            print(f"Warning: Models not loaded yet ({e})")
    return predictor

def get_network_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend template not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/api/network-info")
async def get_network_info():
    ip = get_network_ip()
    port = int(os.environ.get("PORT", 8000))
    return JSONResponse({
        "local_ip": ip,
        "port": port,
        "local_url": f"http://127.0.0.1:{port}",
        "network_url": f"http://{ip}:{port}"
    })

@app.get("/api/qr-code")
async def get_qr_code():
    ip = get_network_ip()
    port = int(os.environ.get("PORT", 8000))
    target_url = f"http://{ip}:{port}"
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(target_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#4f46e5", back_color="#ffffff")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return StreamingResponse(bio, media_type="image/png")

@app.get("/api/metrics")
async def get_metrics():
    metrics_file = MODELS_DIR / "metrics.json"
    if not metrics_file.exists():
        return JSONResponse({"status": "training_or_not_found"})
    with open(metrics_file, "r") as f:
        data = json.load(f)
    return JSONResponse(data)

@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...)):
    pred_engine = get_predictor()
    if pred_engine is None:
        raise HTTPException(status_code=503, detail="Model is still training or not available yet.")
        
    # Read uploaded bytes
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file received.")
        
    # Write to a temporary file with appropriate extension to ensure librosa / soundfile can read it
    file_ext = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        result = pred_engine.predict_from_file(tmp_path)
        result['filename'] = file.filename or "uploaded_audio"
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# Demo sample audio endpoints so users can test immediately with 1 click
DEMO_SAMPLES = {
    "happy": "Actor_01/03-01-03-01-01-01-01.wav",     # Happy, normal intensity
    "angry": "Actor_01/03-01-05-02-01-01-01.wav",     # Angry, strong intensity
    "sad": "Actor_01/03-01-04-01-01-01-01.wav",       # Sad, normal intensity
    "calm": "Actor_01/03-01-02-01-01-01-01.wav",      # Calm, normal intensity
    "surprised": "Actor_01/03-01-08-01-01-01-01.wav"  # Surprised, normal intensity
}

@app.get("/api/samples")
async def list_samples():
    available = []
    for emotion_key, rel_path in DEMO_SAMPLES.items():
        full_path = DATA_DIR / rel_path
        if not full_path.exists():
            full_path = BASE_DIR / rel_path
        if full_path.exists():
            available.append({
                "id": emotion_key,
                "label": emotion_key.capitalize(),
                "file": rel_path
            })
    return JSONResponse(available)

@app.get("/api/sample-audio/{sample_id}")
async def get_sample_audio(sample_id: str):
    if sample_id not in DEMO_SAMPLES:
        raise HTTPException(status_code=404, detail="Sample not found")
        
    path = DATA_DIR / DEMO_SAMPLES[sample_id]
    if not path.exists():
        path = BASE_DIR / DEMO_SAMPLES[sample_id]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample audio file not found on disk")
        
    return FileResponse(path, media_type="audio/wav", filename=f"{sample_id}_sample.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
