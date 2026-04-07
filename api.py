import os
import subprocess
import traceback
import logging
import sys
import gc
import builtins
import torch
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
# from deepface import DeepFace (lazy loaded below)
from supabase import create_client, Client
from dotenv import load_dotenv
import cv2
import base64
from scipy.spatial.distance import cosine
import tempfile
import shutil
import time
import wave
import struct
import threading
import torch
import numpy as np
import io
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import torchaudio

# Global lock to protect CPU exhaustion during concurrent CPU-bound AI operations
tts_lock = threading.Lock()
# ------------------------------
# ENVIRONMENT DETECTION
# ------------------------------
# We detect if we are running locally on Windows (constrained 4GB RAM)
# or on a full Cloud environment like Render or Hugging Face.
IS_LOCAL_WINDOWS = os.name == 'nt' and not os.environ.get("RENDER") and not os.environ.get("SPACE_ID")

def get_memory_usage():
    """Print current memory usage if psutil is available."""
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        return f"{mem_mb:.2f} MB"
    except ImportError:
        return "Unknown (psutil missing)"

def clear_memory():
    """Aggressively clear RAM by flushing both Torch and TensorFlow and forcing GC."""
    print(f"🧹 Clearing memory... Current usage approx: {get_memory_usage()}")
    
    # 1. Clear Torch Cache (if torch is already loaded)
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # 2. Clear TensorFlow/Keras Sessions (if DeepFace was used)
    if 'tensorflow' in sys.modules:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except: pass
        
    # 3. Force Python Garbage Collection
    gc.collect()
    gc.collect()
    print(f"✅ Memory cleared. Usage now: {get_memory_usage()}")

# Apply extreme limitations ONLY if local on Windows
if IS_LOCAL_WINDOWS:
    print("⚠️  [LOCAL MODE] Applying 4GB RAM optimizations (Thread & Memory limits)")
    # Set torch to single thread to save on system handles (Fixes WinError 1450)
    torch.set_num_threads(1)
else:
    print("🚀 [CLOUD MODE] Running with full system resources")

# Force unbuffered output for prints
_original_print = builtins.print
def _flushed_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)
builtins.print = _flushed_print

# Accept Coqui TOS automatically to prevent EOFError on input()
os.environ["COQUI_TOS_AGREED"] = "1"

# Automatically answer 'y' to any hidden interactive prompts to firmly avoid EOFError
builtins.input = lambda prompt="": "y"

def ensure_wav(input_path: str) -> str:
    """
    Forcefully normalize ANY incoming audio to 22050Hz 1-channel WAV using ffmpeg,
    preventing crashes caused when browsers upload WebM files falsly named '.wav'.
    """
    out_fd, out_path = tempfile.mkstemp(suffix='.wav')
    os.close(out_fd)
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', input_path, '-ar', '22050', '-ac', '1', out_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        os.remove(out_path)
        raise RuntimeError(f'ffmpeg conversion failed: {result.stderr}')
    return out_path

# Load Env Vars
load_dotenv()

# Configuration
# Dynamic path for models
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models")
if os.path.exists("F:/Amos/AI_models"):
    MODEL_PATH = "F:/Amos/AI_models"
    os.environ["DEEPFACE_HOME"] = MODEL_PATH
else:
    # We are on the Cloud (Docker)
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.environ["DEEPFACE_HOME"] = MODEL_PATH

# Set HF_HOME for weights
os.environ["HF_HOME"] = os.path.join(MODEL_PATH, "huggingface")

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on: {DEVICE}")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Prevent TF from pre-allocating all memory if GPU is present (even if running on CPU, it's good practice)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Initialize Supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
if not url or not key:
    print("⚠️ WARNING: Missing SUPABASE_URL or SUPABASE_KEY")
else:
    supabase: Client = create_client(url, key)

app = FastAPI(title="Cloud Biometric & Voice AI Engine")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# ENCRYPTION STUFF
# -------------------------------------------------------
def get_aes_gcm():
    secret = os.environ.get("SUPABASE_KEY")
    if not secret:
        raise ValueError("Missing SUPABASE_KEY for encryption")
    key = hashlib.sha256(secret.encode('utf-8')).digest()
    return AESGCM(key)

def encrypt_bytes(data: bytes) -> bytes:
    if USE_MOCK_TTS: return data # Skip in mock mode
    aesgcm = get_aes_gcm()
    nonce = os.urandom(12)
    return nonce + aesgcm.encrypt(nonce, data, None)

def decrypt_bytes(data: bytes) -> bytes:
    if USE_MOCK_TTS or len(data) < 12: return data # Skip in mock mode
    aesgcm = get_aes_gcm()
    nonce = data[:12]
    ciphertext = data[12:]
    return aesgcm.decrypt(nonce, ciphertext, None)


# -------------------------------------------------------
# MOCK TTS — activated by env var USE_MOCK_TTS=1
# Lets you debug Supabase / file-handling logic locally
# without loading the real XTTS v2 model (saves ~6 GB RAM).
# -------------------------------------------------------
USE_MOCK_TTS = os.environ.get("USE_MOCK_TTS", "0") == "1"

if USE_MOCK_TTS:
    print("🟡 [MOCK MODE] USE_MOCK_TTS=1 — Real TTS model will NOT be loaded.")
    print("   All audio endpoints will return a silent stub WAV file.")
    print("   Supabase storage, file handling & background tasks are FULLY live.")

def _make_silent_wav(path: str, duration_seconds: float = 1.0, sample_rate: int = 22050):
    """Write a minimal silent mono WAV file to `path`."""
    num_samples = int(sample_rate * duration_seconds)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)        # mono
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack('<' + 'h' * num_samples, *([0] * num_samples)))

class _MockConditioningLatent:
    """Tiny stand-in for a real XTTS conditioning tensor."""
    def __init__(self):
        import torch as _torch
        self._data = _torch.zeros(1, 1, 1024)  # minimal shape

    def cpu(self):
        return self

    def numpy(self):
        return self._data.numpy()

    @property
    def shape(self):
        return self._data.shape

class MockTTS:
    """
    A zero-weight replacement for the real Coqui TTS engine.
    Mirrors the API surface used in this file so all surrounding
    production logic (Supabase uploads, file cleanup, etc.) runs
    identically — without touching the XTTS model or consuming GPU memory.
    """

    class _MockModel:
        """Mirrors engine.model used in /audio/register and /audio/clone."""

        def get_conditioning_latents(self, audio_path):
            print(f"[MOCK] get_conditioning_latents called with: {audio_path}")
            gpt  = _MockConditioningLatent()
            spkr = _MockConditioningLatent()
            gpt._data  = gpt._data.reshape(1, 1400, 1024)   # realistic shape
            spkr._data = spkr._data.reshape(1, 1,    512)   # realistic shape
            return gpt, spkr

        def inference(self, text, language, gpt_cond_latent,
                      speaker_embedding, file_path, **kwargs):
            print(f"[MOCK] inference() called — text='{text[:40]}...' → {file_path}")
            _make_silent_wav(file_path)

    def __init__(self):
        self.model = self._MockModel()
        print("✅ [MOCK] MockTTS initialised — no model weights loaded.")

    def tts_to_file(self, text, speaker_wav, language, file_path, **kwargs):
        print(f"[MOCK] tts_to_file() called — text='{text[:40]}...' → {file_path}")
        _make_silent_wav(file_path)

    def to(self, device):
        """Chainable no-op so MockTTS.to(device) works like the real TTS."""
        return self


# Lazy load TTS to avoid slow startup for other endpoints
tts = None

def get_tts():
    global tts
    if tts is None:
        if USE_MOCK_TTS:
            print("🟡 [MOCK] Returning MockTTS instance (real model skipped).")
            tts = MockTTS()
        else:
            print(f"🎙️ Loading XTTS v2... [Initial Memory: {get_memory_usage()}]")
            try:
                from TTS.api import TTS

                # Explicitly force CPU + no deepspeed for local memory stability
                device = DEVICE
                use_deepspeed = False if IS_LOCAL_WINDOWS else (DEVICE == "cuda")

                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                print(f"✅ TTS Loaded successfully! [Memory: {get_memory_usage()}]")
            except Exception as e:
                print(f"❌ Failed to load TTS: {e}")
                import traceback
                traceback.print_exc()
                raise e
    return tts

# Setup Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
api_log = logging.getLogger("api")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    api_log.info(f"📥 REQUEST IN: {request.method} {request.url.path}")
    api_log.info(f"   Headers: {dict(request.headers)}")
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        api_log.info(f"📤 REQUEST OUT: {request.method} {request.url.path} - Status: {response.status_code} - Took {duration:.2f}s")
        return response
    except Exception as e:
        duration = time.time() - start_time
        api_log.error(f"❌ CRITICAL CRASH: {request.method} {request.url.path} failed after {duration:.2f}s")
        api_log.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={
            "success": False, 
            "error": "Internal Server Error", 
            "details": str(e),
            "traceback": traceback.format_exc()
        })

@app.get("/")
def home():
    return {
        "status": "online", 
        "system": "Cloud-Native Biometric & Voice Engine ☁️", 
        "device": DEVICE,
        "model_path": MODEL_PATH
    }

@app.post("/register")
async def register_face(
    name: str = Form(...), 
    user_id: str = Form(...),
    workspace_id: str = Form(...),
    file: UploadFile = File(...)
):
    temp_filename = ""
    try:
        file_content = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            temp_filename = tmp.name

        # Generate Embedding
        from deepface import DeepFace
        embeddings = DeepFace.represent(
            img_path=temp_filename, 
            model_name="Facenet512", 
            enforce_detection=False
        )
        embedding_vector = embeddings[0]["embedding"]

        # Upload to Supabase Storage
        storage_path = f"{user_id}/{workspace_id}/{name}{suffix}"
        supabase.storage.from_("biometric_faces").upload(
            path=storage_path,
            file=file_content,
            file_options={"content-type": file.content_type, "upsert": "true"}
        )

        # Upsert Metadata + Vector (allows Updating Face ID)
        data = {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "name": name,
            "image_path": storage_path,
            "embedding": embedding_vector,
            "created_at": "now()"
        }
        supabase.table("face_embeddings").upsert(data, on_conflict="user_id,workspace_id,name").execute()

        return {"success": True, "message": f"Cloud enrollment complete for {name}", "image_path": storage_path}

    except Exception as e:
        print(f"❌ Error in register: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/verify")
async def verify_face(
    user_id: str = Form(...),
    workspace_id: str = Form(...),
    file: UploadFile = File(...)
):
    temp_filename = ""
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_filename = tmp.name

        # Liveness / Emotion check
        from deepface import DeepFace
        emotion = "unknown"
        try:
            analysis = DeepFace.analyze(temp_filename, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list): analysis = analysis[0]
            emotion = analysis['dominant_emotion']
        except: pass

        # Generate Target Embedding
        target_embedding = DeepFace.represent(
            img_path=temp_filename, 
            model_name="Facenet512", 
            enforce_detection=False
        )[0]["embedding"]

        # Vector Search
        rpc_params = {
            "query_embedding": target_embedding,
            "match_threshold": 0.4,
            "filter_workspace_id": workspace_id
        }
        response = supabase.rpc("match_faces", rpc_params).execute()
        matches = response.data

        if matches and len(matches) > 0:
            best_match = matches[0]
            return {
                "access": "GRANTED",
                "user": best_match['name'],
                "confidence": round(best_match['similarity'] * 100, 2),
                "emotion_detected": emotion
            }
        else:
            return JSONResponse(status_code=401, content={"access": "DENIED", "error": "Identity Unknown"})

    except Exception as e:
        print(f"❌ Error in verify: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)

import uuid

ACTIVE_JOBS = {}

def _friendly_error(e: Exception) -> str:
    """Map raw Python exceptions to clean, informative user-facing messages."""
    msg = str(e).lower()
    if "no voice registered" in msg:
        return "No voice profile found. Please complete voice enrollment first."
    if "empty audio" in msg or "empty file" in msg:
        return "The audio file appears to be empty. Please record again."
    if "ffmpeg" in msg or "conversion failed" in msg:
        return "Audio format not supported. Please upload a WAV or WebM file."
    if "timeout" in msg or "timed out" in msg:
        return "The synthesis engine took too long. Please try again with shorter text."
    if "out of memory" in msg or "oom" in msg:
        return "The server ran out of memory. Please try again in a few moments."
    if "connection" in msg or "network" in msg:
        return "Could not connect to storage. Please check your connection and retry."
    if "supabase" in msg or "postgrest" in msg:
        return "A database error occurred. Please try again."
    if "not found" in msg or "404" in msg:
        return "Required resource was not found. Please re-enroll your voice."
    return "Something went wrong during processing. Please try again."

# Lazy load Whisper for STT (Anti-Replay)
stt_model = None
def get_stt_model():
    global stt_model
    if stt_model is None:
        try:
            import whisper
            print(f"🎙️ Loading Whisper Tiny... [Memory: {get_memory_usage()}]")
            # Using 'tiny.en' to save RAM on 4GB Windows environments
            stt_model = whisper.load_model("tiny.en") 
            print(f"✅ STT Model Loaded. [Memory: {get_memory_usage()}]")
        except Exception as e:
            print(f"⚠️ STT Loading Failed: {e}")
            return None
    return stt_model

spk_model = None
def get_spk_model():
    global spk_model
    if spk_model is None:
        try:
            print(f"🎙️ Loading SpeechBrain Speaker Recognition... [Memory: {get_memory_usage()}]")
            from speechbrain.inference.speaker import EncoderClassifier
            run_opts = {"device": DEVICE} if not IS_LOCAL_WINDOWS else {"device": "cpu"}
            spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts=run_opts)
            print(f"✅ SpeechBrain Loaded. [Memory: {get_memory_usage()}]")
        except Exception as e:
            print(f"⚠️ SpeechBrain Loading Failed: {e}")
            return None
    return spk_model

@app.post("/audio/verify")
async def verify_voice(
    user_id: str = Form(...),
    challenge_code: str = Form(...),
    file: UploadFile = File(...)
):
    print(f"[AUDIO VERIFY] User: {user_id}. Challenge: {challenge_code}")
    
    file_bytes = await file.read()
    if not file_bytes:
        return JSONResponse(status_code=400, content={"access": "DENIED", "error": "Empty audio file"})

    temp_ref = ""
    wav_path = ""
    try:
        # 1. Prepare Audio
        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_ref = tmp.name
        
        wav_path = ensure_wav(temp_ref)

        # 2. Speaker Verification (Identity Check)
        # Fetch enrolled embeddings from Supabase
        record_resp = supabase.table("voice_embeddings").select("gpt_cond_latent_path").eq("user_id", user_id).execute()
        if not record_resp.data:
            return JSONResponse(status_code=404, content={"access": "DENIED", "error": "No voice profile enrolled"})
        
        record = record_resp.data[0]
        sb_path = record["gpt_cond_latent_path"].replace("gpt_cond_latent.npy", "speechbrain.npy").replace(".enc", "") + ".enc"
        
        try:
            sb_bytes = supabase.storage.from_("biometric_faces").download(sb_path)
            sb_bytes = decrypt_bytes(sb_bytes)
            stored_embedding = np.load(io.BytesIO(sb_bytes))
        except Exception as e:
            # Fallback to old format or handle missing 
            return JSONResponse(status_code=500, content={"access": "ERROR", "error": "Could not retrieve voice profile. It may need to be re-enrolled for extreme accuracy."})

        # Extract live embedding using SpeechBrain
        spk_classifier = get_spk_model()
        if not spk_classifier:
            return JSONResponse(status_code=500, content={"access": "ERROR", "error": "SpeechBrain model not available"})
            
        with tts_lock:
            signal, fs = torchaudio.load(wav_path)
            live_embedding_tensor = spk_classifier.encode_batch(signal)
            live_embedding = live_embedding_tensor.cpu().numpy().flatten()

        # Cosine Similarity
        stored_embedding = stored_embedding.flatten()
        similarity = 1 - cosine(stored_embedding, live_embedding)
        print(f"🎙️ Identity Similarity (SpeechBrain): {similarity:.4f}")

        # IDENTITY THRESHOLD (0.8 as planned)
        if similarity < 0.8:
            return JSONResponse(status_code=401, content={"access": "DENIED", "error": "Voice mismatch", "similarity": round(float(similarity), 4)})

        # 3. STT Verification (Anti-Replay Security)
        stt = get_stt_model()
        if stt:
            print(f"🎙️ Transcribing for challenge code verification...")
            transcription_result = stt.transcribe(wav_path)
            text = transcription_result["text"].lower()
            print(f"🎙️ Transcribed: '{text}'")

            # Clean challenge code and check for its presence
            digits = "".join(filter(str.isdigit, challenge_code))
            # Also check for word version of digits if needed, but digits is usually enough for numeric codes
            # Simple check: does the text contain the digits sequence or individual digits?
            # For a 6-digit code like "123456", we look for "123456" or "1 2 3 4 5 6"
            cleaned_text = "".join(filter(str.isdigit, text))
            
            if digits not in cleaned_text:
                return JSONResponse(status_code=401, content={
                    "access": "DENIED", 
                    "error": "Challenge code mismatch", 
                    "details": f"Expected {digits}, heard something else."
                })
        else:
            print("⚠️ STT Engine unavailable, skipping phrase check (Liveness degraded)")

        return {
            "access": "GRANTED",
            "message": "Voice access authorized",
            "similarity": round(float(similarity), 4)
        }

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"access": "ERROR", "error": _friendly_error(e)})
    finally:
        if temp_ref and os.path.exists(temp_ref): os.remove(temp_ref)
        if wav_path and wav_path != temp_ref and os.path.exists(wav_path): os.remove(wav_path)

@app.get("/audio/status/{job_id}")
async def get_audio_status(job_id: str):
    job = ACTIVE_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"success": False, "error": "Job not found or expired"})
    return job

@app.post("/audio/register")
async def register_voice(
    user_id: str = Form(...),
    webhook_url: str = Form(None),
    reference_audio: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    print(f"[AUDIO REGISTER] Started. User: {user_id}. Webhook: {webhook_url}")
    print(f"   - Filename: {reference_audio.filename}, Content-Type: {reference_audio.content_type}")

    # Read the upload immediately (must happen in the async context)
    file_bytes = await reference_audio.read()
    if not file_bytes:
        return JSONResponse(status_code=400, content={"success": False, "error": "Empty audio file received"})

    original_name = reference_audio.filename or 'audio.webm'
    suffix = os.path.splitext(original_name)[1] or '.webm'

    # Persist to disk so the background thread can read it
    tmp_fd, temp_ref = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(tmp_fd, 'wb') as f:
        f.write(file_bytes)

    job_id = str(uuid.uuid4())
    ACTIVE_JOBS[job_id] = {"status": "PROCESSING", "message": "Enrolling voice profile..."}
    print(f"[AUDIO REGISTER] Queued as job {job_id}")

    def _run_register(job_id: str, temp_ref: str, user_id: str, webhook_url: str):
        wav_path = ""
        try:
            with tts_lock:
                wav_path = ensure_wav(temp_ref)
                print(f"[{job_id}] 🎵 WAV ready: {wav_path}")
    
                engine = get_tts()
                if hasattr(engine, "synthesizer"):
                    gpt_cond_latent, speaker_embedding = engine.synthesizer.tts_model.get_conditioning_latents(audio_path=wav_path)
                else:
                    gpt_cond_latent, speaker_embedding = engine.model.get_conditioning_latents(audio_path=wav_path)
                    
                spk_classifier = get_spk_model()
                if spk_classifier:
                    signal, fs = torchaudio.load(wav_path)
                    speechbrain_embedding = spk_classifier.encode_batch(signal)
                else:
                    speechbrain_embedding = None

            import io, numpy as np
            def tensor_to_npy_bytes(t):
                buf = io.BytesIO()
                np.save(buf, t.cpu().numpy())
                return buf.getvalue()

            gpt_bytes = encrypt_bytes(tensor_to_npy_bytes(gpt_cond_latent))
            spk_bytes = encrypt_bytes(tensor_to_npy_bytes(speaker_embedding))
            sb_bytes = encrypt_bytes(tensor_to_npy_bytes(speechbrain_embedding)) if speechbrain_embedding is not None else None

            gpt_path = f"voice-latents/{user_id}/gpt_cond_latent.npy.enc"
            spk_path = f"voice-latents/{user_id}/speaker_embedding.npy.enc"
            sb_path = f"voice-latents/{user_id}/speechbrain.npy.enc"

            supabase.storage.from_("biometric_faces").upload(
                path=gpt_path, file=gpt_bytes,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
            supabase.storage.from_("biometric_faces").upload(
                path=spk_path, file=spk_bytes,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
            if sb_bytes:
                supabase.storage.from_("biometric_faces").upload(
                    path=sb_path, file=sb_bytes,
                    file_options={"content-type": "application/octet-stream", "upsert": "true"}
                )
                
            supabase.table("voice_embeddings").upsert({
                "user_id": user_id,
                "gpt_cond_latent_path": gpt_path,
                "speaker_embedding_path": spk_path,
                "created_at": "now()"
            }, on_conflict="user_id").execute()

            print(f"[{job_id}] ✅ Voice registered for {user_id}")
            res = {"status": "COMPLETED", "success": True, "message": "Voice profile enrolled successfully"}
            ACTIVE_JOBS[job_id] = res
            if webhook_url:
                import requests
                try: requests.post(webhook_url, json={"jobId": job_id, **res}, timeout=5)
                except Exception as e: print(f"Webhook error: {e}")
        except Exception as e:
            print(f"[{job_id}] ❌ Register failed: {e}")
            print(traceback.format_exc())
            err = {"status": "FAILED", "success": False, "error": _friendly_error(e)}
            ACTIVE_JOBS[job_id] = err
            if webhook_url:
                import requests
                try: requests.post(webhook_url, json={"jobId": job_id, **err}, timeout=5)
                except: pass
        finally:
            if temp_ref and os.path.exists(temp_ref): os.remove(temp_ref)
            if wav_path and wav_path != temp_ref and os.path.exists(wav_path): os.remove(wav_path)

    background_tasks.add_task(_run_register, job_id, temp_ref, user_id, webhook_url)
    return {"success": True, "jobId": job_id, "status": "PROCESSING"}

@app.post("/audio/clone")
async def clone_voice(
    text: str = Form(...),
    user_id: str = Form(None),
    webhook_url: str = Form(None),
    reference_audio: UploadFile = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    print(f"[AUDIO CLONE] Started. User: {user_id}. Webhook: {webhook_url}. Text Length: {len(text)}")

    # Read upload immediately in the async context
    temp_ref = ""
    file_bytes = None
    if reference_audio:
        file_bytes = await reference_audio.read()
        original_name = reference_audio.filename or 'audio.webm'
        suffix = os.path.splitext(original_name)[1] or '.webm'
        tmp_fd, temp_ref = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(tmp_fd, 'wb') as f:
            f.write(file_bytes)
    elif not user_id:
        return JSONResponse(status_code=400, content={"success": False, "error": "Either reference_audio or user_id must be provided"})

    job_id = str(uuid.uuid4())
    ACTIVE_JOBS[job_id] = {"status": "PROCESSING", "message": "Synthesizing your voice clone..."}
    print(f"[AUDIO CLONE] Queued as job {job_id}")

    def _run_clone(job_id: str, text: str, user_id: str, temp_ref: str, webhook_url: str):
        temp_out = ""
        wav_ref = temp_ref
        try:
            with tts_lock:
                engine = get_tts()
                gpt_cond_latent = None
                speaker_embedding = None
    
                if wav_ref:
                    wav_ref = ensure_wav(temp_ref)
                elif user_id:
                    import io, numpy as np
                    print(f"[{job_id}] 🔍 Looking up voice latents for {user_id}...")
                    record_resp = supabase.table("voice_embeddings").select("gpt_cond_latent_path,speaker_embedding_path").eq("user_id", user_id).execute()
                    if not record_resp.data:
                        ACTIVE_JOBS[job_id] = {"status": "FAILED", "success": False, "error": "No voice profile found. Please complete voice enrollment first."}
                        return
                    record = record_resp.data[0]
                    gpt_bytes = supabase.storage.from_("biometric_faces").download(record["gpt_cond_latent_path"])
                    spk_bytes = supabase.storage.from_("biometric_faces").download(record["speaker_embedding_path"])
                    
                    gpt_bytes = decrypt_bytes(gpt_bytes)
                    spk_bytes = decrypt_bytes(spk_bytes)
                    
                    gpt_cond_latent = torch.tensor(np.load(io.BytesIO(gpt_bytes))).to(DEVICE)
                    speaker_embedding = torch.tensor(np.load(io.BytesIO(spk_bytes))).to(DEVICE)
                    print(f"[{job_id}] ✅ Loaded and decrypted voice latents")

            out_fd, temp_out = tempfile.mkstemp(suffix=".wav")
            os.close(out_fd)

            if gpt_cond_latent is not None:
                if hasattr(engine, "synthesizer"):
                    import torchaudio
                    out = engine.synthesizer.tts_model.inference(
                        text=text, language="en",
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding
                    )
                    wav = torch.tensor(out["wav"])
                    if wav.dim() == 1:
                        wav = wav.unsqueeze(0)
                    torchaudio.save(temp_out, wav, 24000)
                else:
                    engine.model.inference(
                        text=text, language="en",
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        file_path=temp_out
                    )
            else:
                engine.tts_to_file(text=text, speaker_wav=wav_ref, language="en", file_path=temp_out)

            # Upload to Supabase Storage
            import io as _io
            with open(temp_out, "rb") as f:
                audio_bytes = f.read()

            file_name = f"voice-clones/{user_id}/{uuid.uuid4()}.wav"
            supabase.storage.from_("biometric_faces").upload(
                path=file_name, file=audio_bytes,
                file_options={"content-type": "audio/wav", "upsert": "true"}
            )
            out_url = supabase.storage.from_("biometric_faces").get_public_url(file_name)

            print(f"[{job_id}] ✅ Synthesis complete -> {out_url}")
            res = {"status": "COMPLETED", "success": True, "audioUrl": out_url}
            ACTIVE_JOBS[job_id] = res
            if webhook_url:
                import requests
                try: requests.post(webhook_url, json={"jobId": job_id, **res}, timeout=5)
                except Exception as e: print(f"Webhook error: {e}")
        except Exception as e:
            print(f"[{job_id}] ❌ Clone failed: {e}")
            print(traceback.format_exc())
            err = {"status": "FAILED", "success": False, "error": _friendly_error(e)}
            ACTIVE_JOBS[job_id] = err
            if webhook_url:
                import requests
                try: requests.post(webhook_url, json={"jobId": job_id, **err}, timeout=5)
                except: pass
        finally:
            if wav_ref and wav_ref != temp_ref and os.path.exists(wav_ref): os.remove(wav_ref)
            if temp_ref and os.path.exists(temp_ref): os.remove(temp_ref)
            if temp_out and os.path.exists(temp_out): os.remove(temp_out)

    background_tasks.add_task(_run_clone, job_id, text, user_id, temp_ref, webhook_url)
    return {"success": True, "jobId": job_id, "status": "PROCESSING"}

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    temp_filename = ""
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_filename = tmp.name

        from deepface import DeepFace
        results = DeepFace.analyze(
            img_path=temp_filename, 
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False
        )
        result = results[0] if isinstance(results, list) else results

        return {
            "success": True,
            "data": {
                "age": int(result.get("age")),
                "gender": str(result.get("dominant_gender")),
                "emotion": str(result.get("dominant_emotion")),
                "emotion_score": float(result["emotion"][result["dominant_emotion"]]) 
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    tmp1, tmp2 = "", ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1:
            f1.write(await file1.read())
            tmp1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
            f2.write(await file2.read())
            tmp2 = f2.name

        from deepface import DeepFace
        result = DeepFace.verify(img1_path=tmp1, img2_path=tmp2, model_name="Facenet512", enforce_detection=False)
        return {"success": True, "data": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if tmp1 and os.path.exists(tmp1): os.remove(tmp1)
        if tmp2 and os.path.exists(tmp2): os.remove(tmp2)

@app.post("/find-face")
async def find_face_in_crowd(target: UploadFile = File(...), crowd: UploadFile = File(...)):
    tmp_target, tmp_crowd = "", ""
    try:
        # 1. Save temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1:
            f1.write(await target.read())
            tmp_target = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
            f2.write(await crowd.read())
            tmp_crowd = f2.name

        # 2. Load Crowd Image for processing
        img_crowd = cv2.imread(tmp_crowd)
        if img_crowd is None:
            raise ValueError("Could not decode crowd image")

        from deepface import DeepFace
        
        # 3. Detect all faces in the crowd
        # We use a robust detector but fallback to opencv if needed
        detector_backend = 'opencv' # Safest for local Windows
        try:
            face_objs = DeepFace.extract_faces(
                img_path=tmp_crowd, 
                detector_backend=detector_backend, 
                enforce_detection=False
            )
        except Exception as e:
            print(f"Extraction error: {e}")
            face_objs = []

        match_count = 0
        
        # 4. Compare each face found in the crowd with the target
        for face_obj in face_objs:
            facial_area = face_obj["facial_area"]
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
            
            # Crop the face from the crowd image
            face_crop = img_crowd[y:y+h, x:x+w]
            
            # DeepFace.verify expects paths or images. We'll use the target path and the crop.
            is_match = False
            try:
                # We use a loose threshold for 'finding' someone in a crowd
                result = DeepFace.verify(
                    img1_path=tmp_target, 
                    img2_path=face_crop, 
                    model_name="Facenet512",
                    detector_backend='skip', # Already detected
                    enforce_detection=False
                )
                is_match = result["verified"]
            except: pass

            # 5. Draw visualization
            if is_match:
                match_count += 1
                color = (0, 255, 0) # Green for match
                thickness = 4
                label = "MATCH"
            else:
                color = (130, 130, 130) # Neutral grey for others
                thickness = 1
                label = ""

            cv2.rectangle(img_crowd, (x, y), (x + w, y + h), color, thickness)
            if label:
                cv2.putText(img_crowd, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 6. Encode result to Base64
        _, buffer = cv2.imencode('.jpg', img_crowd)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        result_url = f"data:image/jpeg;base64,{img_base64}"

        return {
            "success": True,
            "matches": match_count,
            "processed_image": result_url
        }

    except Exception as e:
        print(f"❌ Find Face Error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if tmp_target and os.path.exists(tmp_target): os.remove(tmp_target)
        if tmp_crowd and os.path.exists(tmp_crowd): os.remove(tmp_crowd)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
