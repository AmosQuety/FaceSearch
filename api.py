import os
import subprocess
import traceback
import logging
import sys
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
import builtins
import sys
import gc
import wave
import struct
import threading

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

        # Insert Metadata + Vector
        data = {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "name": name,
            "image_path": storage_path,
            "embedding": embedding_vector
        }
        supabase.table("face_embeddings").insert(data).execute()

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

            import io, numpy as np
            def tensor_to_npy_bytes(t):
                buf = io.BytesIO()
                np.save(buf, t.cpu().numpy())
                return buf.getvalue()

            gpt_bytes = tensor_to_npy_bytes(gpt_cond_latent)
            spk_bytes = tensor_to_npy_bytes(speaker_embedding)

            gpt_path = f"voice-latents/{user_id}/gpt_cond_latent.npy"
            spk_path = f"voice-latents/{user_id}/speaker_embedding.npy"

            supabase.storage.from_("biometric_faces").upload(
                path=gpt_path, file=gpt_bytes,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
            supabase.storage.from_("biometric_faces").upload(
                path=spk_path, file=spk_bytes,
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
                    gpt_cond_latent = torch.tensor(np.load(io.BytesIO(gpt_bytes))).to(DEVICE)
                    speaker_embedding = torch.tensor(np.load(io.BytesIO(spk_bytes))).to(DEVICE)
                    print(f"[{job_id}] ✅ Loaded voice latents")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
