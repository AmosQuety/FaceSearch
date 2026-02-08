import os
import torch
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from deepface import DeepFace
from supabase import create_client, Client
from dotenv import load_dotenv
import cv2
import base64
from scipy.spatial.distance import cosine
import tempfile
import shutil
import time

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

# Initialize Supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
if not url or not key:
    print("⚠️ WARNING: Missing SUPABASE_URL or SUPABASE_KEY")
else:
    supabase: Client = create_client(url, key)

app = FastAPI(title="Cloud Biometric & Voice AI Engine")

# Lazy load TTS to avoid slow startup for other endpoints
tts = None

def get_tts():
    global tts
    if tts is None:
        print("🎙️ Loading XTTS v2...")
        try:
            from TTS.api import TTS
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
        except Exception as e:
            print(f"❌ Failed to load TTS: {e}")
            raise e
    return tts

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    print(f"⏱️ {request.method} {request.url.path} took {duration:.2f}s")
    return response

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

@app.post("/audio/register")
async def register_voice(
    user_id: str = Form(...),
    reference_audio: UploadFile = File(...)
):
    temp_ref = ""
    try:
        # Save reference audio to temp
        suffix = os.path.splitext(reference_audio.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(reference_audio.file, tmp)
            temp_ref = tmp.name

        # Extract Speaker Embedding (Latents)
        print(f"🎙️ Extracting speaker embeddings for user {user_id}...")
        engine = get_tts()
        
        # XTTS v2 specific latent extraction
        gpt_cond_latent, speaker_embedding = engine.model.get_conditioning_latents(audio_path=temp_ref)
        
        # Convert to list for JSON storage
        # gpt_cond_latent: [1, 1, 1024], speaker_embedding: [1, 16, 64]
        data = {
            "user_id": user_id,
            "gpt_cond_latent": gpt_cond_latent.cpu().numpy().tolist(),
            "speaker_embedding": speaker_embedding.cpu().numpy().tolist(),
            "created_at": "now()"
        }

        # Store in Supabase
        supabase.table("voice_embeddings").upsert(data).execute()

        return {"success": True, "message": f"Voice registered for user {user_id}"}

    except Exception as e:
        print(f"❌ Error in voice register: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if temp_ref and os.path.exists(temp_ref): os.remove(temp_ref)

@app.post("/audio/clone")
async def clone_voice(
    text: str = Form(...),
    user_id: str = Form(None),
    reference_audio: UploadFile = File(None),
    background_tasks: BackgroundTasks = None
):
    temp_ref = ""
    temp_out = ""
    try:
        engine = get_tts()
        gpt_cond_latent = None
        speaker_embedding = None

        if reference_audio:
            # use uploaded audio
            suffix = os.path.splitext(reference_audio.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(reference_audio.file, tmp)
                temp_ref = tmp.name
        elif user_id:
            # lookup in database
            print(f"🔍 Looking up voice for user {user_id}...")
            response = supabase.table("voice_embeddings").select("*").eq("user_id", user_id).execute()
            if response.data and len(response.data) > 0:
                record = response.data[0]
                gpt_cond_latent = torch.tensor(record["gpt_cond_latent"]).to(DEVICE)
                speaker_embedding = torch.tensor(record["speaker_embedding"]).to(DEVICE)
                print("✅ Found stored voice embeddings.")
            else:
                return JSONResponse(status_code=404, content={"success": False, "error": "No voice registered for this user"})
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "Either reference_audio or user_id must be provided"})

        # Prepare Output Path
        out_fd, temp_out = tempfile.mkstemp(suffix=".wav")
        os.close(out_fd)

        # Generate Voice
        if gpt_cond_latent is not None:
            # Use stored latents
            engine.model.inference(
                text=text,
                language="en",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                file_path=temp_out
            )
        else:
            # Use reference audio
            engine.tts_to_file(
                text=text,
                speaker_wav=temp_ref,
                language="en",
                file_path=temp_out
            )

        # Cleanup reference
        if temp_ref and os.path.exists(temp_ref): os.remove(temp_ref)

        # Return file response and cleanup output in background
        if background_tasks:
            background_tasks.add_task(os.remove, temp_out)
        
        return FileResponse(temp_out, media_type="audio/wav", filename="cloned_voice.wav")

    except Exception as e:
        print(f"❌ Error in voice clone: {e}")
        if temp_ref and os.path.exists(temp_ref): os.remove(temp_ref)
        if temp_out and os.path.exists(temp_out): os.remove(temp_out)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    temp_filename = ""
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_filename = tmp.name

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
