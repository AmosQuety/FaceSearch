from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import pickle
from deepface import DeepFace
from scipy.spatial.distance import cosine

app = FastAPI(title="Biometric API")

STORAGE_ROOT = "storage"
GLOBAL_DIR = "_global_login" # Special folder for login faces

# --- HELPER: Smart Path Selector ---
def get_storage_path(user_id: str, workspace_id: str):
    # If it's for the Login System, use the special global folder
    if workspace_id == "global":
        path = os.path.join(STORAGE_ROOT, GLOBAL_DIR)
    else:
        # Otherwise, use the User's specific Workspace folder
        path = os.path.join(STORAGE_ROOT, user_id, workspace_id)
    
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# --- HELPER: Load DB ---
def load_db(path):
    pkl_path = os.path.join(path, "embeddings.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return {}

# --- HELPER: Save DB ---
def save_db(path, database):
    pkl_path = os.path.join(path, "embeddings.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(database, f)

@app.get("/")
def home():
    return {"status": "online", "system": "Hybrid (Global + Workspace) Engine"}

@app.post("/register")
async def register_face(
    name: str = Form(...), 
    user_id: str = Form(...),
    workspace_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # 1. Determine where to save
        target_path = get_storage_path(user_id, workspace_id)
        
        # 2. Save Image
        # If global, name MUST be unique (usually the User ID or Email)
        safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c=='_' or c=='.' or c=='@']).strip()
        img_path = os.path.join(target_path, f"{safe_name}.jpg")

        with open(img_path, "wb") as buffer:
            buffer.write(await file.read())

        # 3. Embedding
        embeddings = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False)
        embedding_vector = embeddings[0]["embedding"]

        # 4. Update Index
        db = load_db(target_path)
        db[safe_name] = embedding_vector
        save_db(target_path, db)

        return {"status": "success", "message": "Enrolled successfully", "total_faces": len(db)}

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/verify")
async def verify_face(
    user_id: str = Form(default="unknown"), # Optional for global login
    workspace_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # 1. Save temp
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Liveness
        try:
            analysis = DeepFace.analyze(temp_filename, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
            
            # Allow neutral for easier testing
            if emotion not in ['happy', 'surprise', 'neutral']:
                os.remove(temp_filename)
                return JSONResponse(status_code=401, content={
                    "access": "DENIED", "error": "Liveness Failed", 
                    "message": f"User looked '{emotion}'. Smile required."
                })
        except:
            emotion = "unknown"

        # 3. Recognition
        target_embedding = DeepFace.represent(temp_filename, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
        os.remove(temp_filename)

        # 4. Load Correct DB
        target_path = get_storage_path(user_id, workspace_id)
        db = load_db(target_path)

        # 5. Search
        best_match = "Unknown"
        best_score = 1.0
        
        for name, db_embedding in db.items():
            score = cosine(target_embedding, db_embedding)
            if score < best_score:
                best_score = score
                best_match = name

        # 6. Result
        if best_score < 0.4:
            return {
                "access": "GRANTED",
                "user": best_match, # This will be the UserID/Email if using Global
                "confidence": round((1 - best_score) * 100, 2),
                "emotion_detected": emotion
            }
        else:
            return JSONResponse(status_code=401, content={
                "access": "DENIED", "error": "Identity Unknown", "score": round(best_score, 2)
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})