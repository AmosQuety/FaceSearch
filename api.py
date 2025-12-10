from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import pickle
import cv2
import numpy as np
import base64

from scipy.spatial.distance import cosine
os.environ["DEEPFACE_HOME"] = "F:\\Amos\\AI_models"

from deepface import DeepFace

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


@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    try:
        # 1. Save temp file
        temp_filename = f"temp_analyze_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Run DeepFace Analysis
        # actions: age, gender, race, emotion
        results = DeepFace.analyze(
            img_path=temp_filename, 
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False
        )

        # DeepFace returns a list (in case of multiple faces), we take the first
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        # 3. Cleanup
        os.remove(temp_filename)

        # 4. Return Data
        return {
            "success": True,
            "data": {
                "age": result.get("age"),
                "gender": result.get("dominant_gender"),
                "emotion": result.get("dominant_emotion"),
                "emotion_score": result["emotion"][result["dominant_emotion"]]
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # 1. Save temp files
        filename1 = f"compare_1_{file1.filename}"
        filename2 = f"compare_2_{file2.filename}"
        
        with open(filename1, "wb") as f: f.write(await file1.read())
        with open(filename2, "wb") as f: f.write(await file2.read())

        # 2. Run Verification (Uses Facenet512 - Already downloaded!)
        result = DeepFace.verify(
            img1_path=filename1, 
            img2_path=filename2, 
            model_name="Facenet512",
            enforce_detection=False
        )

        # 3. Cleanup
        os.remove(filename1)
        os.remove(filename2)

        # 4. Calculate "Similarity Score" (Convert Distance to %)
        # Facenet512 Threshold is usually 0.4.
        # Distance 0.0 = 100% Match
        # Distance 0.4 = 50% Match (Threshold)
        # Distance 1.0 = 0% Match
        
        distance = result['distance']
        threshold = result['threshold']
        
        # Simple percentage logic (approximate)
        if distance > 1.0: distance = 1.0
        similarity = (1.0 - distance) * 100

        return {
            "success": True,
            "data": {
                "verified": result['verified'],
                "distance": distance,
                "similarity_score": round(similarity, 1),
                "threshold": threshold
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})



@app.post("/find-face")
async def find_face_in_crowd(
    target: UploadFile = File(...), 
    crowd: UploadFile = File(...)
):
    try:
        # 1. Save temp files
        target_path = f"temp_target_{target.filename}"
        crowd_path = f"temp_crowd_{crowd.filename}"
        
        with open(target_path, "wb") as f: f.write(await target.read())
        with open(crowd_path, "wb") as f: f.write(await crowd.read())

        # 2. Get Target Embedding (The person we are looking for)
        target_embedding = DeepFace.represent(
            img_path=target_path, 
            model_name="Facenet512", 
            enforce_detection=True 
        )[0]["embedding"]

        # 3. Scan the Crowd (Get embeddings & coordinates for EVERYONE)
        # This returns a list of objects for every face found
        crowd_faces = DeepFace.represent(
            img_path=crowd_path,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="retinaface" #or "mtcnn"
        )

        # 4. Load Image with OpenCV for drawing
        img = cv2.imread(crowd_path)
        matches_found = 0

        for face in crowd_faces:
            # Get coordinates
            x = face["facial_area"]["x"]
            y = face["facial_area"]["y"]
            w = face["facial_area"]["w"]
            h = face["facial_area"]["h"]
            
            # Compare current face with target
            current_embedding = face["embedding"]
            distance = cosine(target_embedding, current_embedding)
            
            # Threshold for Facenet512 is 0.4
            if distance < 0.5:
                # MATCH! Draw GREEN Box
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4) # Green
                cv2.putText(img, "FOUND", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                matches_found += 1
            else:
                # NO MATCH. Draw RED Box (Optional: or Blur)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red

        # 5. Convert processed image to Base64 to send back to React
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 6. Cleanup
        os.remove(target_path)
        os.remove(crowd_path)

        return {
            "success": True,
            "matches": matches_found,
            "processed_image": f"data:image/jpeg;base64,{img_base64}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})