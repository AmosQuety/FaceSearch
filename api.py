from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from deepface import DeepFace
from scipy.spatial.distance import cosine
import utils # We reuse the existing logic!

# Initialize the App
app = FastAPI(title="Westeros Biometric API", description="The Backend for the face recognition system", version="1.0")

# Load the database into memory when the server starts
face_db = utils.load_database()

@app.get("/")
def home():
    """Health Check Endpoint"""
    return {"status": "online", "system": "Westeros Gatekeeper v1.0"}

@app.post("/verify")
async def verify_user(file: UploadFile = File(...)):
    """
    The Main Gate:
    1. Checks Liveness (Smile)
    2. Checks Identity (Vector Search)
    3. Returns JSON to the Mobile App
    """
    
    # 1. Save the uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        content = await file.read() # Read the bytes
        buffer.write(content)

    try:
        # --- PHASE 1: LIVENESS CHECK ---
        # Note: We enforce the "happy" or "surprise" rule here
        analysis = DeepFace.analyze(temp_filename, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list): analysis = analysis[0]
        
        emotion = analysis['dominant_emotion']
        print(f"DEBUG: Detected Emotion: {emotion}")

        # Security Rule (You can add 'neutral' here if you want to relax it)
        if emotion not in ['happy', 'surprise']:
            # Log the Intruder
            utils.log_intrusion(content, f"Liveness_Failed_({emotion})")
            
            # Return JSON Error
            return JSONResponse(
                status_code=401, 
                content={
                    "access": "DENIED", 
                    "error": "Liveness Check Failed", 
                    "message": f"User looked '{emotion}'. Smile required."
                }
            )

        # --- PHASE 2: FACE RECOGNITION ---
        target_embedding = DeepFace.represent(
            temp_filename, 
            model_name=utils.MODEL_NAME, 
            enforce_detection=False
        )[0]["embedding"]

        best_match = "Unknown"
        best_score = 1.0
        
        for name, db_embedding in face_db.items():
            score = cosine(target_embedding, db_embedding)
            if score < best_score:
                best_score = score
                best_match = name

        # Cleanup: Delete the temp file
        os.remove(temp_filename)

        # --- DECISION ---
        if best_score < 0.4:
            return {
                "access": "GRANTED",
                "user": best_match.replace("_", " "),
                "confidence": round((1 - best_score) * 100, 2),
                "emotion_detected": emotion
            }
        else:
            # Log the Stranger
            utils.log_intrusion(content, "Unknown_Intruder")
            
            return JSONResponse(
                status_code=401,
                content={
                    "access": "DENIED", 
                    "error": "Identity Unknown", 
                    "best_guess": best_match,
                    "score": round(best_score, 2)
                }
            )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/register")
async def register_user(name: str = Form(...), file: UploadFile = File(...)):
    """
    Admin Endpoint to add new users via API
    """
    # Read image
    content = await file.read()
    
    # Save to known_faces using your util logic (manually implementing here for API)
    safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).strip().replace(" ", "_")
    save_path = os.path.join(utils.DB_FOLDER, f"{safe_name}.jpg")
    
    with open(save_path, "wb") as f:
        f.write(content)
    
    # Trigger Retrain
    count = utils.retrain_database()
    
    # Reload the global variable so the API knows the new person immediately
    global face_db
    face_db = utils.load_database()
    
    return {"status": "success", "message": f"Added {safe_name}", "total_faces": count}