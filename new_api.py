import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. Load Env Vars
load_dotenv()

# 2. Configuration
# Point to external drive for models (keep your existing setup)
os.environ["DEEPFACE_HOME"] = "F:/Amos/AI_models" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 3. Initialize Supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("❌ ERROR: Missing SUPABASE_URL or SUPABASE_KEY in .env")

supabase: Client = create_client(url, key)

app = FastAPI(title="Cloud Biometric API")

@app.get("/")
def home():
    return {"status": "online", "system": "Cloud-Native Biometric Engine ☁️"}

@app.post("/register")
async def register_face(
    name: str = Form(...), 
    user_id: str = Form(...),
    workspace_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # 1. Read File
        file_content = await file.read()
        
        # 2. Save Temp for Processing
        temp_filename = f"temp_reg_{file.filename}"
        with open(temp_filename, "wb") as f:
            f.write(file_content)

        # 3. Generate Embedding (The Math)
        # Force numpy array to python list for JSON serialization
        embeddings = DeepFace.represent(
            img_path=temp_filename, 
            model_name="Facenet512", 
            enforce_detection=False
        )
        embedding_vector = embeddings[0]["embedding"]

        # 4. Upload Image to Supabase Storage (Bucket: 'biometric_faces')
        # Path: user_id/workspace_id/name.jpg
        storage_path = f"{user_id}/{workspace_id}/{name}.jpg"
        
        # Upload (Upsert=true overwrites if exists)
        supabase.storage.from_("biometric_faces").upload(
            path=storage_path,
            file=file_content,
            file_options={"content-type": file.content_type, "upsert": "true"}
        )

        # 5. Insert Metadata + Vector into Database
        data = {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "name": name,
            "image_path": storage_path,
            "embedding": embedding_vector # Postgres vector extension handles the list
        }
        
        supabase.table("face_embeddings").insert(data).execute()

        # Cleanup
        os.remove(temp_filename)

        return {"status": "success", "message": f"Cloud enrollment complete for {name}"}

    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(temp_filename): os.remove(temp_filename)
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/verify")
async def verify_face(
    user_id: str = Form(...),
    workspace_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # 1. Read & Temp Save
        temp_filename = f"temp_ver_{file.filename}"
        with open(temp_filename, "wb") as f:
            f.write(await file.read())

        # 2. Liveness Check (Optional - keep lightweight)
        try:
            analysis = DeepFace.analyze(temp_filename, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list): analysis = analysis[0]
            emotion = analysis['dominant_emotion']
            
            # Simple liveness rule
            if emotion not in ['happy', 'surprise', 'neutral']:
                os.remove(temp_filename)
                return JSONResponse(status_code=401, content={
                    "access": "DENIED", 
                    "error": "Liveness Failed", 
                    "message": f"User looked '{emotion}'. Smile required."
                })
        except:
            emotion = "unknown"

        # 3. Generate Target Embedding
        target_embedding = DeepFace.represent(
            img_path=temp_filename, 
            model_name="Facenet512", 
            enforce_detection=False
        )[0]["embedding"]

        # 4. Perform Vector Search via RPC (Remote Procedure Call)
        # We call the SQL function 'match_faces' we created in Step 1
        rpc_params = {
            "query_embedding": target_embedding,
            "match_threshold": 0.4, # 0.4 threshold (same as before)
            "filter_workspace_id": workspace_id
        }
        
        response = supabase.rpc("match_faces", rpc_params).execute()
        matches = response.data

        # Cleanup
        os.remove(temp_filename)

        # 5. Handle Result
        if matches and len(matches) > 0:
            best_match = matches[0] # { name: "Jon Snow", similarity: 0.85 }
            return {
                "access": "GRANTED",
                "user": best_match['name'],
                "confidence": round(best_match['similarity'] * 100, 2),
                "emotion_detected": emotion
            }
        else:
            return JSONResponse(status_code=401, content={
                "access": "DENIED", 
                "error": "Identity Unknown", 
                "score": 0.0
            })

    except Exception as e:
        if os.path.exists(temp_filename): os.remove(temp_filename)
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
            actions=[ 'emotion'],
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
                 # Convert numpy types to standard python types
                # "age": int(result.get("age")), 
                # "gender": str(result.get("dominant_gender")),
                "age": 25,          # Placeholder
                "gender": "unknown",# Placeholder
                "emotion": str(result.get("dominant_emotion")),
                "emotion_score": float(result["emotion"][result["dominant_emotion"]]) 
            }
        }

    except Exception as e:
        # Print the ACTUAL error to the terminal so we can see it
        import traceback
        traceback.print_exc()
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
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