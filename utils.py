import os
import pickle
from deepface import DeepFace
import datetime

DB_FOLDER = "known_faces"
PKL_FILE = "face_signatures.pkl"
MODEL_NAME = "Facenet512"

LOGS_FOLDER = "security_logs"

def retrain_database():
    """Reads all images in known_faces and saves the pickle file."""
    database = {}
    
    # Ensure folder exists
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)

    files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not files:
        return 0

    print("🔄 Retraining model...")
    for filename in files:
        name = filename.split(".")[0]
        img_path = os.path.join(DB_FOLDER, filename)
        
        try:
            embeddings = DeepFace.represent(img_path, model_name=MODEL_NAME, enforce_detection=False)
            database[name] = embeddings[0]["embedding"]
        except:
            pass

    with open(PKL_FILE, "wb") as f:
        pickle.dump(database, f)
    
    return len(database)

def load_database():
    """Loads the pickle file into memory."""
    if os.path.exists(PKL_FILE):
        with open(PKL_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_uploaded_image(uploaded_file, name):
    """Saves a new image to the known_faces folder."""
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
        
    # Sanitize name
    safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).strip().replace(" ", "_")
    file_path = os.path.join(DB_FOLDER, f"{safe_name}.jpg")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return safe_name


def log_intrusion(image_bytes, reason):
    """Saves the image of an intruder with a timestamp."""
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
    
    # Create a unique filename: "2023-12-08_14-30-01_UnknownFace.jpg"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{reason.replace(' ', '')}.jpg"
    file_path = os.path.join(LOGS_FOLDER, filename)
    
    with open(file_path, "wb") as f:
        f.write(image_bytes)
        
    return filename


def get_logs():
    """Returns a list of all security logs."""
    if not os.path.exists(LOGS_FOLDER):
        return []
    
    # Get files, sort them by newest first
    files = [f for f in os.listdir(LOGS_FOLDER) if f.endswith(".jpg")]
    files.sort(reverse=True) # Newest on top
    return files
