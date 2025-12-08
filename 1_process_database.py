import os
import pickle
from deepface import DeepFace

# --- CONFIGURATION ---
db_folder = "known_faces"
output_file = "face_signatures.pkl"
model_name = "Facenet512" # A balance of speed and accuracy

# --- THE LOGIC ---
print("🚀 Starting to process faces...")

database = {} # We will store names and their math numbers here

# 1. Loop through every image in the folder
for filename in os.listdir(db_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        
        # Get the name from the file (remove .jpg)
        person_name = filename.split(".")[0]
        image_path = os.path.join(db_folder, filename)
        
        print(f"   Processing: {person_name}...")
        
        try:
            # 2. The Magic: Convert image to numbers (Embedding)
            embeddings = DeepFace.represent(
                img_path=image_path, 
                model_name=model_name, 
                enforce_detection=False
            )
            
            # Store the first face found in the image
            database[person_name] = embeddings[0]["embedding"]
            
        except Exception as e:
            print(f"   ⚠️ Error processing {filename}: {e}")

# 3. Save the math to a file (Pickling)
with open(output_file, "wb") as f:
    pickle.dump(database, f)

print(f"✅ Done! Database saved to '{output_file}'. You can now run the app.")