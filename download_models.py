# download_models.py
import os
# Set the home directory for DeepFace to the container's cache folder
os.environ["DEEPFACE_HOME"] = "/app/.deepface"

from deepface import DeepFace

print("⏳ Downloading FaceNet512...")
DeepFace.build_model("Facenet512")

print("⏳ Downloading Emotion Analysis model...")
DeepFace.build_model("Emotion")

# Uncomment these if you enable full analysis later
# print("⏳ Downloading Age model...")
# DeepFace.build_model("Age")
# print("⏳ Downloading Gender model...")
# DeepFace.build_model("Gender")

print("✅ All models downloaded and cached!")