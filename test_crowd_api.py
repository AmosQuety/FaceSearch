import requests
import base64
import os

# API URL
URL = "http://127.0.0.1:8000/find-face"

# Files to use for testing (assuming they exist from previous runs)
TARGET_FILE = "temp_upload.jpg"
CROWD_FILE = "temp_test.jpg"

def test_find_face():
    if not os.path.exists(TARGET_FILE) or not os.path.exists(CROWD_FILE):
        print(f"❌ Test files {TARGET_FILE} or {CROWD_FILE} not found. Please ensure they exist.")
        return

    print(f"🚀 Sending request to {URL}...")
    
    with open(TARGET_FILE, "rb") as t, open(CROWD_FILE, "rb") as c:
        files = {
            "target": (TARGET_FILE, t, "image/jpeg"),
            "crowd": (CROWD_FILE, c, "image/jpeg")
        }
        
        try:
            response = requests.post(URL, files=files)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! Found {data.get('matches')} matches.")
                if "processed_image" in data:
                    print("🖼️ Processed image received (base64).")
                else:
                    print("⚠️ No processed image in response.")
            else:
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_find_face()
