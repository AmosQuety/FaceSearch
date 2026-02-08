import requests
import sys

def test_connection():
    url = "http://127.0.0.1:8000/"
    try:
        print(f"📡 Testing connection to {url}...")
        response = requests.get(url, timeout=5)
        print(f"✅ Status Code: {response.status_code}")
        print(f"📄 Response Body: {response.json()}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

def test_find_face_simple():
    url = "http://127.0.0.1:8000/find-face"
    print(f"\n📡 Testing POST to {url} with empty data...")
    try:
        # This should fail with 422 (Unprocessable Entity) because files are missing
        # But it confirms the route exists and server is responding
        response = requests.post(url, timeout=5)
        print(f"✅ Status Code: {response.status_code}")
        if response.status_code == 422:
            print("👍 Correctly got 422 (Missing files).")
        else:
            print(f"📄 Response: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_connection()
    test_find_face_simple()
