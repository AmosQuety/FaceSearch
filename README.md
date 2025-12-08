# 👁️ Biometric Authentication System (FaceSearch Project)

A privacy-focused, offline-capable Facial Recognition system built to replace traditional passwords with AI biometrics.

![Demo Screenshot](path/to/your/screenshot.png) 
*(Put that screenshot of you smiling vs. the security log here!)*

## 🚀 Key Features

*   **1:N Face Search:** Instantly identifies users from a database of known identities using Vector Search (Cosine Similarity).
*   **Anti-Spoofing (Liveness):** Prevents photo attacks by analyzing facial micro-expressions (e.g., "Smile to Unlock") in real-time.
*   **Security Logs:** Automatically captures and timestamps photos of unauthorized access attempts or spoofing attacks.
*   **Microservice Architecture:** Decoupled Python "Brain" (FastAPI) that communicates with any Frontend (React/React Native) via REST.
*   **Privacy First:** All biometric embeddings are processed locally; no data is sent to third-party cloud APIs.

## 🛠️ Tech Stack

*   **Core AI:** Python, DeepFace, FaceNet512 (State-of-the-art accuracy)
*   **API:** FastAPI, Uvicorn
*   **Vector Search:** SciPy (Cosine Distance), NumPy
*   **UI (Prototype):** Streamlit
*   **Computer Vision:** OpenCV

## ⚡ Quick Start

1.  **Clone the repo**
    ```bash
    git clone https://github.com/yourusername/facesearch-project.git
    cd facesearch-project
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the API Server**
    ```bash
    uvicorn api:app --reload
    ```

4.  **Run the Dashboard (Optional)**
    ```bash
    streamlit run main.py
    ```

## 📸 How it Works

1.  **Enrollment:** Images are converted into 512-dimensional vector embeddings and stored locally.
2.  **Inference:** Incoming video frames are processed to extract facial landmarks.
3.  **Liveness:** The system runs an emotion classification model to ensure the user is present and reactive.
4.  **Matching:** If Liveness passes, the embedding is compared against the database using Cosine Similarity (Threshold < 0.4).

## 🛡️ License
MIT