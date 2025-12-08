import streamlit as st
import os
from deepface import DeepFace
from scipy.spatial.distance import cosine
import utils 

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Westeros Gatekeeper", layout="wide", page_icon="🏰")

st.markdown(
    """
    <h1 style='text-align: center; color: #E50914;'>🏰 Westeros Identity System</h1>
    <p style='text-align: center;'>Biometric Security for the Seven Kingdoms</p>
    """, 
    unsafe_allow_html=True
)

# --- LOAD DATABASE ---
# We load this once at the top so all tabs can use it
face_db = utils.load_database()

# --- THE THREE TABS ---
# This creates the navigation bar
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Face Search (Test)", "🎥 Live Gate (Smile Check)", "🛡️ Admin Panel", "🚨 Security Logs"])
    
# ==========================================
# TAB 1: FACE SEARCH (Testing your GoT Actors)
# ==========================================
with tab1:
    st.header("Search by Photo")
    st.write("Upload a static image to test if the AI recognizes the character.")
    
    if not face_db:
        st.warning("⚠️ Database is empty. Go to the 'Admin Panel' tab to add characters!")
    
    else:
        test_file = st.file_uploader("Upload an image to test...", type=["jpg", "png", "jpeg"], key="test_upload")
        
        if test_file is not None:
            # Show the image
            col1, col2 = st.columns(2)
            with col1:
                st.image(test_file, caption="Suspect", width=300)
            
            with col2:
                st.info("Scanning Database...")
                
                # Save temp file
                with open("temp_test.jpg", "wb") as f:
                    f.write(test_file.getbuffer())

                try:
                    # 1. Get Embedding
                    target_embedding = DeepFace.represent(
                        "temp_test.jpg", 
                        model_name=utils.MODEL_NAME, 
                        enforce_detection=False
                    )[0]["embedding"]

                    # 2. Find Best Match
                    best_match = "Unknown"
                    best_score = 1.0 # High distance = bad match
                    
                    for name, db_embedding in face_db.items():
                        score = cosine(target_embedding, db_embedding)
                        if score < best_score:
                            best_score = score
                            best_match = name
                    
                    # 3. Display Result
                    if best_score < 0.4:
                        st.success(f"✅ MATCH FOUND: **{best_match.replace('_', ' ').upper()}**")
                        st.metric("Confidence Score", f"{int((1-best_score)*100)}%")
                    else:
                        st.error("❌ NO MATCH FOUND")
                        st.write(f"Closest match was {best_match} (Score: {best_score:.2f})")
                        
                except Exception as e:
                    st.error(f"Error processing image: {e}")

# ==========================================
# TAB 2: LIVE GATE (Webcam + Smile)
# ==========================================
with tab2:
    st.header("Live Security Gate")
    st.write("Requires **Liveness Check** (Smile) to unlock.")

    # Camera Input
    captured_image = st.camera_input("Look at camera & SMILE to enter")

    if captured_image:
        st.write("Analyzing...")
        bytes_data = captured_image.getvalue() # Get raw bytes for logging
        
        # Save temp
        with open("temp_live.jpg", "wb") as f:
            f.write(captured_image.getbuffer())

        try:
            # --- PHASE 1: EMOTION CHECK ---
            analysis = DeepFace.analyze("temp_live.jpg", actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list): analysis = analysis[0]
            
            emotion = analysis['dominant_emotion']
            emotion_score = analysis['emotion'][emotion] # How confident is it?
            
            # Show emotion result
            st.info(f"Detected Expression: **{emotion.upper()}** (Confidence: {emotion_score:.2f}%)")
            st.bar_chart(analysis['emotion']) # CHART

            # security rule
            if emotion not in ['happy', 'surprise']:
                st.error(f"⛔ Liveness Failed. AI thinks you look '{emotion}'.")
                st.warning("Try turning on a light in front of your face!")
                
                # LOG THE INTRUDER (Spoofing)
                utils.log_intrusion(bytes_data, f"Liveness_{emotion}") # Log what it saw

            else:
                # --- PHASE 2: RECOGNITION ---
                st.success("✅ Liveness Passed. Checking Identity...")
                
                target_embedding = DeepFace.represent(
                    "temp_live.jpg", 
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

                if best_score < 0.4:
                    st.balloons()
                    st.markdown(f"## 🔓 WELCOME, {best_match.replace('_', ' ').upper()}")
                else:
                    st.error("❌ Face not recognized in database.")
                    # 📸 LOG THE INTRUDER (Unknown Person)
                    utils.log_intrusion(bytes_data, "Unknown_Intruder")

        except Exception as e:
            st.error(f"System Error: {e}")

# ==========================================
# TAB 3: ADMIN PANEL (Add People)
# ==========================================
with tab3:
    st.header("🛡️ Database Management")
    st.write("Add new characters to the system.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_name = st.text_input("Character Name (e.g., Jon Snow)")
        new_photo = st.file_uploader("Upload Photo", type=['jpg', 'png'], key="admin_upload")
        
        if st.button("Add to Database"):
            if new_name and new_photo:
                with st.spinner("Processing & Retraining..."):
                    # Save Image
                    saved_name = utils.save_uploaded_image(new_photo, new_name)
                    # Retrain Pickle
                    count = utils.retrain_database()
                    
                    st.success(f"✅ Successfully added {saved_name}!")
                    st.info(f"Total Database Size: {count} faces")
                    # Reload the DB variable so other tabs see the change immediately
                    face_db = utils.load_database()
            else:
                st.error("Please provide both a Name and a Photo.")
    
    with col2:
        st.subheader("Current Database Stats")
        st.metric("Total Profiles", len(face_db))
        if face_db:
            st.json(list(face_db.keys()))


# ==========================================
# TAB 4: SECURITY LOGS 
# ==========================================
with tab4:
    st.header("🚨 Intrusion History")
    st.write("Recent failed access attempts.")
    
    logs = utils.get_logs()
    
    if not logs:
        st.info("No security incidents reported yet.")
    else:
        # Create a grid of images
        cols = st.columns(3) # 3 images per row
        
        for index, filename in enumerate(logs):
            # Parse filename for info: "2023-12-08_14-30-01_UnknownIntruder.jpg"
            # We want to display: "Unknown Intruder" and "14:30:01"
            parts = filename.split("_")
            time_str = parts[1].replace("-", ":")
            reason = parts[2].split(".")[0]
            
            # Display in the grid (cycle through columns 0, 1, 2)
            with cols[index % 3]:
                st.image(os.path.join(utils.LOGS_FOLDER, filename), use_container_width=True)
                st.caption(f"**{reason}** at {time_str}")