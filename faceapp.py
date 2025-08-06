import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📸 Fast Face Attendance System")

# Initialize session state
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = {}
if "df_log" not in st.session_state:
    st.session_state.df_log = pd.DataFrame(columns=["Name", "Entry Time", "Exit Time", "Duration"])

# Load LBPH face recognizer and Haar cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Correct label mapping
faces_dir = r"D:\Face_recognition_based_attendance_system-master\TrainingImage"
names = {}

if os.path.exists(faces_dir):
    for i, person_name in enumerate(sorted(os.listdir(faces_dir))):
        person_path = os.path.join(faces_dir, person_name)
        if os.path.isdir(person_path):
            names[i] = person_name
else:
    st.error(f"⚠️ Folder not found: {faces_dir}")

# Function to log attendance
def log_attendance(name):
    now = datetime.now()
    if name not in st.session_state.attendance_log:
        st.session_state.attendance_log[name] = {"entry": now, "exit": now}
    else:
        st.session_state.attendance_log[name]["exit"] = now

# Layout
col1, col2 = st.columns([1, 1])
with col1:
    start = st.button("▶️ Start Monitoring")
with col2:
    stop = st.button("⏹️ Stop Monitoring")

frame_placeholder = st.empty()
csv_placeholder = st.empty()

# Start monitoring
if start:
    st.session_state.monitoring = True
    cap = cv2.VideoCapture(0)
    st.info("📷 Monitoring started...")

    while st.session_state.monitoring:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (130, 100))
            label, confidence = recognizer.predict(face)

            if confidence < 80:
                name = names.get(label, "Unknown")
                log_attendance(name)
                cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# Stop monitoring and show log
if stop:
    st.session_state.monitoring = False

    for name, times in st.session_state.attendance_log.items():
        duration = times["exit"] - times["entry"]
        st.session_state.df_log.loc[len(st.session_state.df_log)] = [
            name,
            times["entry"].strftime("%H:%M:%S"),
            times["exit"].strftime("%H:%M:%S"),
            str(duration)
        ]

    # Save to CSV and show log
    csv_placeholder.subheader("📄 Attendance Log")
    st.session_state.df_log.to_csv("attendance_log.csv", index=False)
    csv_placeholder.dataframe(st.session_state.df_log)
    st.success("✅ Attendance logged successfully.")
