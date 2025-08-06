import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.title("📸 Face Attendance Monitor")

# Parameters for detection logic
ABSENCE_FRAME_THRESHOLD = 20  # Number of frames without face before registering an exit
FPS = 15  # Approx webcam FPS (adjust if known for better timing)

# Session state
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "attendance_log" not in st.session_state:
    # Structure: {name: {"present": False, "last_seen": None, "entry_time": None, "records": []}}
    st.session_state.attendance_log = {}
if "attendance_records" not in st.session_state:
    # List of dicts with Name, Entry Time, Exit Time, Duration
    st.session_state.attendance_records = []

# Face recognizer and Haar cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces_dir = r"D:\Face_recognition_based_attendance_system-master\TrainingImage"
names = {}
if os.path.exists(faces_dir):
    for i, person_name in enumerate(os.listdir(faces_dir)):
        names[i] = person_name
else:
    st.error(f"⚠️ Folder not found: {faces_dir}")

def update_attendance(detected_names):
    now = datetime.now()
    # Register detections
    for name in detected_names:
        person = st.session_state.attendance_log.setdefault(name, {
            "present": False,
            "last_seen": None,
            "entry_time": None,
            "absence_count": 0
        })
        if not person["present"]:
            # New entry
            person["entry_time"] = now
            person["present"] = True
            person["absence_count"] = 0
        # Update last seen
        person["last_seen"] = now
        person["absence_count"] = 0

    # Update absence counters & process exits
    for name, person in st.session_state.attendance_log.items():
        if name not in detected_names and person["present"]:
            person["absence_count"] += 1
            # Use ~ABSENCE_FRAME_THRESHOLD*frame_interval as the "grace period"
            if person["absence_count"] >= ABSENCE_FRAME_THRESHOLD:
                # Confirmed exit
                exit_time = now
                duration = exit_time - person["entry_time"]
                st.session_state.attendance_records.append({
                    "Name": name,
                    "Entry Time": person["entry_time"].strftime("%H:%M:%S"),
                    "Exit Time": exit_time.strftime("%H:%M:%S"),
                    "Duration": str(duration)
                })
                # Reset
                person["present"] = False
                person["entry_time"] = None
                person["absence_count"] = 0
                person["last_seen"] = None

# Streamlit controls
start = st.button("▶️ Start Monitoring")
stop = st.button("⏹️ Stop Monitoring")

frame_placeholder = st.empty()
csv_placeholder = st.empty()

if start:
    st.session_state.monitoring = True
    # Reset for new session
    st.session_state.attendance_log = {}
    st.session_state.attendance_records = []
    st.info("📷 Monitoring...")

    cap = cv2.VideoCapture(0)
    while st.session_state.monitoring:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        detected_names = []
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (130, 100))
            label, confidence = recognizer.predict(roi_resized)
            if confidence < 80:
                name = names.get(label, "Unknown")
                detected_names.append(name)
                cv2.putText(frame, name, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        update_attendance(detected_names)
        frame_placeholder.image(frame, channels="BGR")
        if len(st.session_state.attendance_records) > 0:
            csv_placeholder.subheader("📄 Attendance Log")
            df_log = pd.DataFrame(st.session_state.attendance_records)
            csv_placeholder.dataframe(df_log)
        # You can add a small sleep if CPU usage is too high

    cap.release()
    cv2.destroyAllWindows()

if stop:
    st.session_state.monitoring = False
    # On stop, log remaining present entries as exited now
    now = datetime.now()
    for name, person in st.session_state.attendance_log.items():
        if person["present"]:
            duration = now - person["entry_time"]
            st.session_state.attendance_records.append({
                "Name": name,
                "Entry Time": person["entry_time"].strftime("%H:%M:%S"),
                "Exit Time": now.strftime("%H:%M:%S"),
                "Duration": str(duration)
            })
            person["present"] = False
            person["entry_time"] = None
    df_log = pd.DataFrame(st.session_state.attendance_records)
    df_log.to_csv("attendance_log.csv", index=False)
    csv_placeholder.subheader("📄 Attendance Log")
    csv_placeholder.dataframe(df_log)
    st.success("✅ Attendance logged successfully.")

