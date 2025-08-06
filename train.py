import cv2
import numpy as np
import os
from PIL import Image
import pickle

# Folder where face folders are stored
dataset_path = "TrainingImage"

# Initialize recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = []
labels = []
label_dict = {}  # Maps name to ID
current_id = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    if person_name not in label_dict:
        label_dict[person_name] = current_id
        current_id += 1
    person_id = label_dict[person_name]

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = Image.open(image_path).convert("L")  # Grayscale
        img_np = np.array(img, "uint8")

        faces_rect = detector.detectMultiScale(img_np)
        for (x, y, w, h) in faces_rect:
            face = img_np[y:y+h, x:x+w]
            faces.append(face)
            labels.append(person_id)

# Train model
recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

# Save label name map
with open("labels.pickle", "wb") as f:
    pickle.dump(label_dict, f)

print("✅ Training complete! Labels saved.")
