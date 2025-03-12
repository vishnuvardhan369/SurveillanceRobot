# Working depth 
import streamlit as st
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("Webcam Object Position Detection")
st.write("Click 'Start Webcam' to detect objects and their positions.")

start_webcam = st.button("Start Webcam")

if start_webcam:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    result_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            depth_map = model(img).squeeze().numpy()

        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 3

        depth_h, depth_w = depth_map.shape

        fx, fy = w * 1.2, h * 1.2 
        cx, cy = w // 2, h // 2   

        detected_objects = []

        for (x, y, w, h) in faces:
            face_center = (x + w // 2, y + h // 2)

            depth_x = int(face_center[0] * depth_w / w)
            depth_y = int(face_center[1] * depth_h / h)

            depth_x = min(max(depth_x, 0), depth_w - 1)
            depth_y = min(max(depth_y, 0), depth_h - 1)

            depth = depth_map[depth_y, depth_x]

            Z = depth  
            X = (face_center[0] - cx) * Z / fx
            Y = (face_center[1] - cy) * Z / fy

            detected_objects.append(f"Person at (X: {X:.2f}m, Y: {Y:.2f}m, Z: {Z:.2f}m)")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

        if detected_objects:
            result_placeholder.write("\n".join(detected_objects))

        if not start_webcam:
            break

    cap.release()
