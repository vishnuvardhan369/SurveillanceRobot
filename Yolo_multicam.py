# Webcam

import streamlit as st
import cv2
from ultralytics import YOLO
import time

st.title("Real-Time Object Detection and Anomaly Detection with YOLOv8")
st.write("Click the button below to start real-time detection.")

model = YOLO("yolov8s.pt")

confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

anomaly_rules = {
    "knife": "Weapon detected!",
    "gun": "Weapon detected!",
    "scissors": "Suspicious object detected!",
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

start_button = st.button("Start Webcam")

if start_button:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        st.stop()

    video_placeholder = st.empty()
    alert_placeholder = st.empty()
    fps_placeholder = st.empty()

    start_time = time.time()
    frame_count = 0

    staring_start_time = None
    staring_threshold = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from webcam.")
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, conf=confidence_threshold)

        anomaly_detected = False
        anomaly_message = ""

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = box.conf.item()

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)
                if class_name in anomaly_rules:
                    color = (0, 0, 255)
                    anomaly_detected = True
                    anomaly_message = anomaly_rules[class_name]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            if staring_start_time is None:
                staring_start_time = time.time()
            else:
                staring_duration = time.time() - staring_start_time
                if staring_duration > staring_threshold:
                    anomaly_detected = True
                    anomaly_message = "Suspicious activity detected! (Staring at webcam for too long)"
                    cv2.putText(frame, "Suspicious Activity!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            staring_start_time = None

        annotated_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

        if anomaly_detected:
            alert_placeholder.error(f"🚨 {anomaly_message} 🚨")
        else:
            alert_placeholder.empty()

        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        fps_placeholder.write(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    st.write("Webcam stopped.")
