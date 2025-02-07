import streamlit as st
import cv2
from ultralytics import YOLO
import time

st.title("Real-Time Object Detection and Anomaly Detection with YOLOv8")
st.write("Enter the IP addresses of your phones and start processing.")

model = YOLO("yolov8s.pt")

confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

anomaly_rules = {
    "knife": "Weapon detected!",
    "gun": "Weapon detected!",
    "scissors": "Suspicious object detected!",
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.write("Enter the IP addresses and ports of your phones (e.g., 192.168.x.x:8080):")
ip_addresses = st.text_area("Enter IP addresses (one per line):", "192.168.x.x:8080\n192.168.x.x:8081").splitlines()

start_button = st.button("Start Processing")

if start_button and ip_addresses:
    video_placeholders = [st.empty() for _ in ip_addresses]
    alert_placeholders = [st.empty() for _ in ip_addresses]
    fps_placeholders = [st.empty() for _ in ip_addresses]

    caps = [cv2.VideoCapture(f"http://{ip}/video") for ip in ip_addresses]

    for i, cap in enumerate(caps):
        if not cap.isOpened():
            st.error(f"Error: Could not open video stream for {ip_addresses[i]}. Check the IP address and ensure the phone is connected.")
            st.stop()

    start_time = time.time()
    frame_counts = [0] * len(ip_addresses)

    staring_start_times = [None] * len(ip_addresses)
    staring_threshold = 5

    while True:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                st.error(f"Error: Could not read frame from {ip_addresses[i]}.")
                continue

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
                if staring_start_times[i] is None:
                    staring_start_times[i] = time.time()
                else:
                    staring_duration = time.time() - staring_start_times[i]
                    if staring_duration > staring_threshold:
                        anomaly_detected = True
                        anomaly_message = "Suspicious activity detected! (Staring at camera for too long)"
                        cv2.putText(frame, "Suspicious Activity!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                staring_start_times[i] = None

            annotated_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            video_placeholders[i].image(annotated_frame_rgb, channels="RGB", use_column_width=True)

            if anomaly_detected:
                alert_placeholders[i].error(f"🚨 {anomaly_message} 🚨")
            else:
                alert_placeholders[i].empty()

            frame_counts[i] += 1
            fps = frame_counts[i] / (time.time() - start_time)
            fps_placeholders[i].write(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    st.write("Processing stopped.")
