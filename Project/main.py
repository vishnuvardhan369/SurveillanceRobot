import streamlit as st
import cv2
import mediapipe as mp
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from object_detection import detect_objects
from object_tracking import track_people_and_bags
from pose_estimation import detect_pose

pose_model = mp.solutions.pose.Pose()
object_model = YOLO("yolov8m.pt") 
tracker = DeepSort(max_age=30)   

st.title("Real-Time Surveillance System")
st.write("Live Video Feed with Object Detection, Tracking, and Pose Analysis.")

start_button = st.button("Start Webcam")

if start_button:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        st.stop()

    video_placeholder = st.empty()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from webcam.")
            break

        frame = cv2.resize(frame, (640, 480))  

        frame, detected_bags, detected_people = detect_objects(frame)

        frame = track_people_and_bags(frame, detected_people, detected_bags)

        frame = detect_pose(frame, frame_count)

        annotated_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    st.write("Webcam stopped.")
