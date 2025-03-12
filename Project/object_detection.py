from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")  

def detect_objects(frame):
    results = model(frame)
    bags, weapons, people = [], [], []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (255, 255, 255)  
            if label in ["knife", "gun", "pistol"]:
                weapons.append((x1, y1, x2, y2))
                color = (0, 0, 255)
            elif label in ["backpack", "handbag", "suitcase"]:
                bags.append((x1, y1, x2, y2))
                color = (0, 255, 255)  
            elif label == "person":
                people.append((x1, y1, x2, y2))
                color = (0, 255, 0) 

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, bags, people
