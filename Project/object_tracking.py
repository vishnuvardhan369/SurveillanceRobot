from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

tracker = DeepSort(max_age=30)

def track_people_and_bags(frame, detected_people, detected_bags):
    track_detections = []
    label_map = {}

    for x1, y1, x2, y2 in detected_people:
        track_detections.append(([x1, y1, x2 - x1, y2 - y1], 1, 0.9))
        label_map[len(track_detections) - 1] = "person"

    for x1, y1, x2, y2 in detected_bags:
        track_detections.append(([x1, y1, x2 - x1, y2 - y1], 1, 0.9))
        label_map[len(track_detections) - 1] = "bag"

    tracks = tracker.update_tracks(track_detections, frame=frame)

    for i, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        label = label_map.get(i, "unknown")

        color = (0, 255, 0) if label == "person" else (0, 255, 255)
        cv2.putText(frame, f"{label} ID: {track_id}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame
