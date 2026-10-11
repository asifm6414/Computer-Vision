import cv2
from ultralytics import YOLO
from collections import defaultdict, deque

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("walking.mkv")

id_map = {}
next_id = 1
appear = defaultdict(int)
trail = defaultdict(lambda: deque(maxlen=30))

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    results = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated_frame = frame.copy()

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        if boxes.id is not None:
            for box, oid in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                appear[oid] += 1

                if appear[oid] >= 5 and oid not in id_map:
                    id_map[oid] = next_id
                    next_id += 1

                if oid in id_map:
                    sid = id_map[oid]
                    trail[oid].append((cx, cy))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"ID: {sid}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    for point in trail[oid]:
                        cv2.circle(annotated_frame, point, 2, (0, 255, 0), -1)

    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()