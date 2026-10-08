import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("botal.mkv")

if not cap.isOpened():
    raise FileNotFoundError("Could not open video file: botal.mkv")

unique_ids = set()

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        results = model.track(
            frame,
            classes=[39],
            persist=True,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            unique_ids.update(boxes.id.int().tolist())

        annotated_frame = result.plot()
        cv2.putText(
            annotated_frame,
            f"Bottles counted: {len(unique_ids)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Bottle Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
