import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture("walking.mkv")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    results = model.track(
        source=frame,
        classes=[0],
        persist=True,
        conf=0.15,
        imgsz=960,
        verbose=False,
    )
    annotated_frame = frame.copy()

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id
        track_ids = ids.cpu().numpy() if ids is not None else [None] * len(boxes)
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            if masks is not None and i < len(masks):
                mask_resized = cv2.resize(
                    masks[i].astype(np.uint8) * 255,
                    (frame.shape[1], frame.shape[0]),
                )
                contours, _ = cv2.findContours(
                    mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(annotated_frame, contours, -1, (0, 0, 255), 2)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"ID: {int(track_ids[i])}" if track_ids[i] is not None else "Person"
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    cv2.imshow("Tracking with Segmentation", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



