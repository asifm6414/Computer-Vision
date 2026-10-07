import cv2 
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
video_path = "walking.mkv"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {video_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to collect frame")
        break
    results = model(frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Video Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()