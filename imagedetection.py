import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera access.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Webcam Detection", annotated_frame)
    frame_count += 1
    output_path = rf"D:\mycv\webcam_output_{frame_count}.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved frame: {output_path}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam detection stopped.")
