import cv2
import time
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Warmup
for _ in range(20):
    ret, frame = cap.read()
    if ret:
        model(frame, verbose=False)

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Start inference timer
    start = time.time()

    results = model(frame, verbose=False)

    # End inference timer
    end = time.time()

    # Inference FPS
    fps = 1 / (end - start)

    # Latency in ms
    latency = (end - start) * 1000

    # Draw detections
    annotated_frame = results[0].plot()

    # Display FPS and latency
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Latency: {latency:.2f} ms",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLOv8 Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()