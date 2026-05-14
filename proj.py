import cv2 as cv
from ultralytics import YOLO

# Load YOLO model (downloads automatically first time)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv.VideoCapture(0)

# Check camera
if not cap.isOpened():
    print("Cannot open camera")
    exit() 

while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    # Detect objects
    results = model(frame)

    # Draw boxes and labels automatically
    output = results[0].plot()

    # Show result
    cv.imshow("Object Detection", output)

    # Quit on q
    if cv.waitKey(1) == ord("q"):
        break

# Cleanup
cap.release()
cv.destroyAllWindows()