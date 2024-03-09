from ultralytics import YOLO
import cv2
import math 
from datetime import datetime

# Confidence threshold
conf_threshold = 0.6

# Video file path
video_path = "test.mp4"

# model
model = YOLO("best.pt")

# object classes
classNames = ["- Back", "- Front", "- Left", "- Right"]

# Open the result.txt file in append mode
result_file = open("report.txt", "a")

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the frames per second (fps) and total number of frames
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    success, img = cap.read()
    
    if not success:
        break

    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Check confidence threshold
            if box.conf[0] >= conf_threshold:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in the frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                object_name = classNames[cls]
                print("Class name -->", object_name)

                # Calculate timestamp based on current frame position
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                timestamp_seconds = current_frame / fps
                current_time = datetime.utcfromtimestamp(timestamp_seconds).strftime("%H:%M:%S")

                # Save to result.txt with timestamp, class name, and confidence score
                result_file.write(f"{current_time} - Object: {object_name}, Confidence: {confidence}\n")

    cv2.imshow('Video', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Close the result.txt file
result_file.close()
cap.release()
cv2.destroyAllWindows()
