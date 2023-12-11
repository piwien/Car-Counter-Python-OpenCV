from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

carvideo = cv2.VideoCapture("car_adana.mp4")
model = YOLO("Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
              "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
              "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
              "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
              "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
              "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask_v2.jpg")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
lineLimits = [40, 1600, 1020, 1600]
totalCount = []

# Trajectory storage
trajectories = {}

# Define the codec and create a VideoWriter object
output_path = "output_video2.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (int(carvideo.get(3)), int(carvideo.get(4))))

while True:
    success, img = carvideo.read()
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("car_graphic.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, device="mps")

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car"] and conf > 0.6:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                if conf > 0.8:
                    classDetectionName = currentClass

    resultsTracker = tracker.update(detections)
    cv2.line(img, (lineLimits[0], lineLimits[1]), (lineLimits[2], lineLimits[3]), (0, 0, 255), 25)

    for result in resultsTracker:
        x1, y1, x2, y2, track_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw smaller and more frequent dashed trajectory
        if track_id in trajectories:
            for i in range(1, len(trajectories[track_id])):
                if i % 2 == 0:  # Draw dashed line every 2 iterations
                    cv2.line(img, trajectories[track_id][i - 1], trajectories[track_id][i], (0, 255, 0), 2)

            trajectories[track_id].append((x1 + w // 2, y1 + h // 2))

        else:
            trajectories[track_id] = [(x1 + w // 2, y1 + h // 2)]

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{classDetectionName} id:{int(track_id)} conf:{conf}', (max(0, x1), max(85, y1)), scale=1, thickness=1, offset=3)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if lineLimits[0] < cx < lineLimits[2] and lineLimits[1] - 35 < cy < lineLimits[1] + 35:
            if track_id not in totalCount:
                totalCount.append(track_id)
                cv2.line(img, (lineLimits[0], lineLimits[1]), (lineLimits[2], lineLimits[3]), (0, 255, 0), 25)
    if len(totalCount) < 10:
        cv2.putText(img, str(len(totalCount)), (255, 155), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 12)
    else:
        cv2.putText(img, str(len(totalCount)), (235, 140), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 0), 8)
    cv2.imshow("Video", img)
    output_video.write(img)  # Write frame to the output video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoWriter and close all windows
output_video.release()
carvideo.release()
cv2.destroyAllWindows()
