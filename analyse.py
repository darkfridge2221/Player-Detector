import cv2
import numpy as np
from tkinter import messagebox

def analyze_video(video_path):
    #Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    #Get output layer names correctly
    out_layer_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in out_layer_indices]
    #Load COCO class labels
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    #Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open video file.")
        return
    #Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #Delay between frames
    delay = int(1000 / fps)  
    prev_ball_position = None
    #movement_threshold = 20 pixel movement to detect fast-moving ball
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        #Initialize lists for detections
        class_ids = []
        confidences = []
        boxes = []
        center_points = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    center_points.append((center_x, center_y))
        #Non-max suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        #Stance detection for spiking and blocking
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                #Estimate arm position using bounding box proportions
                arm_ratio = h / w  #Height-to-width ratio of the bounding box
                if arm_ratio > 2:  #High vertical box indicates blocking
                    stance = "Blocking"
                elif 1.5 < arm_ratio <= 2:  #Slightly more vertical than wide, indicating spiking
                    stance = "Spiking"
                else:
                    stance = "Person"
                #Update label based on detected stance
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, stance, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #Volleyball Detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
        yellow_mask_blurred = cv2.GaussianBlur(yellow_mask, (9, 9), 2)
        circles = cv2.HoughCircles(yellow_mask_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=50, param2=30, minRadius=10, maxRadius=50)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x, y, r = i[0], i[1], i[2]
                cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
                cv2.putText(frame, "Ball", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if prev_ball_position is not None:
                    ball_movement = np.linalg.norm(np.array([x, y]) - np.array(prev_ball_position))
                    if ball_movement > movement_threshold:
                        cv2.putText(frame, "Fast", (x - r, y + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                prev_ball_position = (x, y)
        #Display the frame with annotations
        cv2.imshow("Video Analysis", frame)
        key = cv2.waitKey(delay)
        if key & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
