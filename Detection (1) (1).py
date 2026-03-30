import torch
import numpy as np
from ultralytics import YOLO
import cv2
import time
import os
from emotion import init as init_emotion_classifier, detect_emotion, emotions as EMOTIONS

class FaceDetector:
    def __init__(self, model_path='./yolov8n-face.pt', device='cpu', conf_threshold=0.5, iou_threshold=0.4):
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model.to(self.device)

    def detect_faces(self, image):
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': int(cls)
                })
        return detections
    
    def draw_detections(self, image, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det.get('emotion', 'Unknown')

            # Draw bounding box and label only (no confidence values)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return image

    
cam = cv2.VideoCapture(0)
detector = FaceDetector()

# Initialize emotion classifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
init_emotion_classifier(device)



from collections import defaultdict

emotion_counts = defaultdict(int)
start_time = time.time()
#you can change to 5 seconds
interval = 900  # 15 minutes

while True:
    ret, frame = cam.read()
    if not ret:
        break

    detections = detector.detect_faces(frame)
    
    if len(detections) > 0:
        face_crops = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face_crop = frame[y1:y2, x1:x2]
            face_crops.append(face_crop)
        
        emotions_results = detect_emotion(face_crops, conf=False, return_probs=False)
        for i, det in enumerate(detections):
            if i < len(emotions_results):
                det['emotion'] = emotions_results[i][0]
                emotion_counts[det['emotion']] += 1

    # 👇 مهم جدًا: ده جوه اللوب
    current_time = time.time()

    if current_time - start_time >= interval:
        if emotion_counts:
            most_common = max(emotion_counts, key=emotion_counts.get)
            print("\n🔥 Most frequent emotion in last 5 sec:", most_common)

        emotion_counts.clear()
        start_time = current_time

    frame_with_detections = detector.draw_detections(frame, detections)

    cv2.imshow('Face Detection with Emotion', frame_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break























# import torch
# import numpy as np
# from ultralytics import YOLO
# import cv2
# import time
# import os
# from emotion import init as init_emotion_classifier, detect_emotion, emotions as EMOTIONS

# class FaceDetector:
#     def __init__(self, model_path='./yolov8n-face.pt', device='cpu', conf_threshold=0.5, iou_threshold=0.4):
#         self.model = YOLO(model_path)
#         self.device = device
#         self.conf_threshold = conf_threshold
#         self.iou_threshold = iou_threshold
#         self.model.to(self.device)

#     def detect_faces(self, image):
#         results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
#         detections = []
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 conf = box.conf[0].cpu().numpy()
#                 cls = box.cls[0].cpu().numpy()
#                 detections.append({
#                     'bbox': [int(x1), int(y1), int(x2), int(y2)],
#                     'confidence': float(conf),
#                     'class': int(cls)
#                 })
#         return detections
    
#     def draw_detections(self, image, detections):
#         for det in detections:
#             x1, y1, x2, y2 = det['bbox']
#             label = det.get('emotion', 'Unknown')

#             # Draw bounding box and label only (no confidence values)
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#         return image

    
# cam = cv2.VideoCapture(0)
# detector = FaceDetector()

# # Initialize emotion classifier
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# init_emotion_classifier(device)




# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break

#     detections = detector.detect_faces(frame)
    
#     # Classify emotions for detected faces
#     if len(detections) > 0:
#         face_crops = []
#         for det in detections:
#             x1, y1, x2, y2 = det['bbox']
#             # Ensure coordinates are within bounds
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
#             face_crop = frame[y1:y2, x1:x2]
#             face_crops.append(face_crop)
        
#         # Get emotions for all faces (no special-case overrides)
#         emotions_results = detect_emotion(face_crops, conf=False, return_probs=False)
#         for i, det in enumerate(detections):
#             if i < len(emotions_results):
#                 # emotions_results entries are [label_string, idx]
#                 det['emotion'] = emotions_results[i][0]



#                 print(f"Detected emotion for face {i}: {det['emotion']}")
        





        
    
#     frame_with_detections = detector.draw_detections(frame, detections)

#     cv2.imshow('Face Detection with Emotion', frame_with_detections)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break






# cam.release()
# cv2.destroyAllWindows()

