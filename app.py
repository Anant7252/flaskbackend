from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)
CORS(app)

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

known_face_encodings = []
known_face_names = []

known_image = face_recognition.load_image_file("anant.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_face_encodings.append(known_face_encoding)
known_face_names.append("Anant Tripathi")

def process_frame(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                 (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the coordinates are within the image bounds
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Extract the face region
            face_offset = frame[startY:endY, startX:endX]

            # Check if face_offset is not empty before resizing
            if face_offset.size > 0:
                rgb_face = cv2.cvtColor(face_offset, cv2.COLOR_BGR2RGB)

                # Get the face encodings
                face_encodings = face_recognition.face_encodings(rgb_face)
                
                # Match the detected face encodings with known face encodings
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    
                    face_names.append(name)

                # Display the name on the frame
                for name in face_names:
                    a=name
                    cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw the bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return frame,name
@app.route('/')
def myapp():
    cap = cv2.VideoCapture(0)
    name = "No face detected"
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame,detected_name = process_frame(frame)
            cv2.imshow("frame",frame)
            if detected_name:
                name = detected_name
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return jsonify({"name": name})

if __name__ == "__main__":
    app.run()
