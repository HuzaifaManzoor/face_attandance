import os
import numpy as np
import cv2
import mediapipe as mp
from keras_facenet import FaceNet
import time

# Initialize FaceNet model
face_net = FaceNet()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Initialize the face detection model from MediaPipe
face_detection_model = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def extract_face_embeddings(face_image):
    """
    Extract embeddings for a given face image using FaceNet model.
    Args:
        face_image (numpy.ndarray): Face image.
    Returns:
        numpy.ndarray: Embeddings for the face image.
    """
    if face_image is not None and face_image.size > 0:
        resized_face = cv2.resize(face_image, (160, 160))
        if resized_face.size > 0:
            image = np.expand_dims(resized_face, axis=0)
            embeddings = face_net.embeddings(image)
            return embeddings
    return None

def load_images_from_folder(folder_path):
    """
    Load images and labels from a specified folder.
    Args:
        folder_path (str): Path to the folder containing images.
    Returns:
        list: List of images.
        list: List of corresponding labels.
    """
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            images.append(image)
            labels.append(folder_path.split('/')[-1])  # Assuming folder structure is 'folder_name/image.jpg'
    return images, labels

# Initialize lists to store known faces and labels
known_faces = []
known_labels = []
folders_path = "face_attandance\images"

# Load images and labels from folders
for folder_name in os.listdir(folders_path):
    folder_path = os.path.join(folders_path, folder_name)
    if os.path.isdir(folder_path):
        images, labels = load_images_from_folder(folder_path)
        known_faces.extend(images)
        known_labels.extend(labels)

# Extract embeddings for known faces
known_embeddings = []
for face_image in known_faces:
    rgb_frame = face_image[:, :, ::-1]
    results = face_detection_model.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            # Get the bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = face_image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            face_roi = face_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            face_embeddings = extract_face_embeddings(face_roi)
            if face_embeddings is not None:
                known_embeddings.append(face_embeddings)

# Set confidence threshold
confidence_threshold = 0.8

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    results = face_detection_model.process(rgb_frame)
    if results.detections:
            for detection in results.detections:
                # Get the bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                face_roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                if face_roi is not None or face_roi.size > 0:
                    frame_embeddings = extract_face_embeddings(face_roi)
                    if frame_embeddings is not None:  
                        for i, frame_embedding in enumerate(frame_embeddings):
                            distances = []
                            for known_emb in known_embeddings:
                                dist = np.linalg.norm(known_emb - frame_embedding, axis=1)
                                distances.append(np.mean(dist))

                            # Choose the label with the minimum distance as the recognized face
                            min_distance = np.min(distances)
                            recognized_label = known_labels[np.argmin(distances)] if min_distance < confidence_threshold else "Unknown"
                            
                            # Display the recognized label and distance on the frame for each face
                            cv2.putText(frame, recognized_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

    cv2.imshow("Frames ", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
