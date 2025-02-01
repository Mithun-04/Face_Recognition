import cv2
import threading
import os
from deepface import DeepFace

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Global variables
count = 0
face_match = False
reference_dataset = []  # List to store reference images

# Load all reference images from a folder
reference_folder = "reference_dataset"
for filename in os.listdir(reference_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(reference_folder, filename)
        reference_img = cv2.imread(img_path)
        if reference_img is not None:
            reference_dataset.append(reference_img)

# Specify a more accurate model
model_name = "VGG-Face"

def face_recog(frame):
    global face_match

    try:
        match_count = 0  # Count how many reference images match the frame

        # Compare the frame against each reference image
        for ref_img in reference_dataset:
            result = DeepFace.verify(frame, ref_img.copy(), model_name=model_name, distance_metric='cosine', enforce_detection=False)
            if result["verified"]:
                match_count += 1

        # Decide the final result based on a threshold (e.g., at least 50% matches)
        if match_count >= len(reference_dataset) * 0.5:  # Adjust threshold as needed
            face_match = True
        else:
            face_match = False

    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if count % 15 == 0:
            try:
                threading.Thread(target=face_recog, args=(frame.copy(),)).start()
            except ValueError:
                pass

        count += 1

        if face_match:
            cv2.putText(frame, "Match", (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Match", (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()