import cv2
import threading
from deepface import DeepFace
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

dataset_path = "reference_dataset"
model_name = "VGG-Face"

identified_person = "Unknown"
lock = threading.Lock()

def recognize_face(frame):
    global identified_person
    try:
        result = DeepFace.find(frame, db_path=dataset_path, model_name=model_name, distance_metric='cosine', enforce_detection=False)
        with lock:
            if result and len(result[0]) > 0:
                identified_person = os.path.basename(result[0]['identity'][0]).split(".")[0]
            else:
                identified_person = "Unknown"
        # print(f"Identified: {identified_person}") 
    except Exception as e:
        with lock:
            identified_person = "Unknown"
        # print("Recognition Error:", str(e))

count = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    if count % 15 == 0:
        threading.Thread(target=recognize_face, args=(frame.copy(),), daemon=True).start()

    count += 1

    with lock:
        cv2.putText(frame, identified_person, (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0) if identified_person != "Unknown" else (0, 0, 255), 3)

    cv2.imshow("Face_Recognition", frame)

    # If a person is identified, stop the process
    with lock:
        if identified_person != "Unknown":
            print(f"Stopping process. Identified: {identified_person}")
            break  # Stop the loop and exit

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
