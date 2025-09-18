import cv2
import csv
from datetime import datetime

# Names should match IDs used in main.py
names = ["Unknown", "kavin","kailash","kvk","murali","pavithran"]

# Load trained classifier
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Attendance function
def mark_attendance(name):
    with open("attendance.csv", "r+", newline="") as f:
        data = f.readlines()
        name_list = [line.split(",")[0] for line in data]

        # Only mark once per session
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d,%H:%M:%S")
            f.writelines(f"\n{name},{dt_string}")

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, predict = clf.predict(gray_image[y:y+h, x:x+w])
        confidence = int(100 * (1 - predict / 300))

        if confidence > 77:
            if id < len(names):
                person_name = names[id]
                cv2.putText(img, person_name, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                mark_attendance(person_name)  # ✅ log attendance
            else:
                cv2.putText(img, "Unknown", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Unknown", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        coords = [x, y, w, h]
    return coords

def recognize(img, clf, faceCascade):
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 0, 0), "Face")
    return img

def face_rec():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = recognize(frame, clf, face_cascade)
        cv2.imshow("Face Recognition - Attendance", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run attendance recognition
face_rec()
