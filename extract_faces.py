import cv2
import os

input_dir = "dataset_faces"
output_dir = "dataset_faces"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

os.makedirs(output_dir, exist_ok=True)

for category in ["training_fake", "training_real"]:
    
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    path = os.path.join(input_dir, category)

    for img_name in os.listdir(path):

        img_path = os.path.join(path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30,30)
        )

        if len(faces) > 0:
            x,y,w,h = faces[0]
            face = img[y:y+h, x:x+w]
        else:
            face = img

        save_path = os.path.join(output_dir, category, img_name)

        cv2.imwrite(save_path, face)

print("Face extraction complete.")