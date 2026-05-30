import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("model/deepfake_model.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_image(path):

    img = cv2.imread(path)

    if img is None:
        return "Error", 0

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30,30)
    )

    # If face detected → crop
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
    else:
        # fallback → full image
        face = img

    # Convert BGR → RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize
    face = cv2.resize(face, (224,224))

    # Normalize
    face = face / 255.0

    # Add batch dimension
    face = np.expand_dims(face, axis=0)

    # Prediction
    pred = model.predict(face, verbose=0)[0][0]

    print("Prediction score:", pred)

    # Convert to label + confidence
    if pred > 0.7:
        return "Real", float(pred * 100)
    else:
        return "Fake", float((1 - pred) * 100)