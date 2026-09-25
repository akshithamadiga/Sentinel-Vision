import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12

# Data generator with augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Training data
train_data = train_gen.flow_from_directory(
    "dataset_faces",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

# Validation data
val_data = train_gen.flow_from_directory(
    "dataset_faces",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Print class mapping
print("Class indices:", train_data.class_indices)

# Load XceptionNet with ImageNet pretrained weights
base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze XceptionNet layers
for layer in base_model.layers:
    layer.trainable = False

# Custom classification head
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

# Binary classification output
output = Dense(
    1,
    activation="sigmoid"
)(x)

# Create final model
model = Model(
    inputs=base_model.input,
    outputs=output
)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save trained model
model.save("model/deepfake_model.h5")

print("✅ XceptionNet model saved successfully!")
