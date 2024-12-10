import tensorflow as tf
import numpy as np
import glob
from PIL import Image

# Path to the folder with your JPEG images
IMAGE_DIR = "path_to_your_images/"
IMAGE_SIZE = (224, 224)  # Change to match your model's input size

def representative_data_gen():
    image_paths = glob.glob(f"{IMAGE_DIR}/*.jpg")
    for image_path in image_paths:
        # Load and preprocess the image
        img = Image.open(image_path).resize(IMAGE_SIZE)
        img = np.array(img) / 255.0  # Normalize if required by your model
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        yield [img.astype(np.float32)]

# Load the TensorFlow model (e.g., SavedModel format)
converter = tf.lite.TFLiteConverter.from_saved_model("path_to_your_saved_model")

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set representative dataset
converter.representative_dataset = representative_data_gen

# Set the target specification for full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Ensure input and output tensors are quantized to int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_model = converter.convert()

# Save the quantized model
with open("model_quantized_int8.tflite", "wb") as f:
    f.write(tflite_model)