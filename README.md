!pip install gradio tensorflow -q

import gradio as gr
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image

# Load model
model = MobileNetV2(weights="imagenet")

def classify_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    results = decode_predictions(predictions, top=3)[0]

    output = ""
    for _, label, score in results:
        output += f"{label}: {score*100:.2f}%\n"

    return output

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AI Image Classifier",
    description="Upload an image and AI will identify it."
)

demo.launch(share=True)
