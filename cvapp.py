import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load model
model = YOLO("best.pt")

st.title("YOLOv8 Object Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert image
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", width=300)

    # Run detection
    results = model(img)

    # Draw results
    annotated_frame = results[0].plot()

    # Show output
    st.image(annotated_frame, caption="Detected Image", width=300)

    # Show details
    st.write("Detection Results:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"Class: {model.names[cls_id]}, Confidence: {conf:.2f}")