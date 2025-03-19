# prompt: yes i need to display the bounding boxs on the image add this also input and rewrite the code

from ultralytics import YOLO
import streamlit as st
from PIL import Image
import io
from PIL import ImageDraw

# Load your YOLO model
model = YOLO(r'C:\Users\navee\Desktop\ML\.ipynb_checkpoints\best.pt')  # Replace with the actual path

st.title("Image Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Perform inference with the YOLO model
    results = model(image)

    # Display the results (e.g., bounding boxes, labels, confidence scores)
    if results:
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            for box in boxes:
                confidence = box.conf.item()
                class_id = box.cls.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # You might want to add logic to determine the class label based on class_id
                # using your model's class definitions.
                st.write(f"Detected Object: {class_id}, Confidence: {confidence:.2f}")
                # Draw bounding box on the image
                draw.rectangle((x1, y1, x2, y2), outline="red", width=2)

        st.image(annotated_image, caption="Annotated Image", use_column_width=True)