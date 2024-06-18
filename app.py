import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Function to load YOLO v4 model and Coco names
@st.cache_resource
def load_yolo_model():
    # Load YOLO v4 model
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    # Load Coco names
    with open("../../Desktop/Slash AI Task/coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    return net, classes

# Function to perform object detection with YOLO v4
def detect_objects_yolo(image, net, classes):
    image_np = np.array(image)
    (H, W) = image_np.shape[:2]
    blob = cv2.dnn.blobFromImage(image_np, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    
    # Get output layer names
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except Exception as e:
        st.error(f"Error accessing YOLO v4 layers: {e}")
        return image_np, []

    outputs = net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)
    detected_objects = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = (255, 0, 0)
            cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]}"
            cv2.putText(image_np, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detected_objects.append(classes[class_ids[i]])

    return image_np, detected_objects

# Load YOLO v4 model and Coco names
net, classes = load_yolo_model()

st.title("Image Component Detection")
st.write("using YOLOv4")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Image uploaded successfully.")

if st.button("Analyze Image") and uploaded_image is not None:
    
    st.write("Analyzing Image...")
    # Perform object detection with YOLO v4
    image_np, detected_objects = detect_objects_yolo(image, net, classes)

    # Display detected objects
    st.write("Detected objects:")
    for obj in detected_objects:
        st.write(f"- {obj}")

    # Display the image with bounding boxes and labels
    st.image(image_np, caption='Detected Objects', use_column_width=True)  