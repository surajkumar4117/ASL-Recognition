import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image

# --------------------- Streamlit Title ---------------------
st.markdown(
    "<h3 style='text-align: center; color: black;'>SignifyAI : Real-Time ASL Recognition System</h3>",
    unsafe_allow_html=True
)

# --------------------- Focal Loss Definition ---------------------
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return loss

# --------------------- Load Model ---------------------
model = load_model("asl_syn_128.h5", custom_objects={
    'loss': categorical_focal_loss(alpha=0.25, gamma=2.0),
    'categorical_focal_loss': categorical_focal_loss
})

# --------------------- Class Labels ---------------------
CLASSES = [
    'A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# --------------------- Mediapipe Setup ---------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

# --------------------- CLAHE Enhancement ---------------------
def apply_clahe(img_rgb):
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced

# --------------------- Initialize Text Buffer ---------------------
if 'text_buffer' not in st.session_state:
    st.session_state.text_buffer = ""

# --------------------- Capture Input from Camera ---------------------
frame = st.camera_input("Capture your hand sign", key="camera")

if frame is not None:
    # Convert to numpy array
    image = Image.open(frame)
    image_np = np.array(image)

    # Remove alpha channel if present
    if image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]

    # Convert to correct color format for Mediapipe
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_mediapipe = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    # Display the image being processed
    # st.image(image_mediapipe, caption="Processed Input Frame", channels="RGB")

    # Detect hands
    results = hands.process(image_mediapipe)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = image_mediapipe.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            # Add padding
            padding = 30
            x_min, x_max = max(min(x_coords) - padding, 0), min(max(x_coords) + padding, w)
            y_min, y_max = max(min(y_coords) - padding, 0), min(max(y_coords) + padding, h)

            # Crop the hand region
            cropped_hand = image_mediapipe[y_min:y_max, x_min:x_max]
            cropped_hand = apply_clahe(cropped_hand)

            if cropped_hand.size == 0:
                st.warning("Empty hand region detected.")
                continue

            # Resize and normalize
            test_image = cv2.resize(cropped_hand, (128, 128))
            test_image = test_image / 255.0
            im = tf.expand_dims(test_image, axis=0)

            # Predict
            pred = model(im, training=False)
            pred_class = tf.argmax(pred, axis=-1).numpy()[0]
            predicted_label = CLASSES[pred_class]
            conf = tf.reduce_max(pred).numpy() * 100

            if predicted_label != "Blank":
                st.session_state.text_buffer += predicted_label
                st.success(f"Prediction: {predicted_label} ({conf:.2f}%)")
            else:
                st.info("Detected: Blank")
    else:
        st.warning("No hand detected.")

# --------------------- Display Final Text Output ---------------------
st.markdown("### ASL to Text Output")
st.text_area("Output Text", value=st.session_state.text_buffer, height=100)

# Clear buffer button
if st.button("Clear Output"):
    st.session_state.text_buffer = ""


