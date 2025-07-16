# 🤟 ASL Recognition System in Real-Time

## 🚀 [Live Demo]([https://signifyai-asl-recognition.streamlit.app/])

An AI-powered real-time American Sign Language (ASL) recognition system that takes webcam input, detects the hand using MediaPipe, and recognizes ASL alphabets using a custom-trained deep learning model. Includes a user-friendly interface with a live notepad to collect recognized characters.

---

## 🧠 Features

- 📸 Real-time webcam-based input
- ✋ Hand detection and cropping using **MediaPipe**
- 🔍 ASL alphabet recognition with **MobileNetV2**-based custom-trained model  
- 📈 Achieved **99.25% validation accuracy**
- 💻 Clean and responsive **Streamlit UI**
- 📝 Built-in notepad for typing recognized ASL alphabets

---

## 📷 How It Works

1. **Webcam Capture**: The app captures frames from your webcam in real time.
2. **Hand Detection**: MediaPipe is used to detect and crop hand regions from the frame.
3. **Model Prediction**: The cropped hand image is passed to a pretrained **MobileNetV2** model, fine-tuned on custom ASL image data.
4. **Prediction Display**: The predicted letter is shown on screen and automatically added to a notepad area.
5. **Streamlit Interface**: A responsive and easy-to-use interface for smooth real-time interaction.

---

## 🧰 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Model**: TensorFlow/Keras with pretrained **MobileNetV2**
- **Image Processing**: OpenCV + MediaPipe
- **Languages**: Python

---

## 📊 Model Training

- **Base Model**: MobileNetV2 (ImageNet weights)
- **Custom Layers**: Added dense layers on top for ASL classification
- **Dataset**: Custom ASL alphabet dataset
- **Accuracy**: 
  - Training Accuracy: ~99.40%
  - Validation Accuracy: **99.25%**

---

## 🛠️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/asl-realtime-recognition.git
   cd asl-realtime-recognition
