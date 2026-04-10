🚀 Bi-directional Multimodal System for Real-time Translation between Deaf, Blind and Hearing Individuals


🌍 Overview

The Bi-directional Multimodal Communication System is an intelligent assistive platform that enables seamless real-time communication between Deaf, Blind, and Hearing individuals.

Unlike traditional systems that support only one-way translation, this project introduces a complete bidirectional communication loop:

➡️ Sign Language ↔ Text ↔ Speech

It integrates Computer Vision, Deep Learning, Natural Language Processing, and Avatar Rendering into a unified system.

🎯 Key Features
🔁 Bidirectional Communication
👁️ Sign Language Recognition (MediaPipe + CNN)
🧠 NLP-based Sentence Refinement (LLaMA)
🗣️ Speech Recognition & Text-to-Speech
🧍 Avatar-based Sign Language Generation
🌐 Multilingual Support (English, Hindi, Telugu)
⚡ Real-time Performance (25–30 FPS)
💻 Runs on Standard Hardware (No special sensors)
🏗️ System Architecture
Input (Camera / Mic)
        ↓
MediaPipe (Hand Landmarks)
        ↓
CNN Model (Gesture Recognition)
        ↓
NLP Engine (Sentence Refinement)
        ↓
Output (Text / Speech / Avatar)

🔄 Workflow
Capture input (gesture / speech / text)
Extract features (MediaPipe landmarks)
Classify gestures using CNN
Refine output using NLP
Generate output (Text / Speech / Avatar)

📊 Model Performance
🔹 Training Accuracy & Loss

📈 Performance Metrics
Metric	Value
Accuracy	92–96%
FPS	25–30
Latency	<120 ms

🔍 Insights
Rapid convergence within first few epochs
Minimal overfitting (train ≈ validation curves)
Stable low loss indicates strong generalization

📂 Dataset
Custom Indian Sign Language (ISL) dataset
~17,200 samples
Includes:
Alphabets (A–Z)
Numbers (0–9)
Common phrases
🔧 Preprocessing:
Grayscale conversion
Image resizing
Normalization
Data augmentation

🧠 Tech Stack
Category	Tools
Language	Python 3.9
Deep Learning	TensorFlow, Keras
Computer Vision	OpenCV, MediaPipe
NLP	LLaMA (Ollama)
Speech	SpeechRecognition, gTTS
Web	Flask

⚙️ Installation & Setup
🔹 1. Clone Repository

git clone [https://github.com/your-username/your-repo-name](https://github.com/NikhithaNeelam/BI-DIRECTIONAL-MULTIMODAL-SYSTEM-FOR-TRANSLATION-BETWEEN-DEAF-BLIND-AND-HEARING-INDIVIDUALS).git

cd BI-DIRECTIONAL-MULTIMODAL-SYSTEM-FOR-TRANSLATION-BETWEEN-DEAF-BLIND-AND-HEARING-INDIVIDUALS

🔹 2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate   # Windows

🔹 3. Install Dependencies
pip install -r requirements.txt

🔹 4. Run Application

ollama serve
python app.py

🎥 Input → Output Flow
Input Type	Output
✋ Gesture	Text + Speech
🎤 Speech	Text + Avatar
⌨️ Text	Avatar
🖥️ Web Interface
Live camera input
Real-time predictions
Language selection
Output display (Text + Avatar + Audio)

🏆 Novelty / Contribution

✔ Full bidirectional communication system
✔ Supports Deaf + Blind + Hearing users simultaneously
✔ Combines Vision + NLP + Speech + Avatar
✔ Lightweight & deployable (no expensive hardware)

⚠️ Limitations
Sensitive to lighting conditions
Limited gesture vocabulary
No facial expression recognition
🔮 Future Scope
Transformer-based sign recognition
Emotion-aware gestures
Mobile app deployment
Expanded multilingual support

📜 References
IEEE Research Papers
arXiv Publications
Sign Language Recognition Studies


⭐ Final Note

This project demonstrates a real-world, scalable solution for inclusive communication, combining AI and accessibility to bridge a critical societal gap.
