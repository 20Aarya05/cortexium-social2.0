# Cortexium Whisper - Real-time English Transcription

A premium AI-powered live transcription application that converts English speech into text with near-zero latency. Built with a futuristic glassmorphic UI and a robust Python backend using the Faster-Whisper engine.

## 🚀 Features
- **Real-time Transcription**: Capture audio directly from the browser and see text appears instantly.
- **English-Only Focus**: Optimized for high-accuracy English speech recognition.
- **Dynamic Visualizer**: Interactive "Ralph Loop" visualizer that reacts to your voice volume.
- **Hard-Stop Handshake**: Immediate session cleanup to prevent "ghost" recording or background processing.
- **Minimalist UI**: Sleek, dark-mode design with premium aesthetics.

## 🛠️ Tech Stack
- **Frontend**: React, Vite, CSS (Vanilla), Web Audio API.
- **Backend**: FastAPI (Python), Faster-Whisper (Whisper Base model).
- **Communication**: WebSockets for low-latency streaming.

## 🏗️ Architecture
1. **Audio Capture**: Browser's MediaRecorder/Web Audio API captures raw PCM data.
2. **Streaming**: 2-second audio chunks are sent as binary blobs over a WebSocket.
3. **Transcription**: The backend saves chunks to a temporary buffer and passes them to the `WhisperAgent`.
4. **Feedback**: Transcribed text and English confidence scores are sent back to the frontend.
5. **Display**: React updates the UI and scrolls to the latest transcription.

## 📋 Installation

### Prerequisites
- Node.js (v18+)
- Python (3.9+)
- FFmpeg (Required by Whisper for audio processing)

### Backend Setup
1. Navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup
1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

## 🏃 Running the App

1. **Start Backend**:
   ```bash
   cd backend
   .\venv\Scripts\python main.py
   ```
2. **Start Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser.

## 🤖 Agents
- **WhisperAgent**: A specialized wrapper around `faster-whisper`. It handles model loading, VAD (Voice Activity Detection), and transcription with specific confidence thresholds to ignore silence.

---
Built by [20Aarya05](https://github.com/20Aarya05)
