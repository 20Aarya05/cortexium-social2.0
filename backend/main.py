import os
import shutil
import tempfile
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from agents.whisper_agent import WhisperAgent

# Disable symlink warnings and force local cache to avoid permission issues on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize Agent as None globally
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    print(f"INFO:     Initializing Whisper model in {MODELS_DIR}...")
    try:
        # Load model into the local models directory
        agent = WhisperAgent(model_size="base", download_root=MODELS_DIR)
        print("INFO:     Whisper model loaded successfully.")
    except Exception as e:
        print(f"ERROR:    Failed to initialize Whisper model: {e}")
    yield
    print("INFO:     Shutting down...")

app = FastAPI(title="Cortexium Whisper API", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, language: str = "en"):
    await websocket.accept()
    import time
    session_id = int(time.time() * 1000) % 10000
    print(f"INFO: [{session_id}] WebSocket connected. Language: {language}")
    
    is_processing = False
    
    try:
        while True:
            # 1. Wait for ANY message (Text or Binary)
            message = await websocket.receive()
            
            # 2. Check for disconnect or specific text signals
            if message.get("type") == "websocket.disconnect":
                print(f"INFO: [{session_id}] Received disconnect signal.")
                break
                
            if "text" in message:
                if message["text"] == "STOP":
                    print(f"INFO: [{session_id}] STOP signal received from UI.")
                    break
                continue
            
            # 3. Handle binary audio data
            data = message.get("bytes")
            if not data:
                continue

            # Check if client disconnected while receiving
            from fastapi.websockets import WebSocketState
            if websocket.client_state == WebSocketState.DISCONNECTED:
                print(f"INFO: [{session_id}] Client state is DISCONNECTED. Breaking.")
                break

            # Prevent pile-up
            if is_processing:
                continue

            # Whisper needs a file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav.write(data)
                temp_path = temp_wav.name
            
            try:
                if agent:
                    is_processing = True
                    target_lang = "en"
                    
                    import asyncio
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        lambda: agent.transcribe(temp_path, language=target_lang)
                    )
                    
                    # Final check before sending - did they disconnect during transcription?
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        print(f"INFO: [{session_id}] Client disconnected during transcription.")
                        break

                    if result and result.get("text"):
                        prob = result.get("language_probability", 0)
                        text = result["text"].strip()
                        
                        if prob > 0.6 and text:
                            print(f"INFO: [{session_id}] Transcribed: {text}")
                            await websocket.send_json({
                                "text": text,
                                "language": result["language"],
                                "status": "success"
                            })
            except Exception as e:
                print(f"ERROR: [{session_id}] Transcription loop error: {e}")
            finally:
                is_processing = False
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
    except WebSocketDisconnect:
        print(f"INFO: [{session_id}] WebSocket disconnected normally.")
    except Exception as e:
        print(f"ERROR: [{session_id}] Fatal WebSocket error: {e}")
    finally:
        print(f"INFO: [{session_id}] SESSION CLOSED AND CLEANED UP.")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not agent:
        raise HTTPException(status_code=503, detail="Whisper model is still loading or failed to load")
    
    # Save uploaded file to temp
    suffix = os.path.splitext(file.filename)[1]
    if not suffix: suffix = ".wav"
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Transcribe
        result = agent.transcribe(temp_path)
        return result
    except Exception as e:
        print(f"ERROR:    Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    # Use 9090 to avoid the WinError 10013 on common ports
    print("INFO:     Starting server on http://127.0.0.1:9090")
    uvicorn.run(app, host="127.0.0.1", port=9090)
