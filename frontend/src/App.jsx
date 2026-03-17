import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [status, setStatus] = useState('AI Model Initialized');
  const [error, setError] = useState(null);
  const [volume, setVolume] = useState(0);
  
  const isRecordingRef = useRef(false);
  const socketRef = useRef(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const transcriptionEndRef = useRef(null);

  // Sync ref with state for closures
  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  // Auto-scroll to bottom of transcription
  useEffect(() => {
    if (transcriptionEndRef.current) {
      transcriptionEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [transcription]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Connect to Whisper WebSocket
      socketRef.current = new WebSocket(`ws://localhost:9090/ws/transcribe?language=en`);

      socketRef.current.onopen = () => {
        console.log("WebSocket Connected");
        setIsRecording(true);
        isRecordingRef.current = true;
        setStatus('Listening...');
        setError(null);
        setupAudioProcessing(stream);
      };

      socketRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("WebSocket message received:", data);
        if (data.status === 'success' && data.text) {
          setTranscription(prev => (prev ? prev + ' ' + data.text : data.text));
        }
      };

      socketRef.current.onerror = (err) => {
        console.error("WebSocket Error:", err);
        setError("WebSocket connection failed.");
        setStatus('Error');
      };

      socketRef.current.onclose = () => {
        console.log("WebSocket Closed");
        setIsRecording(false);
      };

    } catch (err) {
      console.error("Error accessing microphone:", err);
      setError("Microphone access denied or not available.");
      setStatus('Error');
    }
  };

  const setupAudioProcessing = (stream) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    audioContextRef.current = audioContext;
    
    const source = audioContext.createMediaStreamSource(stream);
    
    // Setup Analyser for volume visualization
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    analyserRef.current = analyser;
    source.connect(analyser);

    // Volume update loop
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    const updateVolume = () => {
      if (!isRecordingRef.current) return;
      analyser.getByteFrequencyData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
      }
      const average = sum / dataArray.length;
      setVolume(average); // Normalize slightly if needed
      animationFrameRef.current = requestAnimationFrame(updateVolume);
    };
    animationFrameRef.current = requestAnimationFrame(updateVolume);

    // Use a ScriptProcessor for simplicity in this MVP
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    let audioBuffer = [];
    const chunkTimeSeconds = 2; // Send chunks every 2 seconds for Whisper

    processor.onaudioprocess = (e) => {
      if (!isRecordingRef.current) return;
      const inputData = e.inputBuffer.getChannelData(0);
      
      // Convert Float32 to Int16
      const int16Data = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        int16Data[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
      }
      
      audioBuffer.push(...int16Data);
      
      const requiredSamples = audioContext.sampleRate * chunkTimeSeconds;
      if (audioBuffer.length >= requiredSamples) {
        const wavBlob = createWavBlob(new Int16Array(audioBuffer), audioContext.sampleRate);
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
          socketRef.current.send(wavBlob);
        }
        audioBuffer = []; // Reset for next chunk
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
  };

  const createWavBlob = (samples, sampleRate) => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 32 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    for (let i = 0; i < samples.length; i++) {
      view.setInt16(44 + i * 2, samples[i], true);
    }

    return new Blob([view], { type: 'audio/wav' });
  };

  const stopRecording = () => {
    setIsRecording(false);
    isRecordingRef.current = false;
    setVolume(0);
    setStatus('AI Model Initialized');
    
    // 1. Send explicit STOP signal to backend to flush its work
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      try {
        socketRef.current.send("STOP");
      } catch (e) {
        console.warn("Could not send stop signal:", e);
      }
    }

    // 2. Kill all UI and Audio elements immediately
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.suspend().then(() => {
        audioContextRef.current.close().catch(e => {});
        audioContextRef.current = null;
      });
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.enabled = false;
        track.stop();
      });
      streamRef.current = null;
    }

    if (socketRef.current) {
      socketRef.current.onmessage = null;
      socketRef.current.onclose = null;
      socketRef.current.onerror = null;
      socketRef.current.close();
      socketRef.current = null;
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-left">
          <div className="status-badge">
            <span className={`status-dot ${status.toLowerCase().includes('error') ? 'error' : ''}`}></span>
            {status}
          </div>
        </div>
        <h1>CORTEXIUM<span>WHISPER</span></h1>
      </header>

      <main className="app-main">
        <div className={`ralph-loop ${isRecording ? 'active' : ''}`}>
          <div className="loop-inner" style={{ 
            transform: `scale(${1 + volume / 80})`,
            borderWidth: `${4 + volume / 10}px`
          }}></div>
          <div className="loop-glow" style={{ opacity: 0.1 + volume / 100 }}></div>
          <div className="volume-bars">
            {[...Array(32)].map((_, i) => (
              <div 
                key={i} 
                className="volume-bar" 
                style={{ 
                  transform: `rotate(${i * (360/32)}deg) translateY(-145px)`,
                  height: `${5 + (volume * (Math.random() * 0.8 + 0.4))}px`,
                  width: `${3 + volume / 40}px`,
                  opacity: isRecording ? 1 : 0
                }}
              ></div>
            ))}
          </div>
          {isRecording && (
            <>
              <div className="wave-pulse wave-1" style={{ 
                transform: `scale(${1 + volume / 40})`,
                opacity: volume / 255
              }}></div>
              <div className="wave-pulse wave-2" style={{ 
                transform: `scale(${1.2 + volume / 30})`,
                opacity: volume / 300
              }}></div>
              <div className="wave-pulse wave-3" style={{ 
                transform: `scale(${1.4 + volume / 20})`,
                opacity: volume / 400
              }}></div>
            </>
          )}
        </div>

        <div className="controls">
          <button 
            className={`record-btn ${isRecording ? 'recording' : ''}`}
            onClick={isRecording ? stopRecording : startRecording}
          >
            <div className="btn-icon"></div>
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        <section className="transcription-container">
          <div className="glass-card">
            <div className="card-header">
              <h3>TRANSCRIPTION</h3>
              <div className="card-actions">
                {transcription && (
                  <button className="clear-btn" onClick={() => setTranscription('')}>
                    Clear
                  </button>
                )}
              </div>
            </div>
            <div className="text-display">
              {transcription || "Your text will appear here..."}
              <div ref={transcriptionEndRef} />
            </div>
          </div>
        </section>
      </main>

      <footer className="api-info">
        Endpoint: localhost:9090 | Model: Whisper Base | <span style={{color: 'var(--accent-cyan)'}}>LIVE MODE</span>
      </footer>
    </div>
  );
}

export default App;
