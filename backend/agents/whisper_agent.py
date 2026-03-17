import os
import logging
import time
import tempfile
from typing import List, Dict, Any, Optional, Generator, Union
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import argparse
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WhisperAgent")

class WhisperAgent:
    """
    A robust agent for audio-to-text transcription using faster-whisper,
    now with microphone support.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        download_root: Optional[str] = None,
        cpu_threads: int = 0,
        num_workers: int = 1
    ):
        """
        Initializes the Whisper model.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        logger.info(f"Loading Whisper model '{model_size}' on '{device}' with '{compute_type}'...")
        start_time = time.time()
        
        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
                cpu_threads=cpu_threads,
                num_workers=num_workers
            )
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def record_audio(self, duration: int = 5, sample_rate: int = 16000) -> str:
        """
        Records audio from the microphone and saves it to a temporary WAV file.

        Args:
            duration: Recording duration in seconds.
            sample_rate: Sample rate for recording (Whisper prefers 16kHz).

        Returns:
            Path to the temporary WAV file.
        """
        logger.info(f"Recording for {duration} seconds...")
        try:
            # Record audio
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished
            logger.info("Recording finished.")

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav.write(temp_file.name, sample_rate, recording)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            raise

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        language: Optional[str] = None,
        task: str = "transcribe",
        vad_filter: bool = True,
        word_timestamps: bool = False,
        vad_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Transcribes an audio file with improved VAD and probability reporting.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Default VAD parameters to reduce silence hallucinations
        if vad_parameters is None:
            vad_parameters = {
                "min_speech_duration_ms": 750,
                "min_silence_duration_ms": 1000,
                "speech_pad_ms": 400
            }

        logger.info(f"Starting {task} for: {audio_path} (Language: {language})")
        start_time = time.time()

        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=beam_size,
                language=language,
                task=task,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                word_timestamps=word_timestamps
            )

            results = []
            full_text = []

            for segment in segments:
                # Filter out very short or empty segments
                if not segment.text.strip():
                    continue
                
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob
                }
                
                # Faster-Whisper probability check: if no_speech_prob is high, skip
                if segment.no_speech_prob > 0.7:
                    continue

                results.append(segment_data)
                full_text.append(segment.text.strip())

            transcription_time = time.time() - start_time
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds.")

            return {
                "text": " ".join(full_text),
                "segments": results,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "transcription_time": transcription_time
            }

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

    def transcribe_from_mic(self, duration: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Captures audio from mic and transcribes it.
        """
        temp_audio = None
        try:
            temp_audio = self.record_audio(duration=duration)
            result = self.transcribe(temp_audio, **kwargs)
            return result
        finally:
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Whisper Agent CLI")
    parser.add_argument("audio", nargs="?", help="Path to audio file (optional if using --mic)")
    parser.add_argument("--mic", action="store_true", help="Capture from microphone")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--model", default="base", help="Model size (tiny, base, small, etc.)")
    
    args = parser.parse_args()

    if not args.audio and not args.mic:
        parser.print_help()
        sys.exit(1)

    try:
        agent = WhisperAgent(model_size=args.model)
        
        if args.mic:
            print(f"\n--- Recording from Microphone ({args.duration}s) ---")
            result = agent.transcribe_from_mic(duration=args.duration)
        else:
            print(f"\n--- Transcribing File: {args.audio} ---")
            result = agent.transcribe(args.audio)
        
        print("\n--- Transcription Result ---")
        print(f"Detected Language: {result['language']} ({result['language_probability']:.2f})")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Full Text: {result['text']}")
        print("\nDetail segments (JSON):")
        print(json.dumps(result['segments'], indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
