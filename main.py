import torch
import hydra
import cv2
import time
from pipelines.pipeline import InferencePipeline
import numpy as np
from datetime import datetime
from ollama import chat
from pydantic import BaseModel
from pynput import keyboard
from concurrent.futures import ThreadPoolExecutor
import os
from huggingface_hub import hf_hub_download
import tempfile
import shutil
from utils.config import load_config
import sys
import configparser
import pyttsx3

# pydantic model for the chat output
class ChaplinOutput(BaseModel):
    list_of_changes: str
    corrected_text: str

class Chaplin:
    def __init__(self):
        self.vsr_model = None
        self.model_cache_dir = None
        self.setup_model_cache()

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.recording = False
        self.processing_output = False
        self.pre_record_countdown = None
        self.recording_countdown = None
        self.countdown_started = False
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25
        self.kb = keyboard.Controller()
        # Remove persistent tts_engine
        # Circle detection smoothing
        self.p_x, self.p_y, self.p_w, self.p_h = 0, 0, 0, 0
        self.box_alpha = 0.01

    def setup_model_cache(self):
        """Setup cache directory and download models from HuggingFace"""
        self.model_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        print(f"Using model cache directory: {self.model_cache_dir}")
        vsr_model_dir = os.path.join(self.model_cache_dir, "LRS3_V_WER19.1")
        lm_model_dir = os.path.join(self.model_cache_dir, "lm_en_subword")
        print(f"Creating directories:")
        print(f"VSR dir: {vsr_model_dir}")
        print(f"LM dir: {lm_model_dir}")
        os.makedirs(vsr_model_dir, exist_ok=True)
        os.makedirs(lm_model_dir, exist_ok=True)
        try:
            vsr_files = ["model.pth", "model.json"]
            missing_vsr_files = [f for f in vsr_files if not os.path.exists(os.path.join(vsr_model_dir, f))]
            if missing_vsr_files:
                print(f"Downloading VSR model files to {vsr_model_dir}:")
                for file in missing_vsr_files:
                    print(f"  Downloading {file}...")
                    path = hf_hub_download(
                        repo_id="willwade/LRS3_V_WER19.1",
                        filename=file,
                        local_dir=vsr_model_dir
                    )
                    print(f"  Downloaded to: {path}")
            else:
                print("VSR model files already present")
            lm_files = ["model.json", "model.pth"]
            missing_lm_files = [f for f in lm_files if not os.path.exists(os.path.join(lm_model_dir, f))]
            print("\nVerifying downloaded files:")
            for file in vsr_files:
                path = os.path.join(vsr_model_dir, file)
                exists = os.path.exists(path)
                print(f"  {path}: {'✓' if exists else '✗'}")
            for file in lm_files:
                path = os.path.join(lm_model_dir, file)
                exists = os.path.exists(path)
                print(f"  {path}: {'✓' if exists else '✗'}")
            print("\nModel files ready!")
        except Exception as e:
            print(f"Error downloading model files: {e}")
            import traceback
            print(traceback.format_exc())
            raise

    def perform_inference(self, video_path):
        self.processing_output = True
        output = self.vsr_model(video_path)
        for char in output:
            self.kb.press(char)
            time.sleep(0.02)
            self.kb.release(char)
            time.sleep(0.02)
        formatted_time_str = datetime.now().strftime("%B %d, %I:%M %p")
        print(f"\nAt {formatted_time_str}, Judy said: \n")
        print(f"\t: {output}")
        print(f'\n----------------------------------\n')
        # Re-initialize TTS engine for each playback
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 120)
        tts_engine.say(f"{output}")
        tts_engine.runAndWait()
        self.processing_output = False
        return {
            "output": output,
            "video_path": video_path
        }

    def detect_mouth_circle(self, gray, faces):
        for (x, y, w, h) in faces:
            # Smooth face position
            y = int(self.box_alpha * self.p_y + (1 - self.box_alpha) * y)
            x = int(self.box_alpha * self.p_x + (1 - self.box_alpha) * x)
            w = int(self.box_alpha * self.p_w + (1 - self.box_alpha) * w)
            h = int(self.box_alpha * self.p_h + (1 - self.box_alpha) * h)
            self.p_x, self.p_y, self.p_w, self.p_h = x, y, w, h

            # Focus on mouth area
            y_mouth = int(y + h * 0.6)
            h_mouth = int(h * 0.5)
            x_mouth = int(x + w * 0.15)
            w_mouth = int(w * 0.7)

            mouth_roi = gray[y_mouth:y_mouth + h_mouth, x_mouth:x_mouth + w_mouth]
            mouth_roi_blur = cv2.GaussianBlur(mouth_roi, (9, 9), 2)
            mouth_roi_edges = cv2.Canny(mouth_roi_blur, 50, 150)

            circles = cv2.HoughCircles(
                mouth_roi_edges,
                cv2.HOUGH_GRADIENT,
                dp=1.5,
                minDist=int(h_mouth * 0.5),
                param1=100,
                param2=45,  # Less sensitive (was 35)
                minRadius=int(h_mouth * 0.15),  # Larger min radius (was 0.05)
                maxRadius=int(h_mouth * 0.45)
            )

            circle_count = 0
            if circles is not None:
                circles = np.uint16(np.around(circles))
                filtered_circles = []
                for i in circles[0, :]:
                    cx, cy, r = i
                    if (cx > w_mouth * 0.25 and cx < w_mouth * 0.75 and
                        cy > h_mouth * 0.25 and cy < h_mouth * 0.75):
                        filtered_circles.append(i)
                circle_count = len(filtered_circles)
            return circle_count > 0
        return False

    def start_webcam(self):
        # --- Clean up mp4 files at the start of the run ---
        for file in os.listdir():
            if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                os.remove(file)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        ret, prev_frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            cap.release()
            return
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        last_frame_time = time.time()
        futures = []
        output_path = ""
        out = None
        frame_count = 0

        self.recording = False
        self.processing_output = False
        self.pre_record_countdown = None
        self.recording_countdown = None
        self.countdown_started = False

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                for file in os.listdir():
                    if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                        os.remove(file)
                break

            current_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            mouth_circle = self.detect_mouth_circle(gray, faces)

            if not self.processing_output:
                if mouth_circle and not self.recording and not self.countdown_started:
                    self.pre_record_countdown = current_time
                    self.countdown_started = True
                    print("Mouth 'O' detected: Starting 1.5-second countdown.")

            if self.countdown_started and not self.recording:
                elapsed = current_time - self.pre_record_countdown
                if elapsed < 0.5:
                    cv2.putText(frame, "3", (frame.shape[1] // 2 - 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                elif elapsed < 1.0:
                    cv2.putText(frame, "2", (frame.shape[1] // 2 - 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                elif elapsed < 1.5:
                    cv2.putText(frame, "1", (frame.shape[1] // 2 - 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                else:
                    self.recording = True
                    self.recording_countdown = current_time
                    self.countdown_started = False
                    print("Countdown complete: Starting recording.")

            if self.recording:
                elapsed_recording = current_time - self.recording_countdown
                seconds_left = max(0, 5 - int(elapsed_recording))
                # --- Draw recording text on a flipped overlay ---
                overlay = frame.copy()
                cv2.putText(overlay, f"Recording: {seconds_left}", (frame.shape[1] // 2 - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                overlay = cv2.flip(overlay, 1)
                # Blend overlay with frame for flipped text
                alpha = 0.7
                frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
                if seconds_left == 0:
                    self.recording = False
                    self.processing_output = True
                    print("Recording countdown complete: Stopping recording.")

            if current_time - last_frame_time >= self.frame_interval:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
                if self.recording:
                    if out is None:
                        output_path = self.output_prefix + str(time.time_ns() // 1_000_000) + '.mp4'
                        out = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            self.fps,
                            (frame_width, frame_height),
                            False
                        )
                    out.write(compressed_frame)
                    last_frame_time = current_time
                    cv2.circle(compressed_frame, (frame_width - 20, 20), 10, (0, 0, 0), -1)
                    frame_count += 1
                elif not self.recording and frame_count > 0:
                    if out is not None:
                        out.release()
                    if frame_count >= self.fps * 2:
                        futures.append(self.executor.submit(self.perform_inference, output_path))
                    else:
                        os.remove(output_path)
                    output_path = self.output_prefix + str(time.time_ns() // 1_000_000) + '.mp4'
                    out = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        self.fps,
                        (frame_width, frame_height),
                        False
                    )
                    frame_count = 0
                # Flip for front camera effect
                cv2.imshow('Chaplin', cv2.flip(frame, 1))

            for fut in futures:
                if fut.done():
                    result = fut.result()
                    os.remove(result["video_path"])
                    futures.remove(fut)
                    self.processing_output = False
                else:
                    break

            prev_gray = gray.copy()

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    def on_press(self, key):
        if key == keyboard.Key.alt:
            self.recording = not self.recording

def main():
    config = load_config()
    args = sys.argv[3:] if sys.argv[1:2] == ['run'] else sys.argv[1:]
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=')
            if key == "detector":
                config["model_config"]["detector"] = value
            elif key == "config_filename":
                config["model_config"]["config_filename"] = value
    detector = config["model_config"]["detector"]
    gpu_idx = config["model_config"]["gpu_idx"]
    config_filename = config["model_config"].get("config_filename", "./configs/LRS3_V_WER19.1.ini")
    chaplin = Chaplin()
    config_parser = configparser.ConfigParser()
    config_parser.read(config_filename)
    model_dir = chaplin.model_cache_dir
    config_parser['model']['model_path'] = os.path.join(model_dir, "LRS3_V_WER19.1", "model.pth")
    config_parser['model']['model_conf'] = os.path.join(model_dir, "LRS3_V_WER19.1", "model.json")
    config_parser['model']['rnnlm'] = os.path.join(model_dir, "lm_en_subword", "model.pth")
    config_parser['model']['rnnlm_conf'] = os.path.join(model_dir, "lm_en_subword", "model.json")
    temp_config = os.path.join(model_dir, 'temp_config.ini')
    with open(temp_config, 'w') as f:
        config_parser.write(f)
    chaplin.vsr_model = InferencePipeline(
        temp_config,
        device=torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() and gpu_idx >= 0 else "cpu"),
        detector=detector,
        face_track=True
    )
    os.remove(temp_config)
    print("Model loaded successfully!")
    chaplin.start_webcam()

if __name__ == '__main__':
    main()