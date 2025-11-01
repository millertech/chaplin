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

        # flag to toggle recording
        self.recording = False

        self.processing_output = False

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

        # pynput keyboard listener
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()
        self.kb = keyboard.Controller()

        # Initialize the TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 120)

        # Lip movement detection params
        self.alpha = 0.35
        self.mean_face_movement_filtered = 0.0
        self.mean_mouth_movement_filtered = 0.0
        self.last_lip_movement_time = None

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
        self.tts_engine.say(f"{output}")
        self.tts_engine.runAndWait()
        self.processing_output = False
        return {
            "output": output,
            "video_path": video_path
        }

    def detect_lip_movement(self, prev_gray, gray, faces):
        mouth_moving = False
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            prev_face_roi = prev_gray[y:y+h, x:x+w]
            mouth_roi = gray[y + h//2:y + h, x:x + w]
            prev_mouth_roi = prev_gray[y + h//2:y + h, x:x + w]
            flow_face = cv2.calcOpticalFlowFarneback(prev_face_roi, face_roi, None,
                                                     pyr_scale=0.5, levels=3, winsize=15,
                                                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag_face, _ = cv2.cartToPolar(flow_face[..., 0], flow_face[..., 1])
            mean_face_movement = np.mean(mag_face)
            flow_mouth = cv2.calcOpticalFlowFarneback(prev_mouth_roi, mouth_roi, None,
                                                      pyr_scale=0.5, levels=3, winsize=15,
                                                      iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag_mouth, _ = cv2.cartToPolar(flow_mouth[..., 0], flow_mouth[..., 1])
            mean_mouth_movement = np.mean(mag_mouth)
            self.mean_face_movement_filtered = self.alpha * mean_face_movement + (1 - self.alpha) * self.mean_face_movement_filtered
            self.mean_mouth_movement_filtered = self.alpha * mean_mouth_movement + (1 - self.alpha) * self.mean_mouth_movement_filtered
            filtered_movement_diff = self.mean_mouth_movement_filtered - self.mean_face_movement_filtered
            print(f'D:{mean_mouth_movement - mean_face_movement:.2f}  \tFD:{filtered_movement_diff:.2f}  \tM:{self.mean_mouth_movement_filtered:.2f}  \tF:{self.mean_face_movement_filtered:.2f}')
            if self.mean_mouth_movement_filtered > 1.1 and self.mean_mouth_movement_filtered < 1.2:
                mouth_moving = True
                self.last_lip_movement_time = time.time()
                cv2.putText(gray, f"Lips Moving!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
            cv2.rectangle(gray, (x, y + h//2), (x + w, y + h), (255, 0, 0), 2)
        return mouth_moving

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        last_frame_time = time.time()
        futures = []
        output_path = ""
        out = None
        frame_count = 0
        self.recording = False
        self.last_lip_movement_time = None

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
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            mouth_moving = self.detect_lip_movement(prev_gray, gray, faces)
            if self.processing_output:
                # Start recording if lips start moving
                if mouth_moving and not self.recording:
                    self.recording = True
                    self.last_lip_movement_time = time.time()
                    print("Lip movement detected: Starting recording.")

            # Stop recording 2 seconds after lips stop moving
            if self.recording:
                if mouth_moving:
                    self.last_lip_movement_time = time.time()
                elif self.last_lip_movement_time and (current_time - self.last_lip_movement_time > 6):
                    self.recording = False
                    print("No lip movement for 6 seconds: Stopping recording.")

            #stop recording after 10 seconds max
            if self.recording and self.last_lip_movement_time and (current_time - self.last_lip_movement_time > 10):
                self.recording = False
                print("Max recording time reached: Stopping recording.")    

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
                        self.processing_output = True
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
                cv2.imshow('Chaplin', cv2.flip(compressed_frame, 1))

            for fut in futures:
                if fut.done():
                    result = fut.result()
                    os.remove(result["video_path"])
                    futures.remove(fut)
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