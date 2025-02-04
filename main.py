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

    def setup_model_cache(self):
        """Setup cache directory and download models from HuggingFace"""
        # Use a persistent directory in the project
        self.model_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        print(f"Using model cache directory: {self.model_cache_dir}")
        
        # Create directories if they don't exist
        vsr_model_dir = os.path.join(self.model_cache_dir, "LRS3_V_WER19.1")
        lm_model_dir = os.path.join(self.model_cache_dir, "lm_en_subword")
        
        print(f"Creating directories:")
        print(f"VSR dir: {vsr_model_dir}")
        print(f"LM dir: {lm_model_dir}")
        
        os.makedirs(vsr_model_dir, exist_ok=True)
        os.makedirs(lm_model_dir, exist_ok=True)
        
        try:
            # Download VSR model files
            vsr_files = [
                "model.pth",
                "model.json"
            ]
            
            # Check which files need downloading
            missing_vsr_files = [f for f in vsr_files 
                                if not os.path.exists(os.path.join(vsr_model_dir, f))]
            
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

            # Download language model files
            lm_files = [
                "model.json",
                "model.pth"
            ]
            
            # Check which files need downloading
            missing_lm_files = [f for f in lm_files 
                               if not os.path.exists(os.path.join(lm_model_dir, f))]
            
            if missing_lm_files:
                print(f"Downloading language model files to {lm_model_dir}:")
                for file in missing_lm_files:
                    print(f"  Downloading {file}...")
                    path = hf_hub_download(
                        repo_id="willwade/lm_en_subword",
                        filename=file,
                        local_dir=lm_model_dir
                    )
                    print(f"  Downloaded to: {path}")
            else:
                print("Language model files already present")
            
            # Verify files exist
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
        # perform inference on the video with the vsr model
        output = self.vsr_model(video_path)

        # write the raw output
        keyboard.write(output)

        # shift left to select the entire output
        cmd = ""
        for i in range(len(output)):
            cmd += 'shift+left, '
        cmd = cmd[:-2]
        keyboard.press_and_release(cmd)

        # perform inference on the raw output to get back a "correct" version
        response = chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'system',
                    'content': f"You are an assistant that helps make corrections to the output of a lipreading model. The text you will receive was transcribed using a video-to-text system that attempts to lipread the subject speaking in the video, so the text will likely be imperfect.\n\nIf something seems unusual, assume it was mistranscribed. Do your best to infer the words actually spoken, and make changes to the mistranscriptions in your response. Do not add more words or content, just change the ones that seem to be out of place (and, therefore, mistranscribed). Do not change even the wording of sentences, just individual words that look nonsensical in the context of all of the other words in the sentence.\n\nAlso, add correct punctuation to the entire text. ALWAYS end each sentence with the appropriate sentence ending: '.', '?', or '!'. The input text in all-caps, although your respose should be capitalized correctly and should NOT be in all-caps.\n\nReturn the corrected text in the format of 'list_of_changes' and 'corrected_text'."
                },
                {
                    'role': 'user',
                    'content': f"Transcription:\n\n{output}"
                }
            ],
            format=ChaplinOutput.model_json_schema()
        )

        # get only the corrected text
        chat_output = ChaplinOutput.model_validate_json(
            response.message.content)

        # if last character isn't a sentence ending (happens sometimes), add a period
        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        # write the corrected text
        keyboard.write(chat_output.corrected_text + " ")

        # return the corrected text and the video path
        return {
            "output": chat_output.corrected_text,
            "video_path": video_path
        }

    def start_webcam(self):
        # init webcam
        cap = cv2.VideoCapture(0)

        # set webcam resolution, and get frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        last_frame_time = time.time()

        futures = []
        output_path = ""
        out = None
        frame_count = 0

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # remove any remaining videos that were saved to disk
                for file in os.listdir():
                    if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                        os.remove(file)
                break

            current_time = time.time()

            # conditional ensures that the video is recorded at the correct frame rate
            if current_time - last_frame_time >= self.frame_interval:
                ret, frame = cap.read()
                if ret:
                    # frame compression
                    encode_param = [
                        int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    compressed_frame = cv2.imdecode(
                        buffer, cv2.IMREAD_GRAYSCALE)

                    if self.recording:
                        if out is None:
                            output_path = self.output_prefix + \
                                str(time.time_ns() // 1_000_000) + '.mp4'
                            out = cv2.VideoWriter(
                                output_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                self.fps,
                                (frame_width, frame_height),
                                False  # isColor
                            )

                        out.write(compressed_frame)

                        last_frame_time = current_time

                        # circle to indicate recording, only appears in the window and is not present in video saved to disk
                        cv2.circle(compressed_frame, (frame_width -
                                                      20, 20), 10, (0, 0, 0), -1)

                        frame_count += 1
                    # check if not recording AND video is at least 2 seconds long
                    elif not self.recording and frame_count > 0:
                        if out is not None:
                            out.release()

                        # only run inference if the video is at least 2 seconds long
                        if frame_count >= self.fps * 2:
                            futures.append(self.executor.submit(
                                self.perform_inference, output_path))
                        else:
                            os.remove(output_path)

                        output_path = self.output_prefix + \
                            str(time.time_ns() // 1_000_000) + '.mp4'
                        out = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            self.fps,
                            (frame_width, frame_height),
                            False  # isColor
                        )

                        frame_count = 0

                    # display the frame in the window
                    cv2.imshow('Chaplin', cv2.flip(compressed_frame, 1))

            # ensures that videos are handled in the order they were recorded
            for fut in futures:
                if fut.done():
                    result = fut.result()
                    # once done processing, delete the video with the video path
                    os.remove(result["video_path"])
                    futures.remove(fut)
                else:
                    break

        # release everything
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    def on_press(self, key):
        if key == keyboard.Key.alt:  # or Key.alt_l for left alt specifically
            self.recording = not self.recording


def main():
    # Load configuration
    config = load_config()
    
    # Override config with command line arguments
    args = sys.argv[3:] if sys.argv[1:2] == ['run'] else sys.argv[1:]
    
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=')
            if key == "detector":
                config["model_config"]["detector"] = value
            elif key == "config_filename":
                config["model_config"]["config_filename"] = value
    
    # Use configuration values
    detector = config["model_config"]["detector"]
    gpu_idx = config["model_config"]["gpu_idx"]
    config_filename = config["model_config"].get("config_filename", "./configs/LRS3_V_WER19.1.ini")
    
    chaplin = Chaplin()
    
    # Convert config paths to absolute paths
    config_parser = configparser.ConfigParser()
    config_parser.read(config_filename)
    
    # Get the model directory that was actually used
    model_dir = chaplin.model_cache_dir
    
    # Update paths to use the actual model directory
    config_parser['model']['model_path'] = os.path.join(model_dir, "LRS3_V_WER19.1", "model.pth")
    config_parser['model']['model_conf'] = os.path.join(model_dir, "LRS3_V_WER19.1", "model.json")
    config_parser['model']['rnnlm'] = os.path.join(model_dir, "lm_en_subword", "model.pth")
    config_parser['model']['rnnlm_conf'] = os.path.join(model_dir, "lm_en_subword", "model.json")
    
    # Write the updated config to a temporary file
    temp_config = os.path.join(model_dir, 'temp_config.ini')
    with open(temp_config, 'w') as f:
        config_parser.write(f)
    
    # load the model using the config file with absolute paths
    chaplin.vsr_model = InferencePipeline(
        temp_config,
        device=torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() and gpu_idx >= 0 else "cpu"),
        detector=detector,
        face_track=True
    )
    
    # Clean up temporary config
    os.remove(temp_config)
    
    print("Model loaded successfully!")

    # start the webcam video capture
    chaplin.start_webcam()


if __name__ == '__main__':
    main()
