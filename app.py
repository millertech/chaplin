import gradio as gr
import cv2
import torch
from pipelines.pipeline import InferencePipeline
import time
from huggingface_hub import hf_hub_download
import os
from utils.config import load_config


class ChaplinGradio:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vsr_model = None
        self.download_models()
        self.load_models()
        
        # Video params
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25
        self.last_frame_time = time.time()
        
        # Frame buffer
        self.frame_buffer = []
        self.min_frames = 32  # 2 seconds of video at 16 fps
        self.last_prediction = ""
        print(f"Initialized with device: {self.device}, fps: {self.fps}, min_frames: {self.min_frames}")

    def download_models(self):
        """Download required model files from HuggingFace"""
        # Create directories if they don't exist
        os.makedirs("benchmarks/LRS3/models/LRS3_V_WER19.1", exist_ok=True)
        os.makedirs("benchmarks/LRS3/language_models/lm_en_subword", exist_ok=True)
        
        # Download VSR model files
        hf_hub_download(repo_id="willwade/LRS3_V_WER19.1", 
                       filename="model.pth",
                       local_dir="benchmarks/LRS3/models/LRS3_V_WER19.1")
        hf_hub_download(repo_id="willwade/LRS3_V_WER19.1", 
                       filename="model.json",
                       local_dir="benchmarks/LRS3/models/LRS3_V_WER19.1")
        
        # Download language model files
        hf_hub_download(repo_id="willwade/lm_en_subword", 
                       filename="model.pth",
                       local_dir="benchmarks/LRS3/language_models/lm_en_subword")
        hf_hub_download(repo_id="willwade/lm_en_subword", 
                       filename="model.json",
                       local_dir="benchmarks/LRS3/language_models/lm_en_subword")
        
        print("Models downloaded successfully!")

    def load_models(self):
        """Load models using the InferencePipeline with LRS3 config"""
        config_path = "configs/LRS3_V_WER19.1.ini"
        
        self.vsr_model = InferencePipeline(
            config_path,
            device=self.device,
            detector="mediapipe",
            face_track=True
        )
        print("Model loaded successfully!")

    def process_frame(self, frame):
        """Process frames with buffering"""
        current_time = time.time()
        debug_log = []  # List to collect debug messages
        
        # Add initial debug info
        debug_log.append(f"Current time: {current_time}")
        debug_log.append(f"Last prediction: {self.last_prediction}")
        
        if current_time - self.last_frame_time < self.frame_interval:
            debug_log.append("Skipping frame - too soon")
            return self.last_prediction, "\n".join(debug_log)
            
        self.last_frame_time = current_time
        
        if frame is None:
            debug_log.append("Received None frame")
            return "No video input detected", "\n".join(debug_log)
        
        try:
            debug_log.append(f"Received frame with shape: {frame.shape}")
            
            # Convert frame to grayscale if it's not already
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                debug_log.append("Converted frame to grayscale")
            
            # Add frame to buffer
            self.frame_buffer.append(frame)
            debug_log.append(f"Buffer size now: {len(self.frame_buffer)}/{self.min_frames}")
            
            # Process when we have enough frames
            if len(self.frame_buffer) >= self.min_frames:
                debug_log.append("Processing buffer - have enough frames")
                # Create temp directory if it doesn't exist
                os.makedirs("temp", exist_ok=True)
                
                # Generate temporary video file path
                temp_video = f"temp/frames_{time.time_ns()}.mp4"
                debug_log.append(f"Created temp video path: {temp_video}")
                
                # Get frame dimensions from first frame
                frame_height, frame_width = self.frame_buffer[0].shape[:2]
                debug_log.append(f"Video dimensions: {frame_width}x{frame_height}")
                
                # Create video writer
                out = cv2.VideoWriter(
                    temp_video,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    (frame_width, frame_height),
                    False  # isColor
                )
                
                # Write all frames to video
                for i, f in enumerate(self.frame_buffer):
                    out.write(f)
                debug_log.append(f"Wrote {i+1} frames to video")
                out.release()
                
                # Verify video was created
                if not os.path.exists(temp_video):
                    debug_log.append("Error: Video file was not created!")
                else:
                    debug_log.append(f"Video file created successfully, size: {os.path.getsize(temp_video)} bytes")
                
                # Clear buffer but keep last few frames for continuity
                self.frame_buffer = self.frame_buffer[-8:]  # Keep last 0.5 seconds
                debug_log.append(f"Cleared buffer, kept {len(self.frame_buffer)} frames")
                
                try:
                    # Process the video file using the pipeline
                    debug_log.append("Starting model inference...")
                    predicted_text = self.vsr_model(temp_video)
                    debug_log.append(f"Raw model prediction: '{predicted_text}'")
                    if predicted_text:
                        self.last_prediction = predicted_text
                        debug_log.append(f"Updated last prediction to: '{self.last_prediction}'")
                    else:
                        debug_log.append("Model returned empty prediction")
                    return (self.last_prediction or "Waiting for speech..."), "\n".join(debug_log)
                    
                except Exception as e:
                    error_msg = f"Error during inference: {str(e)}"
                    debug_log.append(error_msg)
                    import traceback
                    debug_log.append(f"Full error: {traceback.format_exc()}")
                    return f"Error processing frames: {str(e)}", "\n".join(debug_log)
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
                        debug_log.append("Cleaned up temp video file")
                    else:
                        debug_log.append("No temp file to clean up")
            
            return (self.last_prediction or "Waiting for speech..."), "\n".join(debug_log)
                
        except Exception as e:
            error_msg = f"Error processing: {str(e)}"
            debug_log.append(error_msg)
            import traceback
            debug_log.append(f"Full error: {traceback.format_exc()}")
            return f"Error processing: {str(e)}", "\n".join(debug_log)


# Create Gradio interface
chaplin = ChaplinGradio()

iface = gr.Interface(
    fn=chaplin.process_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs=[
        gr.Textbox(label="Predicted Text", interactive=False),
        gr.Textbox(label="Debug Log", interactive=False)
    ],
    title="Chaplin - Live Visual Speech Recognition",
    description="Speak clearly into the webcam. The model will process your speech in ~2 second chunks.",
    live=True
)

def main():
    # Load configuration
    config = load_config()
    
    # Setup your Gradio interface
    iface.launch(
        server_port=config["web_config"]["port"],
        share=config["web_config"]["share"]
    )

if __name__ == "__main__":
    main() 