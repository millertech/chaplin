import gradio as gr
import cv2
import torch
from pipelines.pipeline import InferencePipeline
import time


class ChaplinGradio:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vsr_model = None
        self.load_models()
        
        # Video params
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25
        self.last_frame_time = time.time()

    def load_models(self):
        """Load models using the InferencePipeline with HF Space defaults"""
        config = {
            "model": {
                "name": "chaplin_vsr",
                "weights": "models/chaplin_vsr.pth",
                "detector": "mediapipe"
            }
        }
        
        self.vsr_model = InferencePipeline(
            config,
            device=self.device,
            detector="mediapipe",
            face_track=True
        )
        print("Model loaded successfully!")

    def process_frame(self, frame):
        """Process a single frame with rate limiting and compression"""
        current_time = time.time()
        
        if current_time - self.last_frame_time < self.frame_interval:
            return None
            
        self.last_frame_time = current_time
        
        if frame is None:
            return "No video input detected"
        
        # Compress frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        
        # Run inference using the VSR model
        predicted_text = self.vsr_model.process_frame(compressed_frame)
        
        return predicted_text


# Create Gradio interface
chaplin = ChaplinGradio()

iface = gr.Interface(
    fn=chaplin.process_frame,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=gr.Textbox(label="Predicted Text"),
    title="Chaplin - Live Visual Speech Recognition",
    description="Use your webcam to perform real-time visual speech recognition.",
    live=True
)

if __name__ == "__main__":
    iface.launch() 