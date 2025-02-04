# Chaplin

![Chaplin Thumbnail](./thumbnail.png)

A visual speech recognition (VSR) tool that reads your lips in real-time and types whatever you silently mouth. Available both as a command-line tool and a web interface.

## Versions

### Command Line Version
The command-line version runs locally and types text directly where your cursor is positioned.

#### Setup
1. Clone the repository and cd into it:
   ```bash
   git clone https://github.com/amanvirparhar/chaplin
   cd chaplin
   ```
2. Install [`uv`](https://github.com/astral-sh/uv)
3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

#### Usage
1. Run:
   ```bash
   uv run main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
   ```
2. Press `alt/option` key to start/stop recording
3. Press `q` to exit

### Web Interface Version
A Gradio-based web interface that runs in your browser.

#### Setup
1. Install dependencies:
   ```bash
   uv pip install -r requirements-gradio.txt
   ```

#### Usage
1. Start the Gradio server:
   ```bash
   uv run app.py
   ```
2. Open your browser to the displayed URL (usually http://localhost:7860)

## Models
Both versions use the same HuggingFace models:
- [LRS3_V_WER19.1](https://huggingface.co/willwade/LRS3_V_WER19.1) - Visual speech recognition
- [lm_en_subword](https://huggingface.co/willwade/lm_en_subword) - Language model

Models are automatically downloaded on first run.

## Configuration
The application can be configured using either:
- Command line arguments (for main.py)
- Environment variables
- config.yaml file

Example config.yaml:
```yaml
version: "cli"  # or "web"
requirements_file: "requirements.txt"  # or "requirements-gradio.txt"
model_config:
  detector: "mediapipe"
  gpu_idx: 0
web_config:
  port: 7860
  share: false
```

## Development
- `main.py` - Command line interface
- `app.py` - Gradio web interface
- `requirements.txt` - Dependencies for CLI version
- `requirements-gradio.txt` - Dependencies for web version

### macOS Setup
1. Go to System Preferences > Security & Privacy > Privacy > Input Monitoring
2. Add Terminal (or your IDE) to the list of allowed applications
3. Restart your terminal/IDE