# Chaplin

![Chaplin Thumbnail](./thumbnail.png)

A visual speech recognition (VSR) tool that reads your lips in real-time and types whatever you silently mouth. Runs fully locally.

Relies on a [model](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages?tab=readme-ov-file#autoavsr-models) trained on the [Lip Reading Sentences 3](https://mmai.io/datasets/lip_reading/) dataset as part of the [Auto-AVSR](https://github.com/mpc001/auto_avsr) project.

Watch a demo of Chaplin [here](https://youtu.be/qlHi0As2alQ).

## Setup

1. Clone the repository, and `cd` into it:
   ```bash
   git clone https://github.com/amanvirparhar/chaplin
   cd chaplin
   ```
2. The required model components are hosted on HuggingFace:
   - [LRS3_V_WER19.1](https://huggingface.co/willwade/LRS3_V_WER19.1)
   - [lm_en_subword](https://huggingface.co/willwade/lm_en_subword)

   They will be automatically downloaded when you first run the application.

3. Install and run `ollama`, and pull the [`llama3.2`](https://ollama.com/library/llama3.2) model.
4. Install [`uv`](https://github.com/astral-sh/uv).

## Usage

1. Run the following command:
   ```bash
   sudo uv run --with-requirements requirements.txt --python 3.12 main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
   ```
2. Once the camera feed is displayed, you can start "recording" by pressing the `option` key (Mac) or the `alt` key (Windows/Linux), and start mouthing words.
3. To stop recording, press the `option` key (Mac) or the `alt` key (Windows/Linux) again. You should see some text being typed out wherever your cursor is.
4. To exit gracefully, focus on the window displaying the camera feed and press `q`.