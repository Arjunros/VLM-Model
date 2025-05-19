# 🤖 Articubot One – VLM Interface Node

This repository is part of the `articubot_one` ROS 2 package. It contains a Vision-Language Model (VLM) node that integrates:

- Real-time camera input via ROS 2
- Image-to-text reasoning using [Ollama](https://ollama.com/)
- Voice input using `SpeechRecognition`
- Voice output using `pyttsx3`

This makes it possible for a robot like **Milo X1** to understand visual scenes and respond to voice queries with voice + ROS topic output.

---

## 📦 Node: `vlm_node.py`

### Features
- Subscribes to `/camera/image_raw` topic (sensor_msgs/Image)
- Uses `cv_bridge` to convert ROS image messages to OpenCV
- Listens for spoken queries using microphone
- Sends image + prompt to Ollama’s LLaVA model
- Speaks the result out loud and publishes it to `vlm_response` topic

---

## 🧠 Prerequisites

- **ROS 2** (tested on Humble or newer)
- Python packages:
  ```bash
  pip install ollama pyttsx3 SpeechRecognition opencv-python numpy
