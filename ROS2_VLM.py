import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import base64
import ollama
import pyttsx3
import speech_recognition as sr
import numpy as np

class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(String, 'vlm_response', 10)
        self.bridge = CvBridge()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 160)
        self.recognizer = sr.Recognizer()
        self.latest_frame = None

        self.get_logger().info("VLM Node initialized. Listening...")

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def run_vlm_loop(self):
        while rclpy.ok():
            if self.latest_frame is None:
                continue

            image_path = "/tmp/frame.jpg"
            cv2.imwrite(image_path, self.latest_frame)
            image_base64 = self.encode_image(image_path)

            with sr.Microphone() as source:
                print("🎙 Speak your question (or say 'exit')...")
                self.recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    user_question = self.recognizer.recognize_google(audio)

                    if user_question.lower() in ["exit", "quit", "stop"]:
                        break

                    response = ollama.generate(
                        model="llava",
                        prompt=f"Analyze this image and answer: {user_question}",
                        images=[image_base64]
                    )
                    answer = response["response"]
                    print(f"🤖 MiloX1: {answer}")
                    self.publisher.publish(String(data=answer))
                    self.tts_engine.say(answer)
                    self.tts_engine.runAndWait()

                except Exception as e:
                    print(f" Error: {e}")

    def encode_image(self, path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')


def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    try:
        node.run_vlm_loop()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
