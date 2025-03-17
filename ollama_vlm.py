import cv2
import ollama
import pyttsx3
import speech_recognition as sr
import numpy as np
import base64
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)  

recognizer = sr.Recognizer()

cap = cv2.VideoCapture(0)

def encode_image(image_path):
    """Convert image to base64 for LLaVA input."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    cv2.imshow("Live Feed - Press 'q' to Quit", frame)

    image_path = "frame.jpg"
    cv2.imwrite(image_path, frame)

    image_base64 = encode_image(image_path)

    with sr.Microphone() as source:
        print(" Speak your question (or say 'exit' to stop)...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            user_question = recognizer.recognize_google(audio)
            print(f" You asked: {user_question}")

            if user_question.lower() in ["exit", "quit", "stop"]:
                break

            # Use LLaVA to analyze the image
            response = ollama.generate(
                model="llava",
                prompt=f"Analyze this image and answer: {user_question}",
                images=[image_base64]  
            )

            answer = response["response"]
            print(f"🤖 MiloX1: {answer}")

            # Speak the answer
            tts_engine.say(answer)
            tts_engine.runAndWait()

        except sr.UnknownValueError:
            print(" Couldn't understand, please try again.")
        except sr.RequestError:
            print(" Speech Recognition API error.")
        except Exception as e:
            print(f" Error: {e}")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
