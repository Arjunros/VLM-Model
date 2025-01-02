import cv2
import pyttsx3
import speech_recognition as sr
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize LLM (using Hugging Face pipeline for simplicity)
llm = pipeline("text-generation", model="gpt2", device=-1)

# Initialize VLM (Sentence Transformers)
vlm = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to generate response using LLM
def generate_response(prompt):
    return llm(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]

# Function to detect objects in the frame
def detect_objects(frame):
    # Placeholder: Replace with actual object detection (e.g., YOLO, OpenCV)
    return "an object"

# Function to match captions using VLM
def match_caption(frame):
    captions = ["a person", "a robot", "an object"]
    query_embedding = vlm.encode("Frame content", convert_to_tensor=True)
    caption_embeddings = vlm.encode(captions, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(query_embedding, caption_embeddings)
    best_match_idx = similarity.argmax()
    return captions[best_match_idx]

# Function to listen to audio input (speech recognition)
def listen_to_audio():
    with sr.Microphone() as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)
    
    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)  # Recognize the speech
        print(f"User said: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I did not hear anything.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None

# Access the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Accessing webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Object detection or VLM matching
    detected_caption = detect_objects(frame)
    action_prompt = f"The camera detected {detected_caption}. What should the robot do next?"

    # Get LLM response based on detected caption
    response = generate_response(action_prompt)

    # Speak response using text-to-speech engine
    engine.say(response)
    engine.runAndWait()

    # Display the webcam feed
    cv2.imshow("Webcam", frame)

    # Check for user speech command
    user_command = listen_to_audio()
    
    if user_command:
        # If user says "Hey Milo", the system will respond
        if "hey milo" in user_command.lower():
            # Example: Ask the LLM something or trigger any response.
            llm_response = generate_response("Hey Milo, what's up?")
            print(f"LLM Response: {llm_response}")
            engine.say(llm_response)
            engine.runAndWait()
        
        else:
            # Process other user commands
            user_response = generate_response(user_command)
            print(f"LLM Response: {user_response}")
            engine.say(user_response)
            engine.runAndWait()

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
