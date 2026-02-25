from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import json
import tempfile
import os
from collections import Counter
import requests
from dotenv import load_dotenv

# ================= LOAD ENV =================
load_dotenv()

# ================= CONFIG =================
MODEL_PATH = "model.tflite"
LABEL_MAP_PATH = "sign_to_prediction_index_map.json"

FIXED_FRAMES = 30
N_LANDMARKS = 543
CONFIDENCE_HARD_FLOOR = 0.45

app = FastAPI(title="SignSense Pro Video API")

# ================= Sliding Window =================
class SlidingSequence:
    def __init__(self, max_len=FIXED_FRAMES):
        self.max_len = max_len
        self.frames = []

    def add(self, kp):
        self.frames.append(kp)
        if len(self.frames) > self.max_len:
            self.frames.pop(0)

    def is_ready(self):
        return len(self.frames) == self.max_len

    def as_array(self):
        return np.array(self.frames, dtype=np.float32)

# ================= Prediction System =================
class PredictionSystem:
    def __init__(self, stabilization_frames=5):
        self.history = []
        self.stabilization_frames = stabilization_frames
        self.sentence_buffer = []
        self.current_stable_word = None

    def add_prediction(self, word, confidence):
        if confidence >= CONFIDENCE_HARD_FLOOR:
            self.history.append(word)
        else:
            self.history.append("")

        self.history = self.history[-self.stabilization_frames:]

        if len(self.history) == self.stabilization_frames:
            best, count = Counter(self.history).most_common(1)[0]
            if count >= 3 and best and best != self.current_stable_word:
                self.current_stable_word = best
                self.sentence_buffer.append(best)

# ================= LOAD MODEL =================
print("⏳ Loading model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

try:
    interpreter.resize_tensor_input(
        input_index,
        [1, FIXED_FRAMES, N_LANDMARKS, 3]
    )
except:
    pass

interpreter.allocate_tensors()

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

idx_to_sign = {v: k for k, v in label_map.items()}

print("✅ Model ready")

mp_holistic = mp.solutions.holistic

# ================= Helpers =================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def extract_landmarks(results):
    def to_arr(lms, n):
        if lms:
            return [[l.x, l.y, l.z] for l in lms.landmark]
        return [[np.nan]*3]*n

    return np.concatenate([
        to_arr(results.face_landmarks, 468),
        to_arr(results.left_hand_landmarks, 21),
        to_arr(results.pose_landmarks, 33),
        to_arr(results.right_hand_landmarks, 21),
    ])

def call_llm_api(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not found"

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": "Convert the following sign language gloss words into one simple natural English sentence. Only output the sentence."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2
            },
            timeout=20
        )

        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"

# ================= VIDEO ENDPOINT =================
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    try:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(await file.read())
        temp.close()

        cap = cv2.VideoCapture(temp.name)

        if not cap.isOpened():
            return {"error": "Video could not be opened"}

        sequence = SlidingSequence()
        engine = PredictionSystem()

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                hand_present = (
                    results.left_hand_landmarks is not None or
                    results.right_hand_landmarks is not None
                )

                if not hand_present:
                    continue

                kp = extract_landmarks(results)
                sequence.add(kp)

                if not sequence.is_ready():
                    continue

                inp = sequence.as_array()

                if inp.shape != (FIXED_FRAMES, N_LANDMARKS, 3):
                    continue

                inp = inp[np.newaxis]

                interpreter.set_tensor(input_index, inp)
                interpreter.invoke()

                out = interpreter.get_tensor(output_index)

                if out.ndim == 2:
                    logits = out[0]
                elif out.ndim == 1:
                    logits = out
                else:
                    continue

                probs = softmax(logits)

                if probs.ndim == 0:
                    continue

                top = int(np.argmax(probs))
                conf = float(probs[top])
                word = idx_to_sign.get(top, str(top))

                engine.add_prediction(word, conf)

        cap.release()
        os.unlink(temp.name)

        raw_sentence = " ".join(engine.sentence_buffer)

        llm_sentence = None
        if raw_sentence:
            llm_sentence = call_llm_api(raw_sentence)

        return {
            "words": engine.sentence_buffer,
            "raw_sentence": raw_sentence,
            "llm_sentence": llm_sentence
        }

    except Exception as e:
        return {"error": str(e)}
print("API KEY:", os.getenv("GROQ_API_KEY"))