from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import json
import tempfile

app = FastAPI()

MODEL_PATH = "model.tflite"
LABEL_MAP_PATH = "sign_to_prediction_index_map.json"

# تحميل الموديل مرة واحدة
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

FIXED_FRAMES = 30
interpreter.resize_tensor_input(input_index, [1, FIXED_FRAMES, 543, 3])
interpreter.allocate_tensors()

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
    idx_to_sign = {v: k for k, v in label_map.items()}

mp_holistic = mp.solutions.holistic

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def extract_landmarks(results):
    def to_array(landmarks, count):
        if landmarks:
            return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        return [[float('nan'), float('nan'), float('nan')]] * count

    face = to_array(results.face_landmarks, 468)
    lh = to_array(results.left_hand_landmarks, 21)
    pose = to_array(results.pose_landmarks, 33)
    rh = to_array(results.right_hand_landmarks, 21)
    return np.concatenate([face, lh, pose, rh])

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    # حفظ الفيديو مؤقتًا
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()

    cap = cv2.VideoCapture(temp.name)
    sequence = []

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            keypoints = extract_landmarks(results)
            sequence.append(keypoints)

            sequence = sequence[-FIXED_FRAMES:]

            if len(sequence) == FIXED_FRAMES:
                break

    cap.release()

    if len(sequence) < FIXED_FRAMES:
        return {"error": "Video too short (need at least 30 frames)"}

    input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_index)
    prediction_logits = raw_output[0] if raw_output.ndim == 2 else raw_output

    probs = softmax(prediction_logits)

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    word = idx_to_sign[top_idx]

    return {
        "prediction": word,
        "confidence": confidence
    }