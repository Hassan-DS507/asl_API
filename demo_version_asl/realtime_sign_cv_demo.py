import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":0")

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import time
from collections import Counter
import threading
import requests

# ================= مسارات الملفات =================
MODEL_PATH = "/home/mohamed_mahmoud/asl-ecosystem/artifacts/tflite/model.tflite"
LABEL_MAP_PATH = "/home/mohamed_mahmoud/asl-ecosystem/artifacts/tflite/sign_to_prediction_index_map.json"
FIXED_FRAMES = 30
N_LANDMARKS  = 543

# ================= العتبة الصلبة =================
CONFIDENCE_HARD_FLOOR = 0.45

# ================= Sliding Window =================
class SlidingSequence:
    def __init__(self, max_len=FIXED_FRAMES):
        self.max_len = max_len
        self.frames  = []

    def add(self, kp):
        self.frames.append(kp)
        if len(self.frames) > self.max_len:
            self.frames.pop(0)

    def drop_oldest(self, n=1):
        self.frames = self.frames[n:]

    def clear(self):
        self.frames = []

    def is_ready(self):
        return len(self.frames) == self.max_len

    def as_array(self):
        return np.array(self.frames, dtype=np.float32)

# ================= Prediction System =================
class PredictionSystem:
    def __init__(self, stabilization_frames=5):
        self.history             = []
        self.stabilization_frames = stabilization_frames
        self.sentence_buffer     = []
        self.current_stable_word = None

    def add_prediction(self, word: str, confidence: float):
        # نتجاهل التنبؤات الضعيفة تماماً لمنع الضوضاء
        if confidence >= CONFIDENCE_HARD_FLOOR:
            self.history.append(word)
        else:
            self.history.append("")   

        self.history = self.history[-self.stabilization_frames:]

        if len(self.history) == self.stabilization_frames:
            best, count = Counter(self.history).most_common(1)[0]
            # يجب أن تتكرر الكلمة 3 مرات على الأقل لتعتمد
            if count >= 3 and best and best != self.current_stable_word:
                self.current_stable_word = best
                self.sentence_buffer.append(best)
                return best
        return None

    def reset_word_lock(self):
        """يسمح بتكرار نفس الكلمة بعد اختفاء اليد"""
        self.current_stable_word = None
        self.history = []

    def force_sentence_completion(self):
        if self.sentence_buffer:
            buf = self.sentence_buffer.copy()
            self.sentence_buffer     = []
            self.history             = []
            self.current_stable_word = None
            return buf
        return None

# ================= Softmax + LLM API =================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

final_llm_translation = "Awaiting input..."

def call_llm_api(prompt):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY missing."
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant",
                  "messages": [
                      {"role": "system", "content": "Rewrite sign language glosses into one natural English sentence. Output ONLY the sentence without any formatting."},
                      {"role": "user",   "content": prompt}],
                  "temperature": 0.2},
            timeout=10)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip().replace('\n', ' ')
    except Exception as e:
        return f"API Error: {str(e)[:40]}"

def _llm_worker(text):
    global final_llm_translation
    final_llm_translation = "Thinking..."
    final_llm_translation = call_llm_api(text)

# ================= تحميل الموديل =================
print("⏳ Loading model...")
try:
    interpreter    = tf.lite.Interpreter(model_path=MODEL_PATH)
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index    = input_details[0]['index']
    output_index   = output_details[0]['index']

    try:
        interpreter.resize_tensor_input(input_index, [1, FIXED_FRAMES, N_LANDMARKS, 3])
        interpreter.allocate_tensors()
        has_batch = True
    except Exception:
        interpreter.resize_tensor_input(input_index, [FIXED_FRAMES, N_LANDMARKS, 3])
        interpreter.allocate_tensors()
        has_batch = False

    print(f"✅ Model ready | has_batch={has_batch}")
except Exception as e:
    print(f"❌ Model error: {e}")
    exit()

try:
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    idx_to_sign = {v: k for k, v in label_map.items()}
    print(f"✅ Labels: {len(idx_to_sign)} signs")
except Exception as e:
    print(f"⚠️  Labels error: {e}")
    idx_to_sign = None

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

# ================= استخراج النقاط (Raw Data فقط) =================
def extract_landmarks(results):
    def to_arr(lms, n):
        if lms:
            return [[l.x, l.y, l.z] for l in lms.landmark]
        # الموديل مصمم للتعامل مع NaN للقيم المفقودة
        return [[np.nan]*3]*n

    raw = np.concatenate([
        to_arr(results.face_landmarks,       468),
        to_arr(results.left_hand_landmarks,   21),
        to_arr(results.pose_landmarks,        33),
        to_arr(results.right_hand_landmarks,  21),
    ])
    return raw # إرسال البيانات الخام مباشرة للموديل

# ================= الحلقة الرئيسية =================
print("📷 Opening camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sequence    = SlidingSequence(FIXED_FRAMES)
engine      = PredictionSystem(stabilization_frames=5)

last_stable_word           = "Waiting..."
current_raw_conf           = 0.0
completed_sentence_display = ""
_llm_thread                = None
frame_count                = 0

hand_present_prev   = False
no_hand_frame_count = 0   
NO_HAND_RESET_AFTER = 15  

print("✅ Starting. Press SPACE to translate buffer, Q to quit.")

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        frame_count += 1
        image = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t0    = time.time()

        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        hand_present = (results.left_hand_landmarks is not None or
                        results.right_hand_landmarks is not None)

        if hand_present:
            no_hand_frame_count = 0
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        else:
            no_hand_frame_count += 1
            if no_hand_frame_count == NO_HAND_RESET_AFTER:
                sequence.clear()
                engine.reset_word_lock()
                last_stable_word = "..." # مسح الشاشة عند اختفاء اليد

        if hand_present:
            kp = extract_landmarks(results)
            sequence.add(kp)

            if sequence.is_ready():
                try:
                    inp = sequence.as_array()
                    if has_batch:
                        inp = inp[np.newaxis]

                    interpreter.set_tensor(input_index, inp)
                    interpreter.invoke()
                    out    = interpreter.get_tensor(output_index)
                    logits = out[0] if out.ndim == 2 else out
                    probs  = softmax(logits)
                    top    = int(np.argmax(probs))
                    conf   = float(probs[top])
                    word   = idx_to_sign[top] if idx_to_sign else str(top)

                    # تحديث الثقة اللحظية (للعرض فقط)
                    current_raw_conf = conf

                    # محاولة استخراج كلمة مستقرة
                    new_word = engine.add_prediction(word, conf)
                    
                    # ✅ FIX LOGIC BUG: تحديث الكلمة على الشاشة فقط إذا كانت مستقرة
                    if new_word:
                        last_stable_word = new_word
                    elif conf < CONFIDENCE_HARD_FLOOR and no_hand_frame_count > 0:
                         pass # نحتفظ بآخر كلمة مستقرة ما لم تختف اليد تماماً

                except Exception as e:
                    sequence.drop_oldest(3)
                    print(f"[Inference Error] {e}")

        latency_ms = (time.time() - t0) * 1000

        # ── HUD ──────────────────────────────────────────
        cv2.rectangle(image, (0, 0), (640, 170), (0, 0, 0), -1)

        if not hand_present:
            word_color   = (120, 120, 120)
            display_word = "(no hand)"
            display_conf = 0.0
        elif current_raw_conf >= CONFIDENCE_HARD_FLOOR:
            word_color   = (0, 255, 0)
            display_word = last_stable_word
            display_conf = current_raw_conf
        else:
            word_color   = (80, 80, 80)
            display_word = last_stable_word
            display_conf = current_raw_conf

        cv2.putText(image,
            f"Stable: {display_word}  (Raw Conf: {display_conf:.1%})",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, word_color, 2)

        cv2.putText(image,
            f"Min conf: {CONFIDENCE_HARD_FLOOR:.0%}  |  Hand: {'Yes' if hand_present else 'No'}",
            (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 200, 100) if hand_present else (0, 80, 200), 1)

        buf_disp = " ".join(engine.sentence_buffer)[-50:] if engine.sentence_buffer else "(empty)"
        cv2.putText(image,
            f"Buffer: {buf_disp}",
            (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        lat_col = (0, 255, 0) if latency_ms <= 200 else (0, 0, 255)
        cv2.putText(image,
            f"Latency: {latency_ms:.1f}ms  |  Frame#{frame_count}",
            (10, 113), cv2.FONT_HERSHEY_SIMPLEX, 0.52, lat_col, 1)

        llm_d = final_llm_translation[:65] + "..." \
                if len(final_llm_translation) > 65 else final_llm_translation
        cv2.putText(image,
            f"LLM: {llm_d}",
            (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)

        ratio = len(sequence.frames) / FIXED_FRAMES
        bar_color = (0, 200, 100) if hand_present else (80, 80, 80)
        cv2.rectangle(image, (540, 8), (630, 22), (40, 40, 40), -1)
        cv2.rectangle(image, (540, 8), (int(540 + 90*ratio), 22), bar_color, -1)
        cv2.putText(image, "seq", (544, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

        cv2.imshow("SignSense Pro v4.1", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            done = engine.force_sentence_completion()
            if done:
                completed_sentence_display = " ".join(done)
                if _llm_thread is None or not _llm_thread.is_alive():
                    _llm_thread = threading.Thread(
                        target=_llm_worker,
                        args=(completed_sentence_display,),
                        daemon=True)
                    _llm_thread.start()

cap.release()
cv2.destroyAllWindows()
print("👋 Done.")