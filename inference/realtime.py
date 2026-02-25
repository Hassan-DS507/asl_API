import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import time


MODEL_PATH = "/home/mohamed_mahmoud/Downloads/run_time/model.tflite"
LABEL_MAP_PATH = "/home/mohamed_mahmoud/Downloads/run_time/sign_to_prediction_index_map.json"

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


print("⏳ Loading resources...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    
    FIXED_FRAMES = 30
    interpreter.resize_tensor_input(input_index, [1, FIXED_FRAMES, 543, 3]) 
    interpreter.allocate_tensors()
    
    print("✅ Model Loaded & Memory Allocated.")
    
except Exception as e:

    try: 
        print("⚠️ Retrying allocation without batch dim...")
        interpreter.resize_tensor_input(input_index, [FIXED_FRAMES, 543, 3])
        interpreter.allocate_tensors()
        print("✅ Model Loaded (No Batch Dim).")
    except Exception as e2:
        print(f"❌ Error loading model: {e2}")
        exit()

try:
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
        idx_to_sign = {v: k for k, v in label_map.items()}
except:
    idx_to_sign = None


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
# there are 543 keypoints in total (468 face + 21 left hand + 33 pose + 21 right hand) nan if not detected instead of 0 to avoid confusion with actual coordinates
    def to_array(landmarks, count):
        if landmarks:
            return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        return [[float('nan'), float('nan'), float('nan')]] * count

    face = to_array(results.face_landmarks, 468)
    lh = to_array(results.left_hand_landmarks, 21)
    pose = to_array(results.pose_landmarks, 33)
    rh = to_array(results.right_hand_landmarks, 21)
    return np.concatenate([face, lh, pose, rh])

# ================= Main loop=================
cap = cv2.VideoCapture(0)
sequence = []
last_prediction = "Waiting..."
prediction_conf = 0.0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # رسم اليدين فقط
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # تجميع البيانات
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-FIXED_FRAMES:] # نحافظ دائماً على آخر 30 فريم

        # التشغيل فقط عند امتلاء الذاكرة
        if len(sequence) == FIXED_FRAMES:
            try:
         
                input_data = np.array(sequence, dtype=np.float32)
                
          
                if len(interpreter.get_input_details()[0]['shape']) == 4:
                     input_data = np.expand_dims(input_data, axis=0)

                # input (without resize or allocate)
                interpreter.set_tensor(input_index, input_data)
                
                # Inference 
                interpreter.invoke()
                
                # استخراج النتيجة
                raw_output = interpreter.get_tensor(output_index)
                
                # --- حل مشكلة الـ Indexing ---
                # لو المخرج [1, 250] -> ناخذ [0] عشان يبقى [250]
                # لو المخرج [250] -> ناخذه زي ما هو
                if raw_output.ndim == 2:
                    prediction_logits = raw_output[0]
                else:
                    prediction_logits = raw_output

                # --- تطبيق الـ Softmax ---
                probs = softmax(prediction_logits)
                
                top_idx = np.argmax(probs)
                current_conf = probs[top_idx]
                
                # فلترة وتحديث
                if current_conf > 0.4: # نسبة ثقة 40%
                    word = idx_to_sign[top_idx] if idx_to_sign else str(top_idx)
                    last_prediction = word
                    prediction_conf = current_conf
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

            except Exception as e:
                print(f"Runtime Error: {e}")
                sequence = [] # تفريغ في حالة الخطأ

        # العرض
        cv2.rectangle(image, (0,0), (640, 60), (0,0,0), -1)
        cv2.putText(image, f"{last_prediction}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Conf: {prediction_conf:.1%}", (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('SignSense Pro - Optimized', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()