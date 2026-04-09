"""
camera.py — COMPLETE FIXED VERSION
Fixes:
  1. TFLite CNN detects ISL signs (A-Z, 0-9) after training
  2. Template matching for custom recorded phrases (BYE, CAN I GO etc.)
  3. Debounce: same sign only added every 2 seconds
  4. Normalisation identical to preprocess_datasets.py
  5. Debug mode: set DEBUG=True to see distances in terminal
"""
import cv2, json, os, time
import numpy as np
import mediapipe as mp

# ── CONFIG ──────────────────────────────────────────────
CNN_MODEL_PATH   = "model/isl_model.tflite"
LABELS_PATH      = "model/class_labels.json"
MEAN_PATH        = "model/feature_mean.npy"
STD_PATH         = "model/feature_std.npy"
SIGN_DATA_DIR    = "static/sign_data"
CNN_CONFIDENCE   = 0.80   # raise for fewer false positives
TEMPLATE_THRESH  = 1.4    # lower = stricter template match
DEBOUNCE_SECS    = 2.0
DEBUG            = False   # True = print CNN probabilities to terminal
# ────────────────────────────────────────────────────────

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: tensorflow not installed, CNN detection disabled")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        self.mp_draw  = mp.solutions.drawing_utils
        self.current_gesture  = "Waiting..."
        self.last_word        = ""
        self.last_detect_time = 0

        # CNN
        self.cnn_ready  = False
        self.interpreter = None
        self.inp_det    = None
        self.out_det    = None
        self.classes    = []
        self.feat_mean  = None
        self.feat_std   = None
        self._load_cnn()

        # Template matching (custom recorded signs)
        self.known_signs = {}
        self._load_templates()

    # ── CNN ──────────────────────────────────────────────
    def _load_cnn(self):
        if not TF_AVAILABLE:
            return
        if not os.path.exists(CNN_MODEL_PATH):
            print(f"No CNN model at {CNN_MODEL_PATH} — run preprocess_datasets.py + train.py")
            return
        try:
            with open(LABELS_PATH) as f:
                self.classes = json.load(f)["classes"]
            self.feat_mean  = np.load(MEAN_PATH)
            self.feat_std   = np.load(STD_PATH)
            self.interpreter = tf.lite.Interpreter(model_path=CNN_MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.inp_det = self.interpreter.get_input_details()
            self.out_det = self.interpreter.get_output_details()
            self.cnn_ready = True
            print(f"CNN loaded: {len(self.classes)} classes -> {self.classes}")
        except Exception as e:
            print(f"CNN load error: {e}")

    def _landmarks_to_vec(self, hand_lms):
        """Convert MediaPipe hand landmarks to normalised 63-dim vector."""
        coords = np.array([[lm.x, lm.y, lm.z]
                           for lm in hand_lms.landmark], dtype=np.float32)
        coords -= coords[0]
        scale   = np.linalg.norm(coords[9])
        if scale > 0: coords /= scale
        return coords.flatten()

    def _cnn_predict(self, hand_lms):
        """Returns (label, confidence) or (None, 0)."""
        if not self.cnn_ready: return None, 0.0
        vec = self._landmarks_to_vec(hand_lms)
        vec = (vec - self.feat_mean) / self.feat_std
        vec = vec.reshape(1, -1).astype(np.float32)
        self.interpreter.set_tensor(self.inp_det[0]["index"], vec)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.out_det[0]["index"])[0]
        best  = int(np.argmax(probs))
        conf  = float(probs[best])
        if DEBUG:
            top3 = np.argsort(probs)[-3:][::-1]
            for i in top3:
                print(f"  {self.classes[i]:15s} {probs[i]*100:.1f}%")
        return self.classes[best], conf

    # ── Template matching ────────────────────────────────
    def _load_templates(self):
        if not os.path.exists(SIGN_DATA_DIR): return
        print("Loading custom signs...")
        for fn in os.listdir(SIGN_DATA_DIR):
            if not fn.endswith(".json"): continue
            name = fn.split(".")[0].upper()
            try:
                with open(os.path.join(SIGN_DATA_DIR, fn)) as f:
                    data = json.load(f)
                if not data: continue
                # Use middle frame
                frame = data[len(data)//2]
                lms   = frame if isinstance(frame, list) else \
                        (frame.get("right") or frame.get("left"))
                if lms:
                    self.known_signs[name] = self._norm_lm_list(lms)
                    print(f"  Loaded template: {name}")
            except Exception as e:
                print(f"  Error {fn}: {e}")

    def _norm_lm_list(self, lms_list):
        pts = np.array([[l["x"], l["y"]] for l in lms_list], dtype=np.float32)
        pts -= pts[0]
        s    = np.linalg.norm(pts[9])
        if s > 0: pts /= s
        return pts.flatten()

    def _template_match(self, lms_list):
        """Return best matching sign name or None."""
        vec      = self._norm_lm_list(lms_list)
        best_n   = None; best_d = float("inf")
        for name, ref in self.known_signs.items():
            d = np.linalg.norm(vec - ref)
            if d < best_d: best_d = d; best_n = name
        return best_n if best_d < TEMPLATE_THRESH else None

    # ── Basic rules fallback ─────────────────────────────
    def _rule_detect(self, hand_lms):
        wrist     = hand_lms.landmark[0]
        thumb_tip = hand_lms.landmark[4]
        thumb_out = abs(thumb_tip.x - wrist.x) > 0.05
        fingers   = [hand_lms.landmark[t].y < hand_lms.landmark[p].y
                     for t,p in [(8,6),(12,10),(16,14),(20,18)]]
        count     = sum(fingers)
        if count==4 and thumb_out: return "HELLO"
        if count==0 and thumb_out: return "YES"
        if count==1 and fingers[0] and not thumb_out: return "NO"
        return None

    # ── Main frame ───────────────────────────────────────
    def get_frame(self):
        ok, frame = self.video.read()
        if not ok: return None, "Error", None

        frame    = cv2.flip(frame, 1)
        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = self.hands.process(img_rgb)
        detected = ""
        frame_data = {"left": None, "right": None}

        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                label = "right"
                if results.multi_handedness:
                    label = results.multi_handedness[idx].classification[0].label.lower()

                lms_list = [{"x":lm.x,"y":lm.y,"z":lm.z}
                            for lm in hand_lms.landmark]
                frame_data[label] = lms_list
                self.mp_draw.draw_landmarks(
                    frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                # Priority 1: custom templates (phrases)
                tmpl = self._template_match(lms_list)
                if tmpl:
                    detected = tmpl
                    cv2.putText(frame, f"CUSTOM: {tmpl}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    continue

                # Priority 2: CNN (ISL alphabet + LEAP gestures)
                cnn_label, cnn_conf = self._cnn_predict(hand_lms)
                if cnn_label and cnn_conf >= CNN_CONFIDENCE:
                    detected = cnn_label
                    color = (255,200,0)
                    cv2.putText(frame,
                                f"{cnn_label} {cnn_conf*100:.0f}%",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    continue

                # Priority 3: basic rules
                rule = self._rule_detect(hand_lms)
                if rule:
                    detected = rule
                    cv2.putText(frame, f"RULE:{rule}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,255), 2)

        now = time.time()
        if detected:
            # Debounce: don't repeat same word too fast
            if (detected != self.last_word or
                    now - self.last_detect_time > DEBOUNCE_SECS):
                self.current_gesture  = detected
                self.last_word        = detected
                self.last_detect_time = now
        else:
            self.current_gesture = "Waiting..."

        _, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes(), self.current_gesture, frame_data

    def __del__(self):
        if hasattr(self,"video") and self.video.isOpened():
            self.video.release()
