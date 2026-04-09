"""
app.py — COMPLETE FIXED VERSION
Fixes: Ollama HTTP, avatar path (js/+sign_data/), TTS audio,
       speech recognition, auto-open browser, /ollama_status route
"""
from flask import Flask,render_template,Response,request,jsonify
from camera import VideoCamera
from avatar_generator import AvatarRenderer
from ai_engine import refine_sentence,translate_text,is_ollama_running
import time,cv2,os,json,threading,webbrowser,re

app = Flask(__name__)

current_words=[]; last_detected_time=0; avatar=AvatarRenderer()
is_recording=False; recording_name=""; recorded_frames=[]
available_classes=[]

if os.path.exists("model/class_labels.json"):
    with open("model/class_labels.json") as f:
        available_classes=json.load(f).get("classes",[])
    print(f"CNN: {len(available_classes)} classes loaded")
else:
    print("No CNN model — run preprocess_datasets.py + train.py")

@app.route("/")
def index(): return render_template("index.html")

@app.route("/get_classes")
def get_classes(): return jsonify(classes=available_classes)

@app.route("/ollama_status")
def ollama_status(): return jsonify(online=is_ollama_running())

# ── Sign → Text ──────────────────────────────────────────
def gen_camera(camera):
    global current_words,last_detected_time,is_recording,recorded_frames
    while True:
        frame,gesture,landmarks=camera.get_frame()
        if frame:
            if is_recording:
                if landmarks: recorded_frames.append(landmarks)
            elif gesture not in ("Waiting...","Error"):
                if time.time()-last_detected_time>2.0:
                    current_words.append(gesture)
                    last_detected_time=time.time()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+frame+b"\r\n\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(gen_camera(VideoCamera()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_words")
def get_words(): return jsonify(words=current_words)

@app.route("/clear_words",methods=["POST"])
def clear_words():
    global current_words; current_words=[]; return jsonify(status="cleared")

@app.route("/process_sentence",methods=["POST"])
def process_sentence():
    data=request.json
    raw=" ".join(data.get("words",[])).strip()
    lang=data.get("language","English")
    if not raw: return jsonify(result="")
    return jsonify(result=refine_sentence(raw,lang))

@app.route("/start_recording",methods=["POST"])
def start_recording():
    global is_recording,recording_name,recorded_frames
    recording_name=request.json.get("sign_name","UNKNOWN").upper()
    recorded_frames=[]; is_recording=True
    print(f"Recording: {recording_name}")
    return jsonify(status="started",name=recording_name)

@app.route("/stop_recording",methods=["POST"])
def stop_recording():
    global is_recording,recording_name,recorded_frames
    is_recording=False; count=len(recorded_frames)
    if count==0: return jsonify(status="error",message="No hand data captured!")
    os.makedirs("static/sign_data",exist_ok=True)
    fn=f"static/sign_data/{recording_name}.json"
    try:
        with open(fn,"w") as f: json.dump(recorded_frames,f)
        print(f"Saved {count} frames -> {fn}")
        return jsonify(status="saved",count=count,name=recording_name)
    except Exception as e:
        return jsonify(status="error",message=str(e))

# ── Text/Audio → Sign ────────────────────────────────────
def find_sign_file(name):
    """Search js/ and sign_data/ for a sign JSON. Returns clean name or None."""
    spaced=name.strip().upper().replace("_"," ")
    for folder in ("static/js","static/sign_data"):
        for n in (spaced, spaced.replace(" ","_")):
            if os.path.exists(os.path.join(folder,f"{n}.json")):
                return spaced
    return None

@app.route("/text_to_sign",methods=["POST"])
def text_to_sign_route():
    text=request.json.get("text","").strip()
    if not text: return jsonify(glosses="")

    # 1. Exact match
    m=find_sign_file(text)
    if m: return jsonify(glosses=m.replace(" ","_"))

    # 2. AI / rule-based gloss breakdown
    glosses_str=translate_text(text,"English")
    words=re.findall(r'\b[A-Za-z0-9\-]+\b',glosses_str)
    result=" ".join(w.upper() for w in words)
    return jsonify(glosses=result)

@app.route("/trigger_avatar",methods=["POST"])
def trigger_avatar():
    word=request.json.get("word","").upper().strip().replace("_"," ")
    ok=avatar.load_sequence(word)
    if ok:
        dur=len(avatar.current_animation)*0.05
        return jsonify(status="playing",duration=round(dur,2))
    return jsonify(status="error",message=f"'{word}' not found")

def gen_avatar():
    while True:
        frame=avatar.generate_frame()
        _,buf=cv2.imencode(".jpg",frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n")

@app.route("/avatar_feed")
def avatar_feed():
    return Response(gen_avatar(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Start ────────────────────────────────────────────────
def open_browser():
    time.sleep(1.5); webbrowser.open("http://localhost:5000")

if __name__=="__main__":
    print("\n"+"="*50)
    print("  NexaComm — Bi-Directional Sign Language System")
    print("="*50)
    print(f"  Ollama : {'ONLINE' if is_ollama_running() else 'OFFLINE (rule-based active)'}")
    print(f"  CNN    : {'Loaded ('+str(len(available_classes))+' classes)' if available_classes else 'Not trained'}")
    print("="*50+"\n")
    threading.Thread(target=open_browser,daemon=True).start()
    app.run(debug=False,threaded=True,port=5000)
