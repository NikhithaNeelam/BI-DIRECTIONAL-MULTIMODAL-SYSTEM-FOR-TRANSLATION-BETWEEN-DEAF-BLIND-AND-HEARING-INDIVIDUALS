"""
preprocess_datasets.py  — COMPLETE WORKING VERSION
ISL classes: 0-9, A-Z, full-stop, space (38 total)
LeapGestRecog: 10 gesture classes
Run: python preprocess_datasets.py --dataset ISL
     python preprocess_datasets.py --dataset ALL  (needs LeapGestRecog too)
"""
import os, cv2, json, argparse
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ── PATHS ─────────────────────────────────────────────────────────────
ISL_TRAIN_DIR = r"datasets\ISL\own-data-preprocessed\own-data-preprocessed\train"
ISL_TEST_DIR  = r"datasets\ISL\own-data-preprocessed\own-data-preprocessed\test"
# Paste kagglehub path here + \leapGestRecog  e.g.:
# r"C:\Users\DELL\.cache\kagglehub\...\leapGestRecog"
LEAP_BASE_DIR = r"PASTE_KAGGLEHUB_PATH_HERE\leapGestRecog"
OUTPUT_DIR    = "datasets"

LEAP_GESTURE_MAP = {
    "01_palm":"PALM","02_l":"L_SHAPE","03_fist":"FIST",
    "04_fist_moved":"FIST_MOVE","05_thumb":"THUMB_UP",
    "06_index":"INDEX_POINT","07_ok":"OK",
    "08_palm_moved":"PALM_MOVE","09_c":"C_SHAPE","10_down":"THUMB_DOWN",
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.3)

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    if len(img.shape)==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks: return None
    lm = result.multi_hand_landmarks[0].landmark
    coords = np.array([[p.x,p.y,p.z] for p in lm], dtype=np.float32)
    coords -= coords[0]
    scale = np.linalg.norm(coords[9])
    if scale > 0: coords /= scale
    return coords.flatten()

def process_flat_dir(root_dir, label_remap=None):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Not found: {root_dir}\nCheck paths at top of preprocess_datasets.py")
    folders = sorted([d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir,d))])
    classes = [label_remap[f] if label_remap and f in label_remap else f for f in folders]
    print(f"  {len(classes)} classes: {classes}")
    X, y = [], []
    for idx,(folder,cls) in enumerate(tqdm(list(zip(folders,classes)),desc="  classes",ncols=70)):
        cls_dir = os.path.join(root_dir, folder)
        images  = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        ok = 0
        for img_file in images:
            vec = extract_landmarks(os.path.join(cls_dir, img_file))
            if vec is not None: X.append(vec); y.append(idx); ok+=1
        print(f"    {cls:20s}: {ok}/{len(images)} ({int(100*ok/len(images)) if images else 0}%)")
    return np.array(X,np.float32), np.array(y,np.int32), classes

def save_labels(classes, source):
    os.makedirs("model", exist_ok=True)
    with open("model/class_labels.json","w") as f:
        json.dump({"classes":list(classes),"num_classes":len(classes),"source":source},f,indent=2)
    print(f"  Saved model/class_labels.json")

def process_isl():
    print("\n"+"="*50+"\nISL Dataset\n"+"="*50)
    X_tr,y_tr,classes = process_flat_dir(ISL_TRAIN_DIR)
    X_te,y_te,_       = process_flat_dir(ISL_TEST_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR,"landmarks_isl.npz")
    np.savez(out,X_train=X_tr,y_train=y_tr,X_test=X_te,y_test=y_te,classes=np.array(classes))
    save_labels(classes,"ISL")
    print(f"\n  Saved {out}  Train:{len(X_tr)} Test:{len(X_te)} Classes:{len(classes)}")
    return X_tr,y_tr,X_te,y_te,classes

def process_leap():
    print("\n"+"="*50+"\nLeapGestRecog\n"+"="*50)
    if not os.path.exists(LEAP_BASE_DIR) or "PASTE" in LEAP_BASE_DIR:
        raise FileNotFoundError(
            "Set LEAP_BASE_DIR at top of file.\n"
            "Run: import kagglehub; p=kagglehub.dataset_download('gti-upm/leapgestrecog'); print(p)\n"
            "Then add \\leapGestRecog at the end.")
    gesture_images={g:[] for g in LEAP_GESTURE_MAP}
    for subj in os.listdir(LEAP_BASE_DIR):
        sdir = os.path.join(LEAP_BASE_DIR,subj)
        if not os.path.isdir(sdir): continue
        for gf in os.listdir(sdir):
            gdir=os.path.join(sdir,gf)
            if not os.path.isdir(gdir): continue
            key=gf.lower()
            if key in gesture_images:
                gesture_images[key].extend(
                    [os.path.join(gdir,f) for f in os.listdir(gdir)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))])
    gestures=sorted(gesture_images.keys())
    classes=[LEAP_GESTURE_MAP[g] for g in gestures]
    print(f"  {len(classes)} classes: {classes}")
    X,y=[],[]
    for idx,(gkey,cls) in enumerate(tqdm(list(zip(gestures,classes)),desc="  gestures",ncols=70)):
        imgs=gesture_images[gkey]; ok=0
        for ip in imgs:
            vec=extract_landmarks(ip)
            if vec is not None: X.append(vec); y.append(idx); ok+=1
        print(f"    {cls:20s}: {ok}/{len(imgs)} ({int(100*ok/len(imgs)) if imgs else 0}%)")
    X,y=np.array(X,np.float32),np.array(y,np.int32)
    idx=np.random.permutation(len(X)); split=int(0.8*len(X))
    X_tr,y_tr=X[idx[:split]],y[idx[:split]]
    X_te,y_te=X[idx[split:]],y[idx[split:]]
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    out=os.path.join(OUTPUT_DIR,"landmarks_leap.npz")
    np.savez(out,X_train=X_tr,y_train=y_tr,X_test=X_te,y_test=y_te,classes=np.array(classes))
    save_labels(classes,"LEAP")
    print(f"\n  Saved {out}  Train:{len(X_tr)} Test:{len(X_te)}")
    return X_tr,y_tr,X_te,y_te,classes

def process_all():
    print("\n"+"="*50+"\nALL = ISL + LeapGestRecog\n"+"="*50)
    print("\n[1/3] ISL...")
    X_isl_tr,y_isl_tr,X_isl_te,y_isl_te,isl_cls = process_isl()
    print("\n[2/3] LEAP...")
    X_lp_tr,y_lp_tr,X_lp_te,y_lp_te,lp_cls = process_leap()
    print("\n[3/3] Merging...")
    merged_classes=list(isl_cls)+list(lp_cls)
    n_isl=len(isl_cls)
    X_tr=np.vstack([X_isl_tr,X_lp_tr]); y_tr=np.concatenate([y_isl_tr,y_lp_tr+n_isl])
    X_te=np.vstack([X_isl_te,X_lp_te]); y_te=np.concatenate([y_isl_te,y_lp_te+n_isl])
    for arr_x,arr_y in [(X_tr,y_tr),(X_te,y_te)]:
        p=np.random.permutation(len(arr_x)); arr_x[:]=arr_x[p]; arr_y[:]=arr_y[p]
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    out=os.path.join(OUTPUT_DIR,"landmarks_all.npz")
    np.savez(out,X_train=X_tr.astype(np.float32),y_train=y_tr.astype(np.int32),
             X_test=X_te.astype(np.float32),y_test=y_te.astype(np.int32),
             classes=np.array(merged_classes))
    save_labels(merged_classes,"ISL+LEAP")
    print(f"\n  Saved {out}")
    print(f"  ISL {len(isl_cls)} + LEAP {len(lp_cls)} = {len(merged_classes)} total classes")
    print(f"  Train:{len(X_tr)}  Test:{len(X_te)}")
    print(f"\n  Next: python train.py --data datasets/landmarks_all.npz")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["ISL","LEAP","ALL"],default="ISL")
    args=parser.parse_args()
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    if args.dataset=="ISL": process_isl()
    elif args.dataset=="LEAP": process_leap()
    else: process_all()
    hands.close()
    print("\nDone!")
