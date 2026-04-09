import cv2
import mediapipe as mp
import json
import os

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

def record_sign(sign_name):
    cap = cv2.VideoCapture(0)
    print(f"Press 'R' to start recording '{sign_name}'. Press 'Q' to quit.")
    
    recording = False
    frames_data = [] # List to store coordinates for every frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw for visual feedback
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if recording:
                    # Save landmarks as a simple list of dicts
                    frame_landmarks = []
                    for lm in hand_landmarks.landmark:
                        frame_landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                    frames_data.append(frame_landmarks)

        # UI Text
        status = "RECORDING..." if recording else "Press 'R' to Record"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Motion Capture', frame)
        
        key = cv2.waitKey(1)
        if key == ord('r'):
            if not recording:
                print("Recording started...")
                recording = True
                frames_data = []
            else:
                print("Recording stopped. Saving...")
                recording = False
                # Save to JSON
                if not os.path.exists('static/sign_data'):
                    os.makedirs('static/sign_data')
                
                with open(f'static/sign_data/{sign_name}.json', 'w') as f:
                    json.dump(frames_data, f)
                print(f"Saved {sign_name}.json")
                break
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Run this to record "HELLO"
if __name__ == "__main__":
    name = input("Enter sign name to record (e.g., HELLO): ").upper()
    record_sign(name)