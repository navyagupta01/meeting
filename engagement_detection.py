from openvino.runtime import Core
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import time
import pandas as pd
from datetime import datetime

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

core = Core()

# Load models
fd = core.read_model("face_detection/face-detection-adas-0001.xml")
em = core.read_model("emotion_detection/emotions-recognition-retail-0003.xml")
gaze = core.read_model("fatique_detection/facial_landmark.xml")
head_pose = core.read_model("head_pose/head_pose.xml")

# Compile models
fd_net = core.compile_model(fd, "CPU")
em_net = core.compile_model(em, "CPU")
gaze_net = core.compile_model(gaze, "CPU")
hp_net = core.compile_model(head_pose, "CPU")

# Outputs
fd_out = fd_net.output(0)
em_out = em_net.output(0)
gaze_out = gaze_net.output(0)
hp_outs = hp_net.outputs

emotion_labels = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# Participant tracking
participants = {}  # {participant_id: {"name": str, "join_time": datetime, "leave_time": datetime, "emotions": [], "fatigue": []}}

def sync_participants():
    """Fetch participants from Firestore"""
    snapshot = db.collection("participants").get()
    for doc in snapshot:
        data = doc.data()
        participants[doc.id] = {
            "name": data.get("name", f"Participant_{doc.id}"),
            "join_time": datetime.fromisoformat(data["join_time"]) if data.get("join_time") else None,
            "leave_time": datetime.fromisoformat(data["leave_time"]) if data.get("leave_time") else None,
            "emotions": data.get("emotions", []),
            "fatigue": data.get("fatigue", [])
        }

def process_frame(frame, participant_id):
    h, w = frame.shape[:2]
    blob = cv2.resize(frame, (672, 384)).transpose((2, 0, 1))[np.newaxis].astype(np.float32)
    detections = fd_net([blob])[fd_out]

    for det in detections[0][0]:
        conf = det[2]
        if conf < 0.6:
            continue

        xmin, ymin, xmax, ymax = map(int, [det[3]*w, det[4]*h, det[5]*w, det[6]*h])
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w - 1, xmax), min(h - 1, ymax)

        face = frame[ymin:ymax, xmin:xmax]
        if face.size == 0:
            continue

        # Emotion Detection
        em_blob = cv2.resize(face, (64, 64)).transpose((2, 0, 1))[np.newaxis].astype(np.float32)
        em_res = em_net([em_blob])[em_out]
        em_label = emotion_labels[np.argmax(em_res)]

        # Head Pose Estimation
        hp_blob = cv2.resize(face, (60, 60)).transpose((2, 0, 1))[np.newaxis].astype(np.float32)
        hp_result = hp_net([hp_blob])
        yaw = hp_result[hp_outs[0]][0][0]
        pitch = hp_result[hp_outs[1]][0][0]
        roll = hp_result[hp_outs[2]][0][0]
        head_pose_angles = np.array([[yaw, pitch, roll]], dtype=np.float32)

        # Eye cropping
        try:
            eye_y = int(face.shape[0] * 0.3)
            left_eye_x = int(face.shape[1] * 0.2)
            right_eye_x = int(face.shape[1] * 0.6)
            eye_h, eye_w = 60, 60

            left_eye = face[eye_y:eye_y+eye_h, left_eye_x:left_eye_x+eye_w]
            right_eye = face[eye_y:eye_y+eye_h, right_eye_x:right_eye_x+eye_w]

            if left_eye.shape[:2] != (eye_h, eye_w) or right_eye.shape[:2] != (eye_h, eye_w):
                fatigue_status = "Unknown"
            else:
                left_blob = left_eye.transpose((2, 0, 1))[np.newaxis].astype(np.float32)
                right_blob = right_eye.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

                gaze_inputs = {
                    "left_eye_image": left_blob,
                    "right_eye_image": right_blob,
                    "head_pose_angles": head_pose_angles
                }

                gaze_vec = gaze_net(gaze_inputs)[gaze_out][0]
                fatigue_status = "Sleepy" if abs(gaze_vec[1]) > 0.15 else "Alert"
        except:
            fatigue_status = "Unknown"

        # Store results
        if participant_id in participants:
            participants[participant_id]["emotions"].append({"time": datetime.now().isoformat(), "emotion": em_label})
            participants[participant_id]["fatigue"].append({"time": datetime.now().isoformat(), "status": fatigue_status})
            db.collection("participants").document(participant_id).update({
                "emotions": participants[participant_id]["emotions"],
                "fatigue": participants[participant_id]["fatigue"]
            })

        return em_label, fatigue_status
    return None, None

def generate_excel():
    data = []
    for pid, info in participants.items():
        data.append({
            "Participant ID": pid,
            "Name": info["name"],
            "Join Time": info["join_time"].isoformat() if info["join_time"] else None,
            "Leave Time": info["leave_time"].isoformat() if info["leave_time"] else None,
            "Emotions": "; ".join([f"{e['time']}: {e['emotion']}" for e in info["emotions"]]),
            "Fatigue": "; ".join([f"{f['time']}: {f['status']}" for f in info["fatigue"]])
        })
    df = pd.DataFrame(data)
    df.to_excel("attendance.xlsx", index=False)

# Main loop
cap = cv2.VideoCapture(0)  # Teacher's webcam
while True:
    ret, frame = cap.read()
    if ret:
        sync_participants()  # Update participants from Firestore
        for pid in participants:
            em_label, fatigue_status = process_frame(frame, pid)
            if em_label and fatigue_status:
                label = f"{participants[pid]['name']}: {em_label}, {fatigue_status}"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Engagement Tracker", frame)
    if cv2.waitKey(1) == 27:
        generate_excel()
        break
cap.release()
cv2.destroyAllWindows()