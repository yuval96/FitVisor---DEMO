import cv2
import numpy as np
import mediapipe as mp
from utils.angles import calculate_angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

IDX = {
    "L_HIP": mp_pose.PoseLandmark.LEFT_HIP.value,
    "R_HIP": mp_pose.PoseLandmark.RIGHT_HIP.value,
    "L_KNEE": mp_pose.PoseLandmark.LEFT_KNEE.value,
    "R_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE.value,
    "L_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE.value,
    "R_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    "L_HEEL": mp_pose.PoseLandmark.LEFT_HEEL.value,
    "R_HEEL": mp_pose.PoseLandmark.RIGHT_HEEL.value,
    "L_FOOT_INDEX": mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
    "R_FOOT_INDEX": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
    "L_SH": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "R_SH": mp_pose.PoseLandmark.RIGHT_SHOULDER.value
}

SQUAT_DOWN_ANGLE = 100
SQUAT_UP_ANGLE = 160
TORSO_OK_ANGLE = 35
KNEE_TRACK_THRESHOLD = 0.5 # knee over foot alignment threshold


def get_joint_xy(lm, idx, w, h):
    p = lm[idx]
    return (p.x * w, p.y * h)


def compute_torso_angle(lm, w, h):
    mid_sh = (
        (lm[IDX["L_SH"]].x + lm[IDX["R_SH"]].x) * 0.5 * w,
        (lm[IDX["L_SH"]].y + lm[IDX["R_SH"]].y) * 0.5 * h,
    )
    mid_hip = (
        (lm[IDX["L_HIP"]].x + lm[IDX["R_HIP"]].x) * 0.5 * w,
        (lm[IDX["L_HIP"]].y + lm[IDX["R_HIP"]].y) * 0.5 * h,
    )

    v = np.array([mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1]], dtype=float)
    vertical = np.array([0.0, -1.0], dtype=float)

    denom = (np.linalg.norm(v) * np.linalg.norm(vertical)) + 1e-9
    cos_theta = np.dot(v, vertical) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    return angle_deg, mid_sh, mid_hip

# Check knee over foot alignment
def get_closer_leg(lm):
    left_knee_z = lm[IDX["L_KNEE"]].z
    right_knee_z = lm[IDX["R_KNEE"]].z

    # more negative z = closer to camera
    return "left" if left_knee_z < right_knee_z else "right"

def knee_over_foot(lm, side="left", tolerance=0.04):
    if side == "left":
        knee = lm[IDX["L_KNEE"]]
        foot = lm[IDX["L_FOOT_INDEX"]]
        return knee.x < foot.x
    else:
        knee = lm[IDX["R_KNEE"]]
        foot = lm[IDX["R_FOOT_INDEX"]]
        return knee.x > foot.x
    
# Status of knee tracking
def knee_tracking_status(sideBad):
    ok = not sideBad
    return ("OK" if ok else "MISALIGNED"), ((0, 255, 0) if ok else (0, 0, 255))
###


def update_rep_count(knee_angle, stage, count):
    if knee_angle < SQUAT_DOWN_ANGLE and stage == "up":
        stage = "down"
    elif knee_angle > SQUAT_UP_ANGLE and stage == "down":
        stage = "up"
        count += 1
    return stage, count


def torso_status(torso_angle):
    ok = torso_angle < TORSO_OK_ANGLE
    return ("Torso OK" if ok else "Torso LEANING"), ((0, 255, 0) if ok else (0, 0, 255))


def draw_overlay(frame, knee_angle, torso_angle, reps, mid_sh, mid_hip, color):
    cv2.putText(frame, f"Angle: {int(knee_angle)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {reps}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame, f"Torso: {int(torso_angle)} deg", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.line(frame,
             (int(mid_hip[0]), int(mid_hip[1])),
             (int(mid_sh[0]), int(mid_sh[1])),
             color, 3)


def process_frame(frame, pose, stage, reps, mirror=False, draw_reps=True):
    if mirror:
        frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return frame, stage, reps, None

    lm = results.pose_landmarks.landmark

    hip = get_joint_xy(lm, IDX["L_HIP"], w, h)
    knee = get_joint_xy(lm, IDX["L_KNEE"], w, h)
    ankle = get_joint_xy(lm, IDX["L_ANKLE"], w, h)

    knee_angle = calculate_angle(hip, knee, ankle)
    torso_angle, mid_sh, mid_hip = compute_torso_angle(lm, w, h)

    ## knee tracking
    closer_leg = get_closer_leg(lm)

    if closer_leg == "left":
        left_bad = knee_over_foot(lm, "left")
        left_status, left_color = knee_tracking_status(left_bad)

        right_status, right_color = "N/A", (180, 180, 180)
    else:
        right_bad = knee_over_foot(lm, "right")
        right_status, right_color = knee_tracking_status(right_bad)

        left_status, left_color = "N/A", (180, 180, 180)

    stage, reps = update_rep_count(knee_angle, stage, reps)
    _, color = torso_status(torso_angle)

    if draw_reps:
        draw_overlay(frame, knee_angle, torso_angle, reps, mid_sh, mid_hip, color)

        # Draw knee tracking status
        cv2.putText(frame, f"L Knee: {left_status}", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, left_color, 2)

        cv2.putText(frame, f"R Knee: {right_status}", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, right_color, 2)
        ##

    else:
        cv2.putText(frame, f"Angle: {int(knee_angle)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Torso: {int(torso_angle)} deg", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.line(frame,
                 (int(mid_hip[0]), int(mid_hip[1])),
                 (int(mid_sh[0]), int(mid_sh[1])),
                 color, 3)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame, stage, reps, results


def run_squat_live():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not available")
        return

    stage, reps = "up", 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, stage, reps, _ = process_frame(frame, pose, stage, reps, mirror=False, draw_reps=True)
            cv2.imshow("FitVisor - Live Squats", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def run_squat_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video not found")
        return

    stage, reps = "up", 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, stage, reps, _ = process_frame(frame, pose, stage, reps, mirror=False, draw_reps=True)
            cv2.imshow("FitVisor - Video Squats", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def run_squat_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return

    with mp_pose.Pose(static_image_mode=True) as pose:
        stage, reps = "up", 0
        image, stage, reps, results = process_frame(image, pose, stage, reps, mirror=False, draw_reps=False)

        if results is None or not results.pose_landmarks:
            print("No person detected.")
            return

        cv2.imshow("FitVisor - Image Squat Analysis", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

