import cv2
import numpy as np
import mediapipe as mp
from utils.angles import calculate_angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==========================
# PUSH-UP DETECTION SETTINGS
# ==========================
PUSHUP_DOWN_ANGLE = 90
PUSHUP_UP_ANGLE = 160

# Plank check (kept same logic you had: hip->shoulder vs vertical)
PLANK_OK_ANGLE = 20

# Arm-to-torso check at shoulder: elbow-shoulder-hip should be ~45 deg
ARM_TORSO_TARGET = 60
ARM_TORSO_TOL = 30  # OK range: 30..60

# UI settings
SHOW_DEBUG_TEXT = True  # set False if you want only reps + status
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ==========================
# GEOMETRY HELPERS
# ==========================
def get_joint_xy(lm, idx, w, h):
    p = lm[idx]
    return (p.x * w, p.y * h)


def choose_side(lm):
    """Pick left/right side by highest landmark visibility (shoulder + elbow + wrist)."""
    left = (
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
    )
    right = (
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
    )

    lv = lm[left[0]].visibility + lm[left[1]].visibility + lm[left[2]].visibility
    rv = lm[right[0]].visibility + lm[right[1]].visibility + lm[right[2]].visibility
    return left if lv >= rv else right


def compute_plank_angle(lm, w, h, use_left=True):
    """Angle between hip->shoulder vector and vertical axis (same idea as your torso angle)."""
    sh_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value if use_left else mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value if use_left else mp_pose.PoseLandmark.RIGHT_HIP.value

    sh = (lm[sh_idx].x * w, lm[sh_idx].y * h)
    hip = (lm[hip_idx].x * w, lm[hip_idx].y * h)

    v = np.array([sh[0] - hip[0], sh[1] - hip[1]], dtype=float)
    vertical = np.array([0.0, -1.0], dtype=float)

    denom = (np.linalg.norm(v) * np.linalg.norm(vertical)) + 1e-9
    cos_theta = np.dot(v, vertical) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    return angle_deg, sh, hip


# ==========================
# LOGIC (COUNTING + CHECKS)
# ==========================
def update_pushup_count(elbow_angle, stage, count):
    if elbow_angle < PUSHUP_DOWN_ANGLE and stage == "up":
        stage = "down"
    elif elbow_angle > PUSHUP_UP_ANGLE and stage == "down":
        stage = "up"
        count += 1
    return stage, count


def plank_ok(plank_angle):
    return plank_angle < PLANK_OK_ANGLE


def arm_torso_ok(arm_torso_angle):
    return abs(arm_torso_angle - ARM_TORSO_TARGET) <= ARM_TORSO_TOL


# ==========================
# UI (CLEAN OVERLAY)
# ==========================
def draw_status_panel(frame, reps, stage, plank_ok_flag, arm_ok_flag, elbow_angle=None, plank_angle=None, arm_angle=None):
    # background panel
    cv2.rectangle(frame, (10, 10), (360, 165 if SHOW_DEBUG_TEXT else 115), (0, 0, 0), -1)

    # reps (big, clear)
    cv2.putText(frame, f"REPS: {reps}", (20, 60), FONT, 1.6, (255, 255, 255), 3)

    # stage
    cv2.putText(frame, f"STAGE: {stage.upper()}", (20, 100), FONT, 0.9, (220, 220, 220), 2)

    # form statuses (only 2 lines, color-coded)
    plank_color = (0, 255, 0) if plank_ok_flag else (0, 0, 255)
    arm_color = (0, 255, 0) if arm_ok_flag else (0, 0, 255)

    cv2.putText(frame, "PLANK", (200, 95), FONT, 0.85, (255, 255, 255), 2)
    cv2.putText(frame, "OK" if plank_ok_flag else "BAD", (290, 95), FONT, 0.85, plank_color, 2)

    cv2.putText(frame, "ARM", (200, 130), FONT, 0.85, (255, 255, 255), 2)
    cv2.putText(frame, "OK" if arm_ok_flag else "BAD", (290, 130), FONT, 0.85, arm_color, 2)

    # optional debug angles (small, tucked away)
    if SHOW_DEBUG_TEXT and elbow_angle is not None and plank_angle is not None and arm_angle is not None:
        cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (20, 150), FONT, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"Plank: {int(plank_angle)}", (140, 150), FONT, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"Arm: {int(arm_angle)}", (265, 150), FONT, 0.7, (200, 200, 200), 2)


def draw_plank_line(frame, hip_pt, sh_pt, ok_flag):
    color = (0, 255, 0) if ok_flag else (0, 0, 255)
    cv2.line(frame, (int(hip_pt[0]), int(hip_pt[1])), (int(sh_pt[0]), int(sh_pt[1])), color, 3)


# ==========================
# MAIN FRAME PROCESSOR
# ==========================
def process_frame_pushups(frame, pose, stage, reps, mirror=False, draw_ui=True):
    if mirror:
        frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return frame, stage, reps, None

    lm = results.pose_landmarks.landmark

    sh_idx, el_idx, wr_idx, hip_idx = choose_side(lm)
    use_left = (sh_idx == mp_pose.PoseLandmark.LEFT_SHOULDER.value)

    shoulder = get_joint_xy(lm, sh_idx, w, h)
    elbow = get_joint_xy(lm, el_idx, w, h)
    wrist = get_joint_xy(lm, wr_idx, w, h)
    hip = get_joint_xy(lm, hip_idx, w, h)

    # Angles (logic kept)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    plank_angle, sh_pt, hip_pt = compute_plank_angle(lm, w, h, use_left=use_left)
    arm_torso_angle = calculate_angle(elbow, shoulder, hip)

    # Counting
    stage, reps = update_pushup_count(elbow_angle, stage, reps)

    # Form checks
    plank_ok_flag = plank_ok(plank_angle)
    arm_ok_flag = arm_torso_ok(arm_torso_angle)

    # Draw
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    draw_plank_line(frame, hip_pt, sh_pt, plank_ok_flag)

    if draw_ui:
        draw_status_panel(
            frame,
            reps=reps,
            stage=stage,
            plank_ok_flag=plank_ok_flag,
            arm_ok_flag=arm_ok_flag,
            elbow_angle=elbow_angle,
            plank_angle=plank_angle,
            arm_angle=arm_torso_angle,
        )

    return frame, stage, reps, results


# ==========================
# RUNNERS (LIVE / VIDEO / IMAGE)
# ==========================
def run_pushups_live():
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

            frame, stage, reps, _ = process_frame_pushups(frame, pose, stage, reps, mirror=True, draw_ui=True)
            cv2.imshow("FitVisor - Live Pushups", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def run_pushups_on_video(video_path):
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

            frame, stage, reps, _ = process_frame_pushups(frame, pose, stage, reps, mirror=False, draw_ui=True)
            cv2.imshow("FitVisor - Video Pushups", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def run_pushups_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return

    with mp_pose.Pose(static_image_mode=True) as pose:
        stage, reps = "up", 0
        image, stage, reps, results = process_frame_pushups(image, pose, stage, reps, mirror=False, draw_ui=True)

        if results is None or not results.pose_landmarks:
            print("No person detected.")
            return

        cv2.imshow("FitVisor - Image Pushup Analysis", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
