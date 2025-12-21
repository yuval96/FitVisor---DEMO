import cv2
import numpy as np
import mediapipe as mp
from utils.angles import calculate_angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ==========================
# PUSH-UP DETECTION HELPERS
# ==========================

# Recommended starting thresholds (tune per camera angle)
PUSHUP_DOWN_ANGLE = 90     # elbow angle at bottom
PUSHUP_UP_ANGLE = 160      # elbow angle at top (arms extended)
PLANK_OK_ANGLE = 20        # torso lean limit (smaller is "more vertical" per current torso_angle def)

def choose_side(lm):
    """
    Choose the side (left/right) with higher landmark visibility for elbow-angle measurement.
    Returns a tuple of indices: (shoulder_idx, elbow_idx, wrist_idx, hip_idx)
    """
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

    lv = (
        lm[left[0]].visibility + lm[left[1]].visibility + lm[left[2]].visibility
    )
    rv = (
        lm[right[0]].visibility + lm[right[1]].visibility + lm[right[2]].visibility
    )

    return left if lv >= rv else right


def compute_plank_angle(lm, w, h, use_left=True):
    """
    Plank indicator: angle between hip->shoulder vector and vertical axis.
    For pushups, you may prefer "horizontal-to-ground" instead,
    but this keeps it consistent with your torso_angle implementation.
    """
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


def update_pushup_count(elbow_angle, stage, count):
    """
    Stage logic:
    - "up" -> "down" when elbow angle gets small (bottom)
    - "down" -> "up" when elbow angle gets big again (top), count +1
    """
    if elbow_angle < PUSHUP_DOWN_ANGLE and stage == "up":
        stage = "down"
    elif elbow_angle > PUSHUP_UP_ANGLE and stage == "down":
        stage = "up"
        count += 1
    return stage, count


def plank_status(plank_angle):
    ok = plank_angle < PLANK_OK_ANGLE
    return ("Plank OK" if ok else "Plank BROKEN"), ((0, 255, 0) if ok else (0, 0, 255))


def draw_pushup_overlay(frame, elbow_angle, plank_angle, reps, sh, hip, color):
    cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {reps}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame, f"Plank: {int(plank_angle)} deg", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.line(frame,
             (int(hip[0]), int(hip[1])),
             (int(sh[0]), int(sh[1])),
             color, 3)


def process_frame_pushups(frame, pose, stage, reps, mirror=False, draw_reps=True):
    """
    Similar to your process_frame() but tailored to pushups:
    - Measures elbow angle (shoulder-elbow-wrist) on best-visible side
    - Measures a "plank" torso angle (hip->shoulder vs vertical) as a form check
    """
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

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    plank_angle, sh_pt, hip_pt = compute_plank_angle(lm, w, h, use_left=use_left)
    stage, reps = update_pushup_count(elbow_angle, stage, reps)

    _, color = plank_status(plank_angle)

    if draw_reps:
        draw_pushup_overlay(frame, elbow_angle, plank_angle, reps, sh_pt, hip_pt, color)
    else:
        cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Plank: {int(plank_angle)} deg", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.line(frame,
                 (int(hip_pt[0]), int(hip_pt[1])),
                 (int(sh_pt[0]), int(sh_pt[1])),
                 color, 3)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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

            frame, stage, reps, _ = process_frame_pushups(frame, pose, stage, reps, mirror=True, draw_reps=True)
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

            frame, stage, reps, _ = process_frame_pushups(frame, pose, stage, reps, mirror=False, draw_reps=True)
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
        image, stage, reps, results = process_frame_pushups(image, pose, stage, reps, mirror=False, draw_reps=False)

        if results is None or not results.pose_landmarks:
            print("No person detected.")
            return

        cv2.imshow("FitVisor - Image Pushup Analysis", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
