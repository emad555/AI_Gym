import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (e.g., shoulder)
    b = np.array(b)  # Middle point (e.g., elbow)
    c = np.array(c)  # Last point (e.g., wrist)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Function to calculate Euclidean distance
def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# Initialize variables for both arms
left_count = 0
left_direction = 0  # 0: down, 1: up
right_count = 0
right_direction = 0  # 0: down, 1: up
arm_open_threshold = 0.6  # Adjust based on camera position

# Open webcam
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert frame to RGB
        frame = cv2.flip(frame, 1)  # Flip for mirror view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose detection
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            # Get landmarks
            landmarks = result.pose_landmarks.landmark

            # Extract keypoints for wrists
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the horizontal distance between wrists
            horizontal_distance = abs(left_wrist[0] - right_wrist[0])

            # Check if arms are open (both wrists far apart horizontally)
            if horizontal_distance > arm_open_threshold:
                left_count = 0  # Reset the left counter
                right_count = 0  # Reset the right counter
                cv2.putText(frame, "Reset!", (200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Left arm tracking
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            if left_angle > 160 and left_direction == 1:  # Arm fully down
                left_direction = 0
            if left_angle < 60 and left_direction == 0:  # Arm fully up
                left_direction = 1
                left_count += 1

            # Right arm tracking
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            if right_angle > 160 and right_direction == 1:  # Arm fully down
                right_direction = 0
            if right_angle < 60 and right_direction == 0:  # Arm fully up
                right_direction = 1
                right_count += 1

            # Display counts for both arms
            cv2.putText(frame, f'Left Count: {left_count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Right Count: {right_count}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Draw pose landmarks
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display frame
        cv2.imshow('Dumbbell Counter', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
