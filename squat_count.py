import cv2
import mediapipe as mp
import numpy as np
import joblib

# Initialize MediaPipe Pose for pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained model
model = joblib.load('exercise_classifier.pkl')


# Function to check if squat is being performed based on keypoints
def count_squats(landmarks, prev_state, squat_count):
    squat_detected = prev_state

    # Get specific landmarks for squat analysis
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

    # Check if squat is being performed based on knee and hip positions
    if knee.y > hip.y and squat_detected == 'up':  # If knee is lower than hip, person is going down
        squat_detected = 'down'

    if knee.y < hip.y and squat_detected == 'down':  # If knee is higher than hip, squat is completed (standing up)
        squat_detected = 'up'
        squat_count += 1  # Increment squat count when the squat motion completes

    return squat_detected, squat_count


# Open webcam (0 is default camera)
cap = cv2.VideoCapture(0)

# Initial state (not doing squats yet)
previous_state = 'up'
squat_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (as MediaPipe works with RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect pose
    results = pose.process(rgb_frame)

    # If pose landmarks are detected
    if results.pose_landmarks:
        # Extract the landmarks (keypoints)
        landmarks = results.pose_landmarks.landmark

        # Make a prediction using the trained model (e.g., squat, hand rise)
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        prediction = model.predict([keypoints])  # Model expects 2D array (samples, features)

        # Display the predicted exercise on the frame
        cv2.putText(frame, f"Predicted Exercise: {prediction[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)

        # If squats are detected, count the squats
        if prediction[0] == 'squat':
            previous_state, squat_count = count_squats(landmarks, previous_state, squat_count)

        # Display the squat count continuously on the frame
        cv2.putText(frame, f"Squats Count: {squat_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame with predictions and squat count
    cv2.imshow("Squat Counter with Feedback", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
