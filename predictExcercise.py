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

# Open webcam (0 is default camera)
cap = cv2.VideoCapture(0)

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
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()  # Flatten into 1D array

        # Make a prediction using the trained model
        prediction = model.predict([keypoints])  # Model expects 2D array (samples, features)

        # Display the predicted exercise on the frame
        cv2.putText(frame, f"Predicted Exercise: {prediction[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame with predictions
    cv2.imshow("Exercise Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
