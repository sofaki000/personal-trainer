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


# Function to provide squat recommendations based on pose landmarks
def give_squat_feedback(landmarks):
    recommendations = []

    # Get specific landmarks for squat analysis
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

    # Check squat depth (hip should be below knee for a proper squat)
    if knee.y > hip.y:  # If knee is above or level with hip, squat is shallow
        recommendations.append("Try to lower your hips more during the squat for better depth.")

    # Check knee alignment (knees should not cave inward)
    if knee.x < ankle.x:  # Knees caving inward
        recommendations.append("Make sure your knees are aligned with your toes, not caving inward.")

    # Check back posture (shoulders should be above hips to avoid leaning forward)
    if shoulder.y > hip.y:  # If shoulders are too far forward compared to hips
        recommendations.append("Keep your chest upright and avoid leaning forward.")

    return recommendations


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

        # Make a prediction using the trained model (e.g., squat, hand rise)
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        prediction = model.predict([keypoints])  # Model expects 2D array (samples, features)

        # Display the predicted exercise on the frame
        cv2.putText(frame, f"Predicted Exercise: {prediction[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)

        # If squats are detected, give recommendations
        if prediction[0] == 'squat':
            recommendations = give_squat_feedback(landmarks)
            y_offset = 50  # Starting position for the recommendations text
            for recommendation in recommendations:
                cv2.putText(frame, recommendation, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30  # Move down for the next line

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame with predictions and recommendations
    cv2.imshow("Exercise Classification with Feedback", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
