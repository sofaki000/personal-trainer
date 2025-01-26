import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


# Define function to collect keypoints data
def collect_data_for_exercises():
    keypoints_data = []
    labels = []

    # Create a folder to save data (if it doesn't exist)
    if not os.path.exists('collected_data'):
        os.makedirs('collected_data')


    # 1. Collect data for "Squats"
    print("Please do squats. Perform 5 squats and press 'q' to finish.")
    for i in range(5):  # Collect data for 5 squats
        print(f"Recording Squat {i + 1}")

        # Start video capture
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and detect pose
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Extract pose landmarks
                landmarks = results.pose_landmarks.landmark
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()  # Flatten into 1D array

                # Save keypoints and corresponding label
                keypoints_data.append(keypoints)
                labels.append('squat')

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame
            cv2.imshow('Squats', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # 2. Collect data for "Hand Rises"
    print("Please do hand raises. Perform 5 hand rises and press 'q' to finish.")
    for i in range(5):  # Collect data for 5 hand rises
        print(f"Recording Hand Rise {i + 1}")

        # Start video capture
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and detect pose
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Extract pose landmarks
                landmarks = results.pose_landmarks.landmark
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()  # Flatten into 1D array

                # Save keypoints and corresponding label
                keypoints_data.append(keypoints)
                labels.append('hand_rise')

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame
            cv2.imshow('Hand Rises', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # 3. Save the collected data to a file
    keypoints_data = np.array(keypoints_data)
    labels = np.array(labels)

    # Save keypoints and labels as numpy arrays
    np.save('collected_data/keypoints.npy', keypoints_data)
    np.save('collected_data/labels.npy', labels)

    print("Data collection complete. Data saved as 'keypoints.npy' and 'labels.npy'.")


# Call the function to collect data
collect_data_for_exercises()
