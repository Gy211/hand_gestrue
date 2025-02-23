import cv2
import mediapipe as mp
import keyboard
import math
import pyautogui
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Load and prepare the dataset
def create_default_dataset():
    """Create a default dataset if Postures.csv is not found"""
    try: 
        print("Creating default gesture dataset...")
        
        # Initialize empty classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train with a simple rule-based approach
        X = np.array([[0.5, 0.5] * 18])  # Default neutral pose
        y = np.array([0])  # Neutral class
        
        clf.fit(X, y)
        
        print("Created default gesture classifier")
        return clf
        
    except Exception as e:
        print("Error creating default dataset:", str(e))
        raise

# Modify prepare_dataset to use default if file not found
def prepare_dataset():
    try:
        # Try to load Postures.csv
        print("Attempting to load Postures.csv...")
        df = pd.read_csv('Postures.csv')
        
        print("Successfully loaded Postures.csv")
        print("Dataset shape:", df.shape)
        print("Dataset columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head())
        
        # Remove rows with missing values (marked as '?')
        df = df.replace('?', np.nan).dropna()
        
        # Extract features (X0 to Z11) and labels (Class)
        X = df.iloc[:, 2:].values.astype(float)  # All columns except Class and User
        y = df['Class'].values.astype(int)
        
        print("Number of samples:", len(X))
        print("Number of features:", X.shape[1])
        print("Unique classes:", np.unique(y))
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Print model accuracy
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print(f"Training accuracy: {train_score:.2f}")
        print(f"Testing accuracy: {test_score:.2f}")
        
        return clf
        
    except FileNotFoundError:
        print("Postures.csv not found, using default gesture set")
        return create_default_dataset()

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen size for mouse control
screen_width, screen_height = pyautogui.size()

pyautogui.FAILSAFE = False

# Train the classifier
gesture_classifier = prepare_dataset()

def calculate_finger_angle(landmark1, landmark2, landmark3):
    # Calculate angle between three points
    radians = math.atan2(landmark3.y - landmark2.y, landmark3.x - landmark2.x) - \
              math.atan2(landmark1.y - landmark2.y, landmark1.x - landmark2.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_finger_raised(hand_landmarks, finger_tip_id, finger_pip_id):
    # Add more strict checking for finger position
    return (hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_pip_id].y 
            and hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[0].y)  # Compare with wrist position

def quit_hand_gestrue(hand_landmarks):
    # Check if all fingers are closed
    fingers_closed = (
        not is_finger_raised(hand_landmarks, 8, 6) and   
        not is_finger_raised(hand_landmarks, 12, 10) and 
        not is_finger_raised(hand_landmarks, 16, 14) and 
        not is_finger_raised(hand_landmarks, 20, 18)     
    )
    return fingers_closed

def space_bar(hand_landmarks):
    # Check if only index and middle fingers are raised
    return (
        is_finger_raised(hand_landmarks, 8, 6) and    # Index raised
        is_finger_raised(hand_landmarks, 12, 10) and  # Middle raised
        not is_finger_raised(hand_landmarks, 16, 14) and  # Ring lowered
        not is_finger_raised(hand_landmarks, 20, 18)      # Pinky lowered
    )

def landmarks_to_feature_vector(hand_landmarks):
    """Convert MediaPipe landmarks to feature vector matching dataset format"""
    # MediaPipe hand landmarks has 21 points, we'll use only x and y coordinates
    # to match the 36 features (18 points * 2 coordinates) expected by the model
    features = []
    for landmark in hand_landmarks.landmark[:18]:  # Use only first 18 landmarks
        features.extend([landmark.x, landmark.y])  # Only use x and y coordinates, skip z
    return np.array(features)

def detect_gesture(hand_landmarks):
    # Get the palm position (using wrist landmark)
    palm_y = hand_landmarks.landmark[0].y
    palm_x = hand_landmarks.landmark[0].x
    
    # Check for single finger click FIRST (before WASD)
    if (is_finger_raised(hand_landmarks, 8, 6) and    # Only index raised
        not is_finger_raised(hand_landmarks, 12, 10) and  # Middle lowered
        not is_finger_raised(hand_landmarks, 16, 14) and  # Ring lowered
        not is_finger_raised(hand_landmarks, 20, 18)):    # Pinky lowered
        
        # Get finger positions
        index_tip_y = hand_landmarks.landmark[8].y
        index_pip_y = hand_landmarks.landmark[6].y
        wrist_y = hand_landmarks.landmark[0].y
        
        # Calculate the distance between tip and PIP joint
        finger_extension = abs(index_tip_y - index_pip_y)
        
        # Check if finger is pointing forward (lower y value means more forward)
        if index_tip_y < wrist_y - 0.2 and finger_extension > 0.1:
            pyautogui.click()
            return 'Click'
        return 'Point'
    
    # Check for Space bar (two fingers raised)
    if (is_finger_raised(hand_landmarks, 8, 6) and    # Index raised
        is_finger_raised(hand_landmarks, 12, 10) and  # Middle raised
        not is_finger_raised(hand_landmarks, 16, 14) and  # Ring lowered
        not is_finger_raised(hand_landmarks, 20, 18)):    # Pinky lowered
        keyboard.press('space')
        return 'Space'
    else:
        keyboard.release('space')
    
    # Release all movement keys before checking new movement
    keyboard.release('w')
    keyboard.release('a')
    keyboard.release('s')
    keyboard.release('d')
    
    # Define center zone where no movement should be detected
    CENTER_Y = 0.5
    CENTER_X = 0.5
    DEAD_ZONE = 0.15  # Increased dead zone in the middle
    
    # If hand is in the dead zone, don't trigger any movement
    if (abs(palm_y - CENTER_Y) < DEAD_ZONE and 
        abs(palm_x - CENTER_X) < DEAD_ZONE):
        return 'None'
    
    # Check vertical movement (W and S)
    if palm_y < 0.3:
        keyboard.press('w')
        return 'Move Forward (W)'
    elif palm_y > 0.7:
        keyboard.press('s')
        return 'Move Backward (S)'
    
    # Check horizontal movement (A and D)
    if palm_x < 0.3:
        keyboard.press('a')
        return 'Move Left (A)'
    elif palm_x > 0.7:
        keyboard.press('d')
        return 'Move Right (D)'
    
    return 'None'

def main():
    # Set window properties
    cv2.namedWindow('Hand Gesture Gaming Control', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Hand Gesture Gaming Control', cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow('Hand Gesture Gaming Control', 800, 600)  # Set a reasonable size

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Detect gesture and get corresponding key
                gesture = detect_gesture(hand_landmarks)
                
                # Display detected gesture
                cv2.putText(image, f'Gesture: {gesture}', (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Add quit instructions to the image
        cv2.putText(image, "Press 'Q' or ESC to quit", (10, image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the image
        cv2.imshow('Hand Gesture Gaming Control', image)
        
        # Check for quit keys with a longer wait time
        key = cv2.waitKey(10) & 0xFF  # Increased wait time slightly
        if key in [ord('q'), ord('Q'), 27]:  # Check for q, Q, or ESC
            print("Quit key pressed:", key)  # Debug print
            break

    # Release all keys before quitting
    keyboard.release('w')
    keyboard.release('a')
    keyboard.release('s')
    keyboard.release('d')
    keyboard.release('space')

    # Clean up
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    
    # Force close all windows
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("Program ended.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        # Release all keys
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        keyboard.release('space')
        # Clean up
        cap.release()
        cv2.destroyAllWindows()