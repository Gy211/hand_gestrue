import cv2
import mediapipe as mp
import keyboard
import math
import pyautogui
import time

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen size for mouse control
screen_width, screen_height = pyautogui.size()

pyautogui.FAILSAFE = False

last_click_time = 0
# Maximum time between clicks for a double click (in seconds)
DOUBLE_CLICK_INTERVAL = 0.5
def calculate_finger_angle(landmark1, landmark2, landmark3):
    # Calculate angle between three points
    radians = math.atan2(landmark3.y - landmark2.y, landmark3.x - landmark2.x) - \
              math.atan2(landmark1.y - landmark2.y, landmark1.x - landmark2.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_finger_raised(hand_landmarks, finger_tip_id, finger_pip_id):
    return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_pip_id].y

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

def detect_gesture(hand_landmarks):
    # Check for quit gesture first
    if quit_hand_gestrue(hand_landmarks):
        # Release all keys before closing
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        keyboard.release('space')
        return 'Fist_Close'
    
    # Check for space bar gesture
    if space_bar(hand_landmarks):
        keyboard.press('space')
        # Movement control
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        return 'Space'
    
    # Get landmarks for gesture detection
    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    
    # Check if only finger is raised
    index_raised = is_finger_raised(hand_landmarks, 8, 6)  
    middle_raised = is_finger_raised(hand_landmarks, 12, 10)  
    ring_raised = is_finger_raised(hand_landmarks, 16, 14) 
    pinky_raised = is_finger_raised(hand_landmarks, 20, 18)
    
    if index_raised and not (middle_raised or ring_raised or pinky_raised):
        # Mouse control mode
        # Convert hand coordinates to screen coordinates
        x = min(max(0, int(index_tip.x * screen_width)), screen_width - 1)
        y = min(max(0, int(index_tip.y * screen_height)), screen_height - 1)
        
        # Move mouse cursor
        pyautogui.moveTo(x, y)
        
        # Click detection with double click support
        global last_click_time
        # Finger moved forward
        if index_tip.z < -0.1:
            current_time = time.time()
            if current_time - last_click_time < DOUBLE_CLICK_INTERVAL:
                pyautogui.doubleClick()
                # Reset to prevent triple clicks
                last_click_time = 0
                return 'Double Click'
            else:
                pyautogui.click()
                last_click_time = current_time
                return 'Mouse Click'
        return 'Mouse Control'
    
    # Calculate relative positions for WASD control
    hand_y = middle_tip.y
    hand_x = wrist.x
    
    # Define thresholds
    vertical_threshold = 0.3
    horizontal_threshold = 0.4
    
    # WASD control logic
    if hand_y < 0.3:  
        # Hand up - W
        keyboard.press('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        return 'W'
    elif hand_y > 0.7:  
        # Hand down - S
        keyboard.press('s')
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('d')
        return 'S'
    elif hand_x < horizontal_threshold:  
        # Hand left - A
        keyboard.press('a')
        keyboard.release('w')
        keyboard.release('s')
        keyboard.release('d')
        return 'A'
    elif hand_x > (1 - horizontal_threshold):  
        # Hand right - D
        keyboard.press('d')
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        return 'D'
    else:
        # Release all keys if no gesture is detected
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        return 'None'

def main():
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
                
                # Check if gesture is detected
                if gesture == 'Fist_Close':
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                # Display detected gesture
                cv2.putText(image, f'Gesture: {gesture}', (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Hand Gesture Gaming Control', image)
        
        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()