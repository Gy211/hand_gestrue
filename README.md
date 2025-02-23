# hand_gestrue
csci218 group assigment
# Hand Gesture Control System

A Python application that uses your webcam to control your computer with hand gestures. You can control both mouse movement and WASD keys.

## Installation Guide

### 1. Python Installation
- Download and install Python 3.x
- Make sure to check "Add Python to PATH" during installation

### 2. Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# For Windows:
.venv\Scripts\activate
# For macOS/Linux:
source .venv/bin/activate
```

### 3. Required Packages
Install all required packages using pip:

```bash
# Core packages
pip install opencv-python     # For webcam and image processing
pip install mediapipe        # For hand tracking
pip install keyboard         # For keyboard control
pip install pyautogui        # For mouse control

# Additional dependencies (if needed)
# For macOS users:
brew install python-tk python-tcl
```


## Quick Start

1. Clone or download this repository
2. Navigate to the project directory
3. Run the program:
   ```bash
   python hand_gesture.py
   ```

## How to Use

### Program Control
- **Make a fist** (close all fingers) to exit the program
- Alternatively, press 'q' key to quit

### Mouse Control
- **Raise ONLY your index finger** to control the mouse
- Move your finger to move the cursor
- Push your finger forward (toward screen) to click
- Push forward twice quickly to double-click

### Keyboard Control
Show your whole hand and:
- Move hand UP → W key (forward)
- Move hand DOWN → S key (backward)
- Move hand LEFT → A key (left)
- Move hand RIGHT → D key (right)
- Raise 2 fingers → Space bar

### Tips
- Keep your hand visible to the camera
- Use in good lighting
- Stay about arm's length from camera
- Keep your movements clear and deliberate
- For double-click, make two quick forward movements within 0.5 seconds
- To close program, make a tight fist with all fingers closed

## Troubleshooting

### Common Issues:
1. **ModuleNotFoundError**:
   ```bash
   # Run these commands one by one:
   pip install opencv-python
   pip install mediapipe
   pip install keyboard
   pip install pyautogui
   ```

2. **Permission Error** (Linux/macOS):
   ```bash
   sudo pip install keyboard
   ```

3. **Webcam Not Found**:
   - Check if webcam is properly connected
   - Try different USB ports
   - Check webcam permissions in system settings

4. **Performance Issues**:
   - Ensure good lighting
   - Reduce background movement
   - Keep appropriate distance from camera

## System Requirements
- Python 3.x
- Webcam
- Operating System: Windows/macOS/Linux
- Minimum 4GB RAM recommended
- Webcam with minimum 720p resolution recommended

