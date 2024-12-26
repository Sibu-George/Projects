```markdown
# Hand Gesture Control with MediaPipe and OpenCV

This project is a Python script that uses MediaPipe and OpenCV to detect hand gestures via webcam. These gestures can control the mouse cursor, enabling movement and clicking based on specific hand poses.

## Features
- Hand detection using MediaPipe
- Cursor movement based on index finger position
- Click action triggered by specific finger configurations
- Frame reduction and smoothening for accurate control

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- Numpy
- pyautogui

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Sibu-George/hand-gesture-control.git
```

2. Navigate to the project directory:
```bash
cd hand-gesture-control
```

3. Install the required dependencies:
```bash
pip install opencv-python mediapipe numpy pyautogui
```

## Usage
1. Ensure your webcam is connected.

2. Run the script:
```bash
python hand_gesture_control.py
```

3. The script will display a window showing the webcam feed. Move your index finger to control the mouse cursor and use specific finger configurations to perform clicks.

## Customization
- **Frame Reduction:** Adjust the `frameR` variable to control the inactive borders.
- **Smoothening Factor:** Modify the `smoothening` variable for smoother cursor movement.
- **Detection Confidence:** Change `detectionCon` and `trackCon` in the `HandDetector` class for varying detection and tracking confidence.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- MediaPipe library
- OpenCV library
