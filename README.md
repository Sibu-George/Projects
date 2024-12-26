```markdown
# Motion Detection with OpenCV

This project is a Python script that uses OpenCV to detect motion using a webcam. When motion is detected, the script plays a sound and saves a photo of the detected frame.

## Features
- Motion detection using webcam
- Plays a sound when motion is detected
- Saves photos of frames where motion is detected
- Periodically updates the initial frame to avoid false positives

## Requirements
- Python 3.x
- OpenCV
- Numpy
- playsound

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Sibu-George/motion-detection-opencv.git
```

2. Navigate to the project directory:
```bash
cd motion-detection-opencv
```

3. Install the required dependencies:
```bash
pip install opencv-python numpy playsound
```

## Usage
1. Make sure your webcam is connected.

2. Specify the path to your sound file in the script:
```python
playsound(r"Enter the path of the Sound")
```

3. Run the script:
```bash
python motion_detection.py
```

4. The script will display a window showing the webcam feed. Motion will be highlighted with a green rectangle.

5. Photos of detected motion will be saved in the `motion_photos` directory.

## Customization
- **Cooldown Period:** Change the `cooldown` variable to adjust the time between sound plays.
- **Frame Update Interval:** Change the `frame_update_interval` variable to adjust how often the initial frame is updated.
- **Save Photos:** Set the `save_photos` variable to `False` if you don't want to save photos of detected motion.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- OpenCV library
- playsound library

