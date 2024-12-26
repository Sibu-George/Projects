import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time


class HandDetector:
    def __init__(self, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def fingersUp(self):
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            for i in range(5):
                if i == 0:  # Thumb special case
                    if hand.landmark[tipIds[i]].x < hand.landmark[tipIds[i] - 1].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if hand.landmark[tipIds[i]].y < hand.landmark[tipIds[i] - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.results.multi_hand_landmarks[0].landmark[p1].x, self.results.multi_hand_landmarks[0].landmark[p1].y
        x2, y2 = self.results.multi_hand_landmarks[0].landmark[p2].x, self.results.multi_hand_landmarks[0].landmark[p2].y
        h, w, c = img.shape
        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        length = np.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


# Initialize webcam and hand detector
wCam, hCam = 1920, 1080
frameR = 100 # Frame reduction
smoothening = 7  # Smoothening factor

pTime, plocX, plocY = 0, 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = HandDetector(maxHands=1)
wScr, hScr = pyautogui.size()

# Main Loop

while True:
        success, img = cap.read()
        if not success:
            print("Error: Camera frame not available.")
            break

        # Process image and find hands
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList:
            x1, y1 = lmList[8][1:]  # Index finger tip
            fingers = detector.fingersUp()

            # Move mode (index finger up, middle finger down)
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                pyautogui.moveTo(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY

            # Click mode (index and middle fingers up)
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, _ = detector.findDistance(8, 12, img)
                if length < 40:
                    pyautogui.click()

        # FPS calculation and display
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3
        )

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break

