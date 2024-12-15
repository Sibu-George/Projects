import cv2
import numpy as np
from playsound import playsound
import time
import os

cam = cv2.VideoCapture(0)

ret,first_frame=cam.read()
first_frame = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
first_frame = cv2.GaussianBlur(first_frame,(21,21),0)

last_played = 0
cooldown = 3

last_frame_update = time.time()
frame_update_interval = 10

save_photos = True  
output_dir = "motion_photos"
if save_photos and not os.path.exists(output_dir):
    os.makedirs(output_dir)

photo_count = 0 

while True:
    ret, frame = cam.read()
    if not ret:
        break

    
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    gray_frame = cv2.GaussianBlur(gray_frame,(21,21),0)

    delta_frame = cv2.absdiff(first_frame,gray_frame)

    thres_frame = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

    thres_frame = cv2.dilate(thres_frame, None,iterations=2)

    thres_frame = cv2.erode(thres_frame, None, iterations=1)
    
    contours, _ = cv2.findContours(thres_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 1000: 
            continue

        motion_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    if motion_detected and (time.time() - last_played > cooldown):
        print("Motion Detected! Playing Sound!")
        playsound(r"C:\Users\Metatron\Downloads\mixkit-slot-machine-win-alert-1931.wav")
        last_played = time.time()

        if save_photos:
            timestamp = time.strftime("%Y%m%d_%H%M%S")  
            filename = f"{output_dir}/motion_{timestamp}_{photo_count}.jpg"
            cv2.imwrite(filename, frame)  
            photo_count += 1
            print(f"Photo saved: {filename}")

    if time.time() - last_frame_update > frame_update_interval:
        first_frame = gray_frame
        last_frame_update = time.time()
        print("First frame updated.")


    if not motion_detected and time.time() - last_played > cooldown:
        first_frame = gray_frame

    cv2.imshow("Motion Detection",frame)

    if time.time() - last_played > 5:
        first_frame = gray_frame

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()