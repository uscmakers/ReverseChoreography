import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import time
import timeit
import csv

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
count = 0
alldata = []
fps_time = 0

pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
              'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
              'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

pose_tangan = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

pose_tangan_2 = ['WRIST2', 'THUMB_CPC2', 'THUMB_MCP2', 'THUMB_IP2', 'THUMB_TIP2', 'INDEX_FINGER_MCP2', 'INDEX_FINGER_PIP2', 'INDEX_FINGER_DIP2', 'INDEX_FINGER_TIP2', 'MIDDLE_FINGER_MCP2',
               'MIDDLE_FINGER_PIP2', 'MIDDLE_FINGER_DIP2', 'MIDDLE_FINGER_TIP2', 'RING_FINGER_PIP2', 'RING_FINGER_DIP2', 'RING_FINGER_TIP2',
               'RING_FINGER_MCP2', 'PINKY_MCP2', 'PINKY_PIP2', 'PINKY_DIP2', 'PINKY_TIP2']
               
cap = cv2.VideoCapture(0)
suc,frame_video = cap.read()
# Timestamp
start = timeit.default_timer()
csvfile = open("time.csv", "w", newline='')
writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

vid_writer = cv2.VideoWriter('pose.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_video.shape[1], frame_video.shape[0]))
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_asli = np.copy(image)
        image = np.zeros(image.shape)
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

      #  if(results.pose_landmarks is not None and results.left_hand_landmarks is not None and results.right_hand_landmarks is not None):
        if results.pose_landmarks: 
            data_tubuh = {}
            flag = False
            for i in range(len(pose_tubuh)):
                results.pose_landmarks.landmark[i].x = results.pose_landmarks.landmark[i].x * image.shape[0]
                results.pose_landmarks.landmark[i].y = results.pose_landmarks.landmark[i].y * image.shape[1]
                data_tubuh.update(
                {pose_tubuh[i] : results.pose_landmarks.landmark[i]}
                )
                flag = True
            # Get timestamp
            if flag:
                current_time = timeit.default_timer()
                time_elapsed = current_time - start
                data_tubuh.update({'Time' : time_elapsed})
                print(time_elapsed)
                
            alldata.append(data_tubuh)
            
        if results.right_hand_landmarks:
            data_tangan_kanan = {}
            flag = False
            for i in range(len(pose_tangan)):
                results.right_hand_landmarks.landmark[i].x = results.right_hand_landmarks.landmark[i].x * image.shape[0]
                results.right_hand_landmarks.landmark[i].y = results.right_hand_landmarks.landmark[i].y * image.shape[1]
                data_tangan_kanan.update(
                {pose_tangan[i] : results.right_hand_landmarks.landmark[i]}
                )
                flag = True
            # Get timestamp
            if flag:
                current_time = timeit.default_timer()
                time_elapsed = current_time - start
                data_tangan_kanan.update({'Time' : time_elapsed})
                print(time_elapsed)
                
            alldata.append(data_tangan_kanan)
                    
        if results.left_hand_landmarks:
            data_tangan_kiri  = {}
            flag = False
            for i in range(len(pose_tangan)):
                results.left_hand_landmarks.landmark[i].x = results.left_hand_landmarks.landmark[i].x * image.shape[0]
                results.left_hand_landmarks.landmark[i].y = results.left_hand_landmarks.landmark[i].y * image.shape[1]
                data_tangan_kiri.update(
                {pose_tangan_2[i] : results.left_hand_landmarks.landmark[i]}
                )
                flag = True
            # Get timestamp
            if flag:
                current_time = timeit.default_timer()
                time_elapsed = current_time - start
                data_tangan_kiri.update({'Time' : time_elapsed})
                print(time_elapsed)
                
            alldata.append(data_tangan_kiri)
            

        cv2.namedWindow('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,)
        cv2.imshow('MediaPipe Holistic', image) #sudah menampilkan backgrounnd hitam dan skeleton
        cv2.imshow('Gambar asli', image_asli)
        count = count + 1
        print(count)
        
        
        print(alldata)
        fps_time = time.time()
        vid_writer.write(image)
        #plt.imshow((image*255).astype(np.uint8))
        #plt.savefig("image-frame/" + str(count) + ".jpg")
        if cv2.waitKey(5) & 0xFF == 27:
            df = pd.DataFrame(alldata)
            #input = input()
            df.to_csv("pose_data.csv")
            break
cap.release()