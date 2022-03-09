import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import time
import sys

song_artist_pairs = {
    '0':('love_girl','cnblue'),
    '1':('party_rock_anthem','lmfao'),
    '2':('lite_spots','kaytranada'),
    '3':('girlfriends_best_friend','walkabout'),
    '4':('scaredy_cat','dpr_ian'),
    '5':('girl_anachronism','the_dresden_dolls'),
    '6':('lil_darlin','count_basie'),
    '7':('el_invento','jose_gonzalez_calvaire'),
    '8':('test_drive','ariana_grande'),
    '9':('weight_in_gold','gallant'),
    '10':('easy','wheein'),
    '11':('comethru','jeremy_zucker'),
    '12':('classic','mkto'),
    '13':('tongue_tied','grouplove'),
    '14':('motley','post_malone'),
    '15':('knife_talk','drake'),
    '16':('feed_the_fire','lucky_daye'),
    '17':('lose_it','swmrs'),
    '18':('thats_what_i_like','bruno_mars'),
    '19':('humble','kendrick_lamar'),
    '20':('skeletons','keshi'),
    '21':('stairway_to_heaven','led_zeppelin'),
    '22':('slow_dancing_in_the_dark','joji'),
    '23':('peaches','justin_bieber'),
    '24':('kilby_girl','backseat_lovers'),
    '25':('jupiter_love','trey_songz'),
    '26':('woman','doja_cat'),
    '27':('get_up','ciara'),
    '28':('throwin_elbows','excision'),
    '29':('power','little mix'),
    '30':('ymca','village_people'),
    '31':('all_i_wanna_do','jay_park'),
    '32':('easily','bruno_major'),
    '33':('feed_the_fire','lucky_daye'),
    '34':('fool_around','yas'),
    '35':('good_4_u','olivia_rodrigo'),
    '36':('knife_talk','drake'),
    '37':('levitating','dua_lipa'),
    '38':('sad_girlz_luv_money','amaarae'),
    '39':('tik_tok','kesha'),
    '40':('a_thousand_miles','vanessa_carlton')
}

# run as 'python3 mediaposetest3.py [dancer_id] [song_artist_id]' e.g. 'python3 mediaposetest3.py 1 1'
def main():
    args = sys.argv[1:]
    dancer_id = args[0]
    song_artist_id = args[1]

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    count = 0
    alldata = []
    fps_time = 0
    start_time = time.time()

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
    vid_writer = cv2.VideoWriter('pose.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_video.shape[1], frame_video.shape[0]))
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            end_time = time.time()
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
                for i in range(len(pose_tubuh)):
                    results.pose_landmarks.landmark[i].x = results.pose_landmarks.landmark[i].x * image.shape[0]
                    results.pose_landmarks.landmark[i].y = results.pose_landmarks.landmark[i].y * image.shape[1]
                    data_tubuh.update(
                    {pose_tubuh[i] : results.pose_landmarks.landmark[i]}
                    )
                alldata.append(data_tubuh)
                
            if results.right_hand_landmarks:
                for i in range(len(pose_tangan)):
                    results.right_hand_landmarks.landmark[i].x = results.right_hand_landmarks.landmark[i].x * image.shape[0]
                    results.right_hand_landmarks.landmark[i].y = results.right_hand_landmarks.landmark[i].y * image.shape[1]
                    data_tubuh.update(
                    {pose_tangan[i] : results.right_hand_landmarks.landmark[i]}
                    )
                alldata.append(data_tubuh)
                        
            if results.left_hand_landmarks:
                for i in range(len(pose_tangan)):
                    results.left_hand_landmarks.landmark[i].x = results.left_hand_landmarks.landmark[i].x * image.shape[0]
                    results.left_hand_landmarks.landmark[i].y = results.left_hand_landmarks.landmark[i].y * image.shape[1]
                    data_tubuh.update(
                    {pose_tangan_2[i] : results.left_hand_landmarks.landmark[i]}
                    )
                alldata.append(data_tubuh)

            if results.pose_landmarks or results.right_hand_landmarks or results.left_hand_landmarks:
                data_tubuh.update(
                    {'TIMESTAMP': end_time-start_time}
                )
                alldata.append(data_tubuh)

            cv2.namedWindow('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,)
            cv2.imshow('MediaPipe Holistic', image) #sudah menampilkan backgrounnd hitam dan skeleton
            cv2.imshow('Gambar asli', image_asli)
            count = count + 1
            print(count)
            fps_time = time.time()
            vid_writer.write(image)
            #plt.imshow((image*255).astype(np.uint8))
            #plt.savefig("image-frame/" + str(count) + ".jpg")
            if (cv2.waitKey(5) & 0xFF == 27) or (start_time-end_time >= 20):
                df = pd.DataFrame(alldata)
                run_name = f'data/{dancer_id}_{song_artist_pairs[song_artist_id][0]}_{song_artist_pairs[song_artist_id][1]}.csv'
                df.to_csv(run_name)
                break
    cap.release()

if __name__ == "__main__":
    main()