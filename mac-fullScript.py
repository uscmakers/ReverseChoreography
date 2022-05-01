import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import time
import sys
import argparse
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
from os.path import exists
import re
import tsfresh
from tsfresh.feature_extraction import MinimalFCParameters

#Initialize Spotify Client
logger = logging.getLogger()
logging.basicConfig()
CLIENT_ID="9793440f0a5047c59c70bcfcf91ad589"
CLIENT_SECRET= "b66dc3a5f9f34207bebee32a25745368"
REDIRECT_URL="http://localhost/"
client_credentials_manager = SpotifyClientCredentials(client_id = CLIENT_ID, client_secret = CLIENT_SECRET)
oAuth = SpotifyOAuth(client_id = CLIENT_ID, client_secret = CLIENT_SECRET, redirect_uri = REDIRECT_URL, scope = 'user-modify-playback-state,playlist-modify-public')
sp = spotipy.Spotify(auth_manager =oAuth)


song_artist_pairs = {
    '1':('thats_what_i_like','bruno_mars'),
    '2':('humble','kendrick_lamar'),
    '3':('skeletons','keshi'),
    '4':('slow_dancing_in_the_dark','joji'),
    '5':('lite_spots','kaytranada'),
    '6':('woman','doja_cat'),
    '7':('get_up','ciara'),
    '8':('throwin_elbows','excision'),
    '9':('power','little mix'),
    '10':('peaches','justin_bieber'),
    '11':('knife_talk','drake'),
    '12':('fool_around','yas'),
    '13':('levitating','dua_lipa'),
    '14':('feed_the_fire','lucky_daye'),
    '15':('easily','bruno_major'),
    '16':('good_4_u','olivia_rodrigo'),
    '17':('all_i_wanna_do','jay_park'),
    '18':('sad_girlz_luv_money','amaarae'),
    '19':('tik_tok','kesha'),
    '20':('ymca','village_people'),
    '21':('intuition_interlude','jamie_foxx'),
    '22':('kilby_girl','the_backseat_lovers'),
    '23':('a_thousand_miles','vanessa_carlton')
}

def get_args():
    parser = argparse.ArgumentParser(description='Recommendations for the given song')
    parser.add_argument('-s', '--song', required=True, help='Name of Song')
    parser.add_argument('-a', '--artist', required=True, help='Name of Artist')
    return parser.parse_args()

def get_song(name_song, name_artist):
    results = sp.search(q=name_song + ' '+name_artist, type='track')
    items = results['tracks']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None

def show_recommendations_for_song(song):
    results = sp.recommendations(seed_tracks=[song['id']], limit=5)
    print("Recommendations:")
    for track in results['tracks']:
        print("TRACK: ",track['name'], " - ",track['artists'][0]['name'])
        sp.add_to_queue(track['uri'])
    return [track['uri'] for track in results['tracks']]

def show_feature_based_recommendations_for_song(song):
    song_features = sp.audio_features([song['uri']])
    kwargs = {"target_danceability":song_features[0]["danceability"], "target_energy":song_features[0]['energy'], "target_key":song_features[0]['key'], "target_loudness":song_features[0]['loudness'], "target_speechiness":song_features[0]['speechiness'], "target_acousticness":song_features[0]['acousticness'], "target_instrumentalness":song_features[0]['instrumentalness'], "target_liveness":song_features[0]['liveness'], "target_valence":song_features[0]['valence'], "target_tempo":song_features[0]['tempo'], "target_time_signature":song_features[0]['time_signature']}
    results = sp.recommendations(seed_artists=None, seed_genres=None, seed_tracks=[song['id']], limit=5, country=None, **kwargs)
    print("Feature-based Recommendations:")
    for track in results['tracks']:
        print("TRACK: ",track['name'], " - ",track['artists'][0]['name'])
        sp.add_to_queue(track['uri'])
    return [track['uri'] for track in results['tracks']]

def get_audio_features(song_name, artist_name):
    song = get_song(song_name, artist_name)
    if(song is None):
        return None
    song_features = sp.audio_features([song['uri']])
    audio_feature_list = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    audio_feat = []
    audio_feat.append(song_features[0].get('danceability'))
    audio_feat.append(song_features[0].get('energy'))
    audio_feat.append(song_features[0].get('loudness'))
    audio_feat.append(song_features[0].get('speechiness'))
    audio_feat.append(song_features[0].get('acousticness'))
    audio_feat.append(song_features[0].get('instrumentalness'))
    audio_feat.append(song_features[0].get('liveness'))
    audio_feat.append(song_features[0].get('valence'))
    audio_feat.append(song_features[0].get('tempo'))
    return audio_feat

def find_diffs(feature1, feature2):
    diffs = 0
    for i in range(len(feature1)):
        diffs += abs(feature1[i] - feature2[i])
    return diffs

def find_diffs_sq(feature1, feature2):
    diffs = 0
    for i in range(len(feature1)):
        diffs += (feature1[i] - feature2[i])**2
    return diffs

def find_closest_song(features):
    minDistance = sum(abs(features))
    minIndex = 0
    for i in range(len(song_artist_features)):
        diffs = find_diffs(features, song_artist_features[i][2])
        if diffs < minDistance:
            minDistance = diffs
            minIndex = i
    return minIndex

def split_ele(x):
    a = np.array(re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", x))
    return a.astype(float)

def preprocess(file_name):
    num_features = 10
    num_nodes = 14
    num_samples = 10
    pose_landmark_subset = ['LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP','LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    df_columns = ['LEFT_EYE_OUTER_POS', 'LEFT_EYE_OUTER_DIST', 'RIGHT_EYE_OUTER_POS', 'RIGHT_EYE_OUTER_DIST', 'LEFT_SHOULDER_POS', 'LEFT_SHOULDER_DIST', 'RIGHT_SHOULDER_POS', 'RIGHT_SHOULDER_DIST', 'LEFT_ELBOW_POS', 'LEFT_ELBOW_DIST', 'RIGHT_ELBOW_POS', 'RIGHT_ELBOW_DIST', 'LEFT_WRIST_POS', 'LEFT_WRIST_DIST', 'RIGHT_WRIST_POS', 'RIGHT_WRIST_DIST', 'LEFT_HIP_POS', 'LEFT_HIP_DIST', 'RIGHT_HIP_POS', 'RIGHT_HIP_DIST', 'LEFT_KNEE_POS', 'LEFT_KNEE_DIST', 'RIGHT_KNEE_POS', 'RIGHT_KNEE_DIST', 'LEFT_ANKLE_POS', 'LEFT_ANKLE_DIST', 'RIGHT_ANKLE_POS', 'RIGHT_ANKLE_DIST']
    df = pd.read_csv(file_name, sep = ',', usecols=[4, 7, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29])
    splitDf = df
    x_data = pd.DataFrame(columns=df_columns, index=range(len(df)))
    y_data = pd.DataFrame(columns=df_columns, index=range(len(df)))
    for node in pose_landmark_subset:
        curr = df[node]
        vals = [split_ele(x) for x in curr]
        for row in range(len(vals)):
            colname_pos = node + "_POS"
            colname_dist = node + "_DIST"
            if(row == 0):
                x_data[colname_dist][row] = 0
                y_data[colname_dist][row] = 0
            else:
                x_data[colname_dist][row] = vals[row][0] - vals[row-1][0]
                y_data[colname_dist][row] = vals[row][1] - vals[row-1][1]
            x_data[colname_pos][row] = vals[row][0]
            y_data[colname_pos][row] = vals[row][1]
    return [x_data, y_data]

def parse(x_data, y_data):
    num_features = 10
    num_nodes = 14
    num_samples = 10
    df_columns = ['LEFT_EYE_OUTER_POS', 'LEFT_EYE_OUTER_DIST', 'RIGHT_EYE_OUTER_POS', 'RIGHT_EYE_OUTER_DIST', 'LEFT_SHOULDER_POS', 'LEFT_SHOULDER_DIST', 'RIGHT_SHOULDER_POS', 'RIGHT_SHOULDER_DIST', 'LEFT_ELBOW_POS', 'LEFT_ELBOW_DIST', 'RIGHT_ELBOW_POS', 'RIGHT_ELBOW_DIST', 'LEFT_WRIST_POS', 'LEFT_WRIST_DIST', 'RIGHT_WRIST_POS', 'RIGHT_WRIST_DIST', 'LEFT_HIP_POS', 'LEFT_HIP_DIST', 'RIGHT_HIP_POS', 'RIGHT_HIP_DIST', 'LEFT_KNEE_POS', 'LEFT_KNEE_DIST', 'RIGHT_KNEE_POS', 'RIGHT_KNEE_DIST', 'LEFT_ANKLE_POS', 'LEFT_ANKLE_DIST', 'RIGHT_ANKLE_POS', 'RIGHT_ANKLE_DIST']
    curr_extracted_vector = pd.DataFrame()
    for col in df_columns:
        col_x = x_data[col]
        col_y = y_data[col]
        xname = col + "_x"
        yname = col + "_y"
        settings = {
            xname: {
                "kurtosis": None, 
                "standard_deviation": None, 
                "autocorrelation": [{"lag": 10}],
                "approximate_entropy": [{"m": 20, "r": 0.05}],
                "c3": [{"lag": 10}],
                "cid_ce": [{"normalize": True}]
            }, 
            yname: {
                "kurtosis": None, 
                "standard_deviation": None, 
                "autocorrelation": [{"lag": 10}],
                "approximate_entropy": [{"m": 20, "r": 0.05}],
                "c3": [{"lag": 10}],
                "cid_ce": [{"normalize": True}]
            }
        }
        comb = pd.DataFrame(data=[col_x, col_y], index=[xname, yname]).T
        comb.rename_axis("time")
        comb["id"] = 1
        comb["time"] = comb.index
        curr_extracted = tsfresh.extract_features(comb, column_id = "id", column_sort="time", column_kind=None, column_value=None, kind_to_fc_parameters=settings, disable_progressbar=True, n_jobs = 10)
        curr_extracted_vector = pd.concat([curr_extracted_vector, curr_extracted], axis=1)
    return curr_extracted_vector

def show_feature_based_recommendations_for_song(audio_features):
    oAuth = SpotifyOAuth(client_id = CLIENT_ID, client_secret = CLIENT_SECRET, redirect_uri = REDIRECT_URL, scope = 'user-modify-playback-state')
    sp = spotipy.Spotify(auth_manager =oAuth)
    #TODO adjust vals as needed
    kwargs = {"min_danceability":float(audio_features[0])-.1,
          "target_danceability":audio_features[0],
          "max_danceability":float(audio_features[0])+.1,
          "min_energy":float(audio_features[1])-.1,
          "target_energy":audio_features[1],
          "max_energy":float(audio_features[1])+.1,
          "min_speechiness":float(audio_features[3])-.1,
          "target_speechiness":audio_features[3],
          "max_speechiness":float(audio_features[3])+.1,
          "min_loudness":float(audio_features[2])-20,
          "target_loudness":audio_features[2],
          "max_loudness":float(audio_features[2])+20,
          "min_acousticness":float(audio_features[4])-.25,
          "target_acousticness":audio_features[4],
          "max_acousticness":float(audio_features[4])+.25,
          "min_instrumentalness":float(audio_features[5])-.1,
          "target_instrumentalness":audio_features[5],
          "max_instrumentalness":float(audio_features[5])+.1,
          "min_liveness":float(audio_features[6])-.25,
          "target_liveness":audio_features[6],
          "max_liveness":float(audio_features[6])+.25,
          "min_valence":float(audio_features[7])-.1,
          "target_valence":audio_features[7],
          "max_valence":float(audio_features[7])+.1,
          "min_tempo":float(audio_features[8])-15,
          "target_tempo":audio_features[8],
          "max_tempo":float(audio_features[8])+15
          }
    results = sp.recommendations(seed_artists=None, seed_genres=['alternative', 'r-n-b', 'rap', 'edm', 'pop'], seed_tracks=None, limit=10, country=None, **kwargs)
    print("Feature-based Recommendations:")
    for track in results['tracks']:
        print("TRACK: ",track['name'], " - ",track['artists'][0]['name'])
        sp.add_to_queue(track['uri'])
    return [track['uri'] for track in results['tracks']] 

def generate_classification_prediction(data):
    features = data.to_numpy()
    
    
    filename = 'classification_model.sav'
    model = pickle.load(open(filename, 'rb'))
    
    prediction = model.predict(features)
    artist_song = song_artist_pairs[prediction[0]]
    song = get_song(artist_song[0], artist_song[1])

    print('\n\nClassification Results')
    
    return show_recommendations_for_song(song)

def create_playlist(filename, regression_songs, classification_songs):
    myId = sp.current_user()['id']
    playlistInfo = sp.user_playlist_create(myId, filename, True, False, 'rev choreo recs for filename')
    playlistId = playlistInfo['id']
    url = playlistInfo['external_urls']['spotify']
    
    sp.user_playlist_add_tracks(myId, playlistId, regression_songs)
    sp.user_playlist_add_tracks(myId, playlistId, classification_songs)
    print('Find YOUR playlist at: ', url)
    return

# for generating data, run as 'python3 mediaposetest3.py [dancer_id] [song_artist_id]' e.g. 'python3 mediaposetest3.py 1 1'
# for tests, run as 'python3 mediaposetest3.py [testName]' e.g. 'python3 mediaposetest3.py jeff'
def videoCap():
    args = sys.argv[1:]

    run_name = "data/dummy.csv"
    if len(args) == 2:
        dancer_id = args[0]
        song_artist_id = args[1]
        run_name = f'data/{dancer_id}_{song_artist_pairs[song_artist_id][0]}_{song_artist_pairs[song_artist_id][1]}.csv'
    elif len(args) == 1:
        testName = args[0]
        run_name = f'data/{testName}_test.csv'

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
            raw_video_feed = np.copy(image)
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

            cv2.namedWindow('MediaPipe Holistic', cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty('MediaPipe Holistic', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,)
            cv2.imshow('MediaPipe Holistic', image)
            # cv2.imshow('Video Feed', raw_video_feed)
            count = count + 1
            print(count)
            fps_time = time.time()
            image = np.uint8(image)
            vid_writer.write(image)
            #plt.imshow((image*255).astype(np.uint8))
            #plt.savefig("image-frame/" + str(count) + ".jpg")
            if (cv2.waitKey(5) & 0xFF == 27) or (end_time-start_time >= 23):
                df = pd.DataFrame(alldata)
                df.to_csv(run_name)
                break
    cap.release()
    vid_writer.release()
    return run_name
    
def recommend(run_name):
    tempo = input("Input tempo: ")
    run_filename = run_name

    filename = 'model.sav'
    model = pickle.load(open(filename, 'rb'))

    [x_data, y_data] = preprocess(run_filename)
    extracted_features = parse(x_data, y_data)
    classification_extracted_features = extracted_features

# For simple decision tree model, disregard the feature selection performed below
# To use the feature selection performed below, first either re-train or download the regressor model and substitute it in
    # selected_features = pd.read_csv("Selected_Feature_List.csv")["0"].to_numpy()
    # extracted_features = extracted_features[selected_features]

    predicted_audio_features = model.predict(extracted_features)

    features = np.append(predicted_audio_features[0][0:len(predicted_audio_features[0])-1], tempo)
    
    print("Recommendations from all songs:")
    regression_recs = show_feature_based_recommendations_for_song(features)
    
    print("Recommendations from training set songs:")
    classification_recs = generate_classification_prediction(classification_extracted_features)
    
    create_playlist(run_filename, regression_recs, classification_recs)

if __name__ == "__main__":
    run_name = videoCap()
    recommend(run_name)
