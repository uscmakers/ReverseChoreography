import argparse
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
from os.path import exists
import numpy as np
import re
import tsfresh
from tsfresh.feature_extraction import MinimalFCParameters


logger = logging.getLogger()
logging.basicConfig()

CLIENT_ID="9793440f0a5047c59c70bcfcf91ad589"
CLIENT_SECRET= "b66dc3a5f9f34207bebee32a25745368" #LOG IN TO SPOTIFY FOR DEVELOPERS DASHBOARD->REVERSE CHOREOGRAPHY APP TO GET THIS. COPY AND PASTE HERE.
REDIRECT_URL="http://localhost/"


client_credentials_manager = SpotifyClientCredentials(client_id = CLIENT_ID, client_secret = CLIENT_SECRET)
oAuth = SpotifyOAuth(client_id = CLIENT_ID, client_secret = CLIENT_SECRET, redirect_uri = REDIRECT_URL, scope = 'user-modify-playback-state')

#Here, sp is the spotify api client
sp = spotipy.Spotify(auth_manager =oAuth)


def get_args():
    parser = argparse.ArgumentParser(description='Recommendations for the '
                                     'given song')
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

def show_feature_based_recommendations_for_song(song):
    song_features = sp.audio_features([song['uri']])
    kwargs = {"target_danceability":song_features[0]["danceability"], "target_energy":song_features[0]['energy'], "target_key":song_features[0]['key'], "target_loudness":song_features[0]['loudness'], "target_speechiness":song_features[0]['speechiness'], "target_acousticness":song_features[0]['acousticness'], "target_instrumentalness":song_features[0]['instrumentalness'], "target_liveness":song_features[0]['liveness'], "target_valence":song_features[0]['valence'], "target_tempo":song_features[0]['tempo'], "target_time_signature":song_features[0]['time_signature']}
    results = sp.recommendations(seed_artists=None, seed_genres=None, seed_tracks=[song['id']], limit=5, country=None, **kwargs)
    print("Feature-based Recommendations:")
    for track in results['tracks']:
        print("TRACK: ",track['name'], " - ",track['artists'][0]['name'])
        sp.add_to_queue(track['uri'])

def get_audio_features(song_name, artist_name):

    song = get_song(song_name, artist_name)

    #if get request didnt get anything
    if(song is None):
        return None

    song_features = sp.audio_features([song['uri']])

    audio_feature_list = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
                          'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

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

    pose_landmark_subset = ['LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
                            'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
                            'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    df_columns = ['LEFT_EYE_OUTER_POS', 'LEFT_EYE_OUTER_DIST', 'RIGHT_EYE_OUTER_POS', 'RIGHT_EYE_OUTER_DIST', 
        'LEFT_SHOULDER_POS', 'LEFT_SHOULDER_DIST', 'RIGHT_SHOULDER_POS', 'RIGHT_SHOULDER_DIST', 
        'LEFT_ELBOW_POS', 'LEFT_ELBOW_DIST', 'RIGHT_ELBOW_POS', 'RIGHT_ELBOW_DIST', 'LEFT_WRIST_POS', 
        'LEFT_WRIST_DIST', 'RIGHT_WRIST_POS', 'RIGHT_WRIST_DIST', 'LEFT_HIP_POS', 'LEFT_HIP_DIST', 
        'RIGHT_HIP_POS', 'RIGHT_HIP_DIST', 'LEFT_KNEE_POS', 'LEFT_KNEE_DIST', 'RIGHT_KNEE_POS', 
        'RIGHT_KNEE_DIST', 'LEFT_ANKLE_POS', 'LEFT_ANKLE_DIST', 'RIGHT_ANKLE_POS', 'RIGHT_ANKLE_DIST']
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

    df_columns = ['LEFT_EYE_OUTER_POS', 'LEFT_EYE_OUTER_DIST', 'RIGHT_EYE_OUTER_POS', 'RIGHT_EYE_OUTER_DIST', 
        'LEFT_SHOULDER_POS', 'LEFT_SHOULDER_DIST', 'RIGHT_SHOULDER_POS', 'RIGHT_SHOULDER_DIST', 'LEFT_ELBOW_POS', 
        'LEFT_ELBOW_DIST', 'RIGHT_ELBOW_POS', 'RIGHT_ELBOW_DIST', 'LEFT_WRIST_POS', 'LEFT_WRIST_DIST', 
        'RIGHT_WRIST_POS', 'RIGHT_WRIST_DIST', 'LEFT_HIP_POS', 'LEFT_HIP_DIST', 'RIGHT_HIP_POS', 'RIGHT_HIP_DIST', 
        'LEFT_KNEE_POS', 'LEFT_KNEE_DIST', 'RIGHT_KNEE_POS', 'RIGHT_KNEE_DIST', 'LEFT_ANKLE_POS', 
        'LEFT_ANKLE_DIST', 'RIGHT_ANKLE_POS', 'RIGHT_ANKLE_DIST']

    #MOTION FEATURE EXTRACTION
    curr_extracted_vector = pd.DataFrame()

    #LOOP THROUGH NODES
    for col in df_columns:
        col_x = x_data[col]
        col_y = y_data[col]

        #Every Node (Body Part) has this set of feaures

        xname = col + "_x"
        yname = col + "_y"

        # Timeseries features to extract
        # TODO: tune parameters for autocorrelation, approximate_entropy, c3, cid_ce
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

        curr_extracted = tsfresh.extract_features(comb, column_id = "id", column_sort="time", 
                                                  column_kind=None, column_value=None, 
                                                  kind_to_fc_parameters=settings, disable_progressbar=True)
        curr_extracted_vector = pd.concat([curr_extracted_vector, curr_extracted], axis=1)
    
    return curr_extracted_vector

def show_feature_based_recommendations_for_song(audio_features):
    kwargs = {"target_danceability":audio_features[0], "target_energy":audio_features[1], "target_speechiness":audio_features[2], "target_loudness":audio_features[3], "target_acousticness":audio_features[4], "target_instrumentalness":audio_features[5], "target_liveness":audio_features[6], "target_valence":audio_features[7], "target_tempo":audio_features[8]}
    results = sp.recommendations(seed_artists=None, seed_genres=['alternative', 'pop', 'dance'], seed_tracks=None, limit=5, country=None, **kwargs)
    print("Feature-based Recommendations:")
    for track in results['tracks']:
        print("TRACK: ",track['name'], " - ",track['artists'][0]['name'])
        #sp.add_to_queue(track['uri'])

def main():

    #SAVED MODEL
    filename = 'model.sav'
    model = pickle.load(open(filename, 'rb'))

    [x_data, y_data] = preprocess("data/10_ymca_village_people.csv")
    extracted_features = parse(x_data, y_data)
    predicted_audio_features = model.predict(extracted_features)
    remove_tempo = predicted_audio_features[0][0:len(predicted_audio_features[0])-1]
    
    tempo = input("Input tempo: ")
    append_tempo = np.append(remove_tempo, tempo)
    show_feature_based_recommendations_for_song(append_tempo)
    #args = get_args()
    #song = get_song(args.song, args.artist)
    #song_features = sp.audio_features([song['uri']])
    #if song:
    #    show_recommendations_for_song(song)
    #    print("\n")
    #    show_feature_based_recommendations_for_song(song)
    #else:
    #    logger.error("Can't find that song", args.song)

    #sp.add_to_queue(song['uri'])
    #sp.start_playback()


if __name__ == '__main__':
    main()
