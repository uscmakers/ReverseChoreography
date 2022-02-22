import argparse
import logging

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth


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

def main():
    args = get_args()
    song = get_song(args.song, args.artist)
    song_features = sp.audio_features([song['uri']])
    if song:
        show_recommendations_for_song(song)
        print("\n")
        show_feature_based_recommendations_for_song(song)
    else:
        logger.error("Can't find that song", args.song)
    

    #sp.add_to_queue(song['uri'])
    #sp.start_playback()


if __name__ == '__main__':
    main()
