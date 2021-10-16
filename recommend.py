import argparse
import logging

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


logger = logging.getLogger()
logging.basicConfig()

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


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
    results = sp.recommendations(seed_tracks=[song['id']])
    print("Recommendations:")
    for track in results['tracks']:
        print("TRACK: ",track['name'], " - ",track['artists'][0]['name'])

def main():
    args = get_args()
    song = get_song(args.song, args.artist)
    if song:
        show_recommendations_for_song(song)
    else:
        logger.error("Can't find that song", args.song)


if __name__ == '__main__':
    main()