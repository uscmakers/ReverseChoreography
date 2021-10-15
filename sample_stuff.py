import requests

# your endpoint, you request to it
SPOTIFY_CREATE_PLAYLIST_URL = 	'https://api.spotify.com/v1/users/alicegusev/playlists'
ACCESS_TOKEN = 'BQCmxQ0dEZ2iQk9LAr__Els1SaVTaawJLKXpCKMv3aKx0Htz_olJegT_G5ZxpvRpAmqNaMj5C5YAvOR37Pwq6vtoUmw4GBxdNdzhOP18Kz3DphijgWziWFYLYvDQ6XvFAdKGca4sqzAkps-GGYLfIwiS3uvxyYibMm76zTrj-EMG-x9X6vGwPdcjZLdk4JHlNLvrwKHbAgpS'
TRACK_ID = 'https://api.spotify.com/v1/tracks/0McOOEVI11ks2PPOVZLou8'

# song recs
SEED_ARTISTS = 'https://api.spotify.com/v1/recommendations?market=US&seed_artists=4NHQUGzhtTLFvgF5SZesLK&seed_tracks=0c6xIDDpzE81m2q797ordA&min_energy=0.4&min_popularity=50'


def create_playlist_on_spotify(name, public):
	response = requests.post(
		SPOTIFY_CREATE_PLAYLIST_URL,
		# pass aithorization header, key: authoriation, val: 
		headers={
			"Authorization": f"Bearer {ACCESS_TOKEN}"
		},
		json={
			"name": name,
			"public": public
		}
	)
	json_resp = response.json()
	return json_resp

def get_track():
	response = requests.get(
		TRACK_ID,
		# pass aithorization header, key: authoriation, val: 
		headers={
			"Authorization": f"Bearer {ACCESS_TOKEN}"
		}
	)
	json_resp = response.json()

	return json_resp



def get_recommendations():
	response = requests.get(
		TRACK_ID,
		# pass aithorization header, key: authoriation, val: 
		headers={
			"Authorization": f"Bearer {ACCESS_TOKEN}"
		}
	)
	json_resp = response.json()

	return json_resp

def main():


	# playlist = create_playlist_on_spotify(
	# 	name = "Makers Sample Playlist",
	# 	public= False
	# )

	track = get_track()

	recommendation = get_recommendations()

	print(f"REC: {recommendation}")

if __name__ == '__main__':
	main()