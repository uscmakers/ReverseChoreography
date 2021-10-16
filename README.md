# ReverseChoreography
Repository for Reverse Choreograph (Fall 2021-Spring 2022)

SETUP: 

Every time a new terminal is opened, authentication info must be
updated to use Spotify's API. Run the following commands to do this:

export SPOTIPY_CLIENT_ID= '5018f42c45cd4e6b8ffa2032c91c524e'
export SPOTIPY_CLIENT_SECRET= '36a70475253244899e168327c207246a'
export SPOTIPY_REDIRECT_URL= 'http://localhost'

RECOMMEND.PY:

python3 recommend.py -s "song name" -a "artist name"

Both -s and -a arguments are required. If either includes
multiple words, use double quotes around the argument and
insert spaces as needed.

This function will generate a list of recommendations 
based on given song. Each time this function is called, 
there may be different recommendations given.
