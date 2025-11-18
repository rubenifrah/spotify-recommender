import os

# Spotify Credentials (Load from environment variables for security)
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_USER_ID = os.getenv("SPOTIFY_USER_ID")
REDIRECT_URI = 'http://localhost:8888/callback'

# Genre Mapping from your notebook
GENRE_MAP = {
    'pop': ['pop', 'pop-film', 'power-pop', 'indie-pop', 'synth-pop', 'piano', 'kids', 'study'],
    'rock': ['rock', 'hard-rock', 'punk', 'punk-rock', 'alt-rock', 'emo', 'metal', 'metalcore', 'hardcore',
             'grunge', 'rock-n-roll', 'garage', 'guitar', 'psych-rock', 'rockabilly', 'singer-songwriter', 'ska'],
    'hiphop_rnb': ['hip-hop', 'r-n-b', 'trap', 'rap'],
    'electronic': ['edm', 'electro', 'electronic', 'house', 'techno', 'trance', 'dubstep', 'deep-house',
                   'minimal-techno', 'detroit-techno', 'idm', 'progressive-house', 'breakbeat', 'drum-and-bass', 'trip-hop'],
    'jazz_blues': ['jazz', 'blues', 'soul', 'funk', 'groove'],
    'classical': ['classical', 'ambient', 'new-age', 'opera', 'sleep', 'piano'],
    'latin_world': ['latin', 'latino', 'brazil', 'mpb', 'salsa', 'samba', 'forro', 'pagode', 'reggaeton',
                    'spanish', 'portuguese', 'world-music', 'sertanejo', 'tango'],
    'folk_country': ['country', 'folk', 'bluegrass', 'acoustic', 'honky-tonk'],
    'asian_pop': ['k-pop', 'j-pop', 'j-rock', 'j-dance', 'j-idol', 'mandopop', 'cantopop', 'anime'],
    'other': ['gospel', 'comedy', 'show-tunes', 'children', 'happy', 'sad', 'party', 'indian', 'iranian',
              'french', 'german', 'swedish', 'malay', 'turkish', 'goth', 'grindcore', 'black-metal',
              'death-metal', 'dub']
}

# Invert the map for easier lookup
FLAT_GENRE_MAP = {sub: group for group, subs in GENRE_MAP.items() for sub in subs}