import requests
import base64
import time
import urllib.parse
from . import config

class SpotifyClient:
    def __init__(self):
        self.access_token = self._get_access_token()
        self.headers = {'Authorization': f'Bearer {self.access_token}'}

    def _get_access_token(self):
        # Note: In a real app, implement the OAuth flow properly. 
        # This assumes you have a mechanism to get the code or refresh token.
        # For simplicity, we usually set this via env var or run the auth flow once.
        print("Please implement OAuth flow or set ACCESS_TOKEN manually for now.")
        return "YOUR_ACCESS_TOKEN"

    def get_playlist_tracks(self, playlist_id):
        tracks = []
        url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
        
        while url:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break
            
            data = response.json()
            for item in data.get('items', []):
                if item.get('track'):
                    tracks.append(item['track'])
            
            url = data.get('next')
            time.sleep(0.1)
        return tracks

    def get_release_years(self, track_ids):
        release_years = {}
        base_url = 'https://api.spotify.com/v1/tracks'
        
        # Chunk IDs into groups of 50
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i+50]
            ids_param = ','.join(batch)
            
            try:
                response = requests.get(f"{base_url}?ids={ids_param}", headers=self.headers)
                if response.status_code == 200:
                    results = response.json().get('tracks', [])
                    for track in results:
                        if track and track.get('album'):
                            date = track['album'].get('release_date', '')
                            if date:
                                release_years[track['id']] = date[:4]
                elif response.status_code == 429:
                    time.sleep(int(response.headers.get('Retry-After', 5)))
            except Exception as e:
                print(f"Error fetching years: {e}")
                
            time.sleep(0.1)
        return release_years