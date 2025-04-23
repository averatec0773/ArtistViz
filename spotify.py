from environment import *

SPOTIFY_CLIENT_ID = '...'
SPOTIFY_CLIENT_SECRET = '...'

def get_token():
    auth_string = SPOTIFY_CLIENT_ID + ":" + SPOTIFY_CLIENT_SECRET
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def search_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=10"

    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)["artists"]["items"]
    if len(json_result) == 0:
        print("No Artist Found...")
        return None
    
    # print(json_result)
    return json_result

def search_top_tracks(token, artist_id, country = "US"): # Default to US
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country={country}"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result

def show_image(url, size=(300,300)):
    try:
        response = get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image = image.resize(size)
        display(image)
    except Exception as e:
        print(f"Error：{e}")

def process_top_track(artist_top_tracks):
    top_tracks_data = []

    for track in artist_top_tracks:
        track_id = track['id']                            # Unique track ID
        track_name = track['name']                        # Track name
        track_album = track['album']['name']              # Track album name
        
        track_date = track['album']['release_date']       # Track release date
        track_date = track['album']['release_date'].replace('-', '/')

        track_popularity = track['popularity']            # Track popularity score
        image_url = track['album']['images'][0]['url']    # Largest album image (always the first)

        # Create a dictionary with the required fields
        track_info = {
            'id': track_id,
            'name': track_name,
            'album_name': track_album,
            'release_date': track_date,
            'popularity': track_popularity,
            'image_url': image_url
        }

        # Append to result list
        top_tracks_data.append(track_info)
    
    return top_tracks_data


