# pip install lyricsgenius
import lyricsgenius

GENIUS_CLIENT_ID = '...'
GENIUS_CLIENT_SECRET = '...'
GENIUS_TOKEN = '...'

def get_lyrics(artist_name, song_name):
    genius = lyricsgenius.Genius(GENIUS_TOKEN)
    lyrics_artist = genius.search_artist(artist_name, max_songs=0, sort='title')
    lyrics_song = lyrics_artist.song(song_name)
    lyrics = lyrics_song.lyrics
    return lyrics

# artist_name = ''
# song_name = ''
# lyrics = get_lyrics(artist_name,song_name)
# print(lyrics)