import re
import os
import csv
import nltk
import torch
import string
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_top_5_words_from_lyrics(lyric_file, stopwords_file = './data/stopwords-en.txt'):
    # Define stopwords file
    # stopwords_file = './lyrics/stopwords-en.txt'

    # Check if lyric file exists
    if not os.path.exists(lyric_file):
        print(f"Lyrics file {lyric_file} does not exist.")
        return

    # Load stopwords
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set(word.strip().lower() for word in f if word.strip())

    all_words = []

    # Read lyrics and extract words
    with open(lyric_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lyric = row['lyric'].lower()
            # Use regex to extract only words
            words = re.findall(r'\b[a-z]+\b', lyric)
            # Filter out stopwords
            filtered_words = [word for word in words if word not in stopwords]
            all_words.extend(filtered_words)

    # Count and show top 5
    word_counts = Counter(all_words)
    top_5_words = word_counts.most_common(5)

    return top_5_words

def calculate_mood_counts(lyrics_file_path, top_n = 3):
    # Load the CSV file
    lyrics_data = pd.read_csv(lyrics_file_path)

    # Apply Topic Modeling with LDA
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(lyrics_data['lyric'])

    # Fit the LDA model
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    # Define a simple mood mapping
    topic_to_mood = {
        0: "Romantic",
        1: "Sad",
        2: "Happy",
        3: "Angry",
        4: "Energetic"
    }

    # Get topic distribution and assign dominant topic to each song
    topic_distribution = lda.transform(X)
    dominant_topics = topic_distribution.argmax(axis=1)

    # Map topics to moods
    lyrics_data['mood'] = [topic_to_mood[topic] for topic in dominant_topics]

    # Count each mood and calculate percentage
    mood_counts = lyrics_data['mood'].value_counts(normalize=True)  # normalize=True gives percentages
    mood_percentage_list = [(mood, round(percent, 4)) for mood, percent in mood_counts.items()]

    top_3_moods = mood_percentage_list[:top_n]

    return top_3_moods

# Function to convert hex color to RGB
def hex_to_RGB(hex_str):
    return tuple(int(hex_str[i:i+2], 16) for i in range(1, 7, 2))

# Function to generate a color gradient
def get_color_gradient(c1, c2, n):
    c1_rgb = np.array(hex_to_RGB(c1)) / 255  # Normalize to [0, 1]
    c2_rgb = np.array(hex_to_RGB(c2)) / 255  # Normalize to [0, 1]
    
    # Generate n colors in the gradient
    mix_pcts = np.linspace(0, 1, n)
    rgb_colors = [(1 - mix) * c1_rgb + mix * c2_rgb for mix in mix_pcts]
    
    # Convert the colors back to [0, 255] range and return them as hex strings
    rgb_colors_hex = ['#' + ''.join([format(int(val * 255), '02x') for val in color]) for color in rgb_colors]
    
    return rgb_colors_hex

# Function to generate gradient background based on the top 2 moods
def generate_mood_gradient(top_2_mood, width=800, height=1200, n_colors=800):
    mood_colors = {
    "Romantic": "#FF66CC",  # Pink
    "Happy": "#FF0000",     # Red
    "Sad": "#0000FF",       # Blue
    "Angry": "#FF6600",     # Orange
    "Energetic": "#00FF00"  # Green
    }
    # Generate the gradient colors between the top 2 moods
    colors = get_color_gradient(mood_colors[top_2_mood[0]], mood_colors[top_2_mood[1]], n_colors)
    
    # Create a new image
    img = Image.new('RGB', (width, height))
    
    # Draw the gradient background
    for i, color in enumerate(colors):
        img.paste(Image.new('RGB', (1, height), color=color), (i, 0))
    
    return img

# Function to get emotions count from the lyrics
def get_emotions(file_path, top_n=3):

    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

    # Load the dataset
    lyrics_data = pd.read_csv(file_path)
    
    emotions = []
    
    # Loop through each song's lyrics and predict emotions
    for lyrics in lyrics_data['lyric']:
        inputs = tokenizer(lyrics, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the top_n emotions
        top_indices = probabilities.topk(top_n).indices[0].tolist()
        top_emotions = [model.config.id2label[idx] for idx in top_indices]
        
        emotions.append(top_emotions)

    # Flatten the list of emotions and count the frequency of each emotion
    emotion_counts = pd.Series([emotion for emotions in emotions for emotion in emotions]).value_counts()

    # Get top N emotions as list of (emotion, count) tuples
    top_n_emotions = list(emotion_counts.head(top_n).items())

    return top_n_emotions

def sentiment_calculation(top_3_moods, top_3_emotions):
    # Define mood sentiment mapping
    positive_moods = {"Happy", "Romantic", "Energetic"}
    negative_moods = {"Sad", "Angry"}

    # Define emotion sentiment mapping
    positive_emotions = {"joy", "surprise", "neutral"}
    negative_emotions = {"anger", "disgust", "fear", "sadness"}

    # Initialize counters
    mood_positive = 0
    mood_total = 0
    for mood, percentage in top_3_moods:
        mood_total += percentage
        if mood in positive_moods:
            mood_positive += percentage
    mood_negative = mood_total - mood_positive

    # Emotions
    emotion_positive = 0
    emotion_total = 0
    for emotion, count in top_3_emotions:
        emotion_total += count
        if emotion in positive_emotions:
            emotion_positive += count
    emotion_negative = emotion_total - emotion_positive

    # Normalize each to 0~1
    if mood_total > 0:
        mood_pos_ratio = mood_positive / mood_total
    else:
        mood_pos_ratio = 0.5  # default neutral

    if emotion_total > 0:
        emotion_pos_ratio = emotion_positive / emotion_total
    else:
        emotion_pos_ratio = 0.5

    # Combine mood and emotion with equal weight
    positive = round((mood_pos_ratio + emotion_pos_ratio) / 2, 2)
    negative = round(1 - positive, 2)

    sentiment_distribution = [positive, negative]
    return sentiment_distribution

def generate_yearly_summary_poster(artist_name, artist_cover_url, top_5_words, top_3_emotions, top_3_moods, sentiment_distribution, font_path_bold, font_path_light):

    # Gradient background
    width, height = 800, 1200

    top_2_moods = [mood for mood, _ in top_3_moods[:3]]
    poster = generate_mood_gradient(top_2_moods, width=width, height=height)

    draw = ImageDraw.Draw(poster)

    # Fonts
    font_title = ImageFont.truetype(font_path_bold, 50)
    font_subtitle = ImageFont.truetype(font_path_bold, 40)
    font_body = ImageFont.truetype(font_path_light, 36)

    # Title
    title = f"{artist_name}'s Yearly Wrapped-Up"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    title_position = ((width - (bbox[2] - bbox[0])) // 2, 20)
    draw.text(title_position, title, font=font_title, fill="black")

    # === Artist Cover (centered below the title) ===
    cover_size = 400
    cover_position = ((width - cover_size) // 2, 120)
    try:
        response = requests.get(artist_cover_url)
        artist_cover = Image.open(BytesIO(response.content)).convert("RGB").resize((cover_size, cover_size))
        artist_cover_with_border = ImageOps.expand(artist_cover, border=3, fill='white')
        poster.paste(artist_cover_with_border, cover_position)
    except Exception as e:
        # fallback: draw a placeholder if loading fails
        draw.rectangle([cover_position,
                        (cover_position[0] + cover_size, cover_position[1] + cover_size)],
                       outline="black", width=3)
        draw.text((cover_position[0] + 20, cover_position[1] + cover_size // 2),
                  "Cover Load Failed", font=font_body, fill="black")

    # Top and bottom sections
    top_half_y = cover_position[1] + cover_size + 50
    bottom_half_y = top_half_y + 250

    # 1. Top 5 Words
    draw.text((50, top_half_y), "Top 5 Words", font=font_subtitle, fill="black")
    for i, (word, count) in enumerate(top_5_words, 1):
        draw.text((50, top_half_y + 30 * (i + 1)), f"{i}. {word} ({count})", font=font_body, fill="black")


    # 2. Top 3 Emotions
    draw.text((400, top_half_y), "Top 3 Emotions", font=font_subtitle, fill="black")
    for i, (emotion,count) in enumerate(top_3_emotions, 1):
        draw.text((400, top_half_y + 30 * (i + 1)), f"{i}. {emotion} ({count})", font=font_body, fill="black")

    # 3. Sentiment
    draw.text((50, bottom_half_y), "Sentiment", font=font_subtitle, fill="black")
    draw.text((50, bottom_half_y + 60), f"Positive: {sentiment_distribution[0]}%", font=font_body, fill="black")
    draw.text((50, bottom_half_y + 120), f"Negative: {sentiment_distribution[1]}%", font=font_body, fill="black")

    # 4. Mood Distribution
    draw.text((400, bottom_half_y), "Top 3 Mood", font=font_subtitle, fill="black")
    for i, (mood,percentage) in enumerate(top_3_moods):
        draw.text((400, bottom_half_y + 60 * (i + 1)),
                  f"{i + 1}. {mood.capitalize()} ({100*percentage}%)", font=font_body, fill="black")

    # Show image using plt
    plt.figure(figsize=(10, 14))
    plt.imshow(poster)
    plt.axis('off')
    plt.tight_layout()
    plt.show()