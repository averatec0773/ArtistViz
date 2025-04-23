import re
import os
import json
import base64
import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from io import BytesIO
from requests import post, get
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import display

def draw_auto_fit_text(draw, text, font_path, max_width, max_size, min_size, position, fill="white", anchor=None):
    font_size = max_size
    while font_size >= min_size:
        font = ImageFont.truetype(font_path, size=font_size)
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            break
        font_size -= 1
    draw.text(position, text, font=font, fill=fill, anchor=anchor)

def generate_top_tracks_image(background_path, top_tracks_data, artist_name, textfont, textfont_bd, signature=True, output_path='./top_10_tracks_result.jpg'):
    background_size = [1080, 1920]
    background = Image.open(background_path).convert("RGB").resize(tuple(background_size))
    draw = ImageDraw.Draw(background)

    # Font setup
    # "arialbd.ttf"
    popularity_font = ImageFont.truetype(textfont, size=28)
    signature_font = ImageFont.truetype(textfont, size=20)

    # Draw title using auto-fit
    title_text = f"Top 10 Tracks of {artist_name}"
    draw_auto_fit_text(
        draw=draw,
        text=title_text,
        font_path=textfont_bd,
        max_width=background_size[0] - 100,
        max_size=64,
        min_size=32,
        position=(background_size[0] // 2, 80),
        fill="white",
        anchor="mm"  # center horizontally
    )

    # Track list
    start_y = 180
    spacing_y = 170
    image_size = (140, 140)
    border_size = 3

    for i, track in enumerate(top_tracks_data[:10]):
        rank = f"#{i + 1}"
        name_text = track['name']
        release_date = track['release_date'].replace('-', '/')
        album_text = f"Album: {track['album_name']} ({release_date})"
        popularity = 'Popularity: ' + str(track['popularity'])
        image_url = track['image_url']

        # Load and process album cover
        response = requests.get(image_url)
        album_img = Image.open(BytesIO(response.content)).resize(image_size)
        album_img_with_border = ImageOps.expand(album_img, border=border_size, fill='white')

        image_x = 160
        image_y = start_y + i * spacing_y
        background.paste(album_img_with_border, (image_x, image_y))

        text_x = image_x + image_size[0] + 2 * border_size + 20

        # Draw track rank
        if i >= 9:
            draw.text((text_x-10, image_y), rank, font=ImageFont.truetype(textfont_bd, size=30), fill="white")
        else:
            draw.text((text_x, image_y), rank, font=ImageFont.truetype(textfont_bd, size=30), fill="white")

        # Draw track name (auto-fit)
        draw_auto_fit_text(
            draw=draw,
            text=name_text,
            font_path=textfont_bd,
            max_width=background_size[0] - text_x - 80,
            max_size=30,
            min_size=18,
            position=(text_x + 60, image_y),
            fill="white"
        )

        # Draw album name (auto-fit)
        draw_auto_fit_text(
            draw=draw,
            text=album_text,
            font_path=textfont,
            max_width=background_size[0] - text_x - 80,
            max_size=28,
            min_size=16,
            position=(text_x + 60, image_y + 36),
            fill="white"
        )

        # Draw popularity (fixed font)
        draw.text((text_x + 60, image_y + 72), popularity, font=popularity_font, fill="white")

    # Signature
    if signature:
        signature_text = "@SCOTT HUANG"
        bbox = signature_font.getbbox(signature_text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        signature_x = (background.width - text_width) // 2
        signature_y = background.height - text_height - 20
        draw.text((signature_x, signature_y), signature_text, font=signature_font, fill="white")

    # Debug display
    plt.figure(figsize=(12, 16))
    plt.imshow(background)
    plt.axis('off')
    plt.tight_layout()
    plt.show()