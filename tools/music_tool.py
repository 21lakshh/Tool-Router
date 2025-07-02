"""
Music Tool - Vividh Bharti Jukebox
===================================

Recommends nostalgic 1900s classic Indian songs and music from the golden era.
Perfect for bringing back memories and enjoying timeless melodies.
"""

import random
from typing import Optional

async def get_music_recommendations(era: Optional[str] = "1900s", mood: Optional[str] = "nostalgic", artist_preference: Optional[str] = None) -> dict:
    """
    Recommends nostalgic 1900s classic Indian songs and music from the golden era.
    Perfect for bringing back memories and enjoying timeless melodies.
    """
    classic_songs_database = {
        "1950s": [
            {
                "title": "Aayega Aanewala",
                "movie": "Mahal (1949)",
                "singer": "Lata Mangeshkar",
                "music_director": "Khemchand Prakash",
                "mood": "mysterious",
                "description": "A hauntingly beautiful song that became an instant classic"
            },
            {
                "title": "Sare Jahan Se Achha",
                "movie": "Patriotic Song",
                "singer": "Various Artists",
                "music_director": "Traditional",
                "mood": "patriotic",
                "description": "The most beloved patriotic song of India"
            }
        ],
        "1960s": [
            {
                "title": "Lag Ja Gale",
                "movie": "Woh Kaun Thi (1964)",
                "singer": "Lata Mangeshkar",
                "music_director": "Madan Mohan",
                "mood": "romantic",
                "description": "One of the most romantic songs ever created in Bollywood"
            },
            {
                "title": "Kahin Door Jab Din Dhal Jaye",
                "movie": "Anand (1971)",
                "singer": "Mukesh",
                "music_director": "Salil Chowdhury",
                "mood": "nostalgic",
                "description": "A deeply moving song about life's journey"
            }
        ],
        "1970s": [
            {
                "title": "Tere Bina Zindagi Se",
                "movie": "Aandhi (1975)",
                "singer": "Lata Mangeshkar, Kishore Kumar",
                "music_director": "R.D. Burman",
                "mood": "romantic",
                "description": "A timeless duet about incomplete love"
            },
            {
                "title": "Rimjhim Gire Sawan",
                "movie": "Manzil (1979)",
                "singer": "Lata Mangeshkar, Kishore Kumar",
                "music_director": "R.D. Burman", 
                "mood": "monsoon",
                "description": "The perfect song for rainy days"
            }
        ]
    }
    
    # Select songs based on era and mood
    era_songs = classic_songs_database.get(era, classic_songs_database["1960s"])
    
    if mood:
        filtered_songs = [song for song in era_songs if song["mood"] == mood]
        if filtered_songs:
            era_songs = filtered_songs
    
    if artist_preference:
        artist_songs = [song for song in era_songs if artist_preference.lower() in song["singer"].lower()]
        if artist_songs:
            era_songs = artist_songs
    
    # Shuffle and select random songs
    selected_songs = random.sample(era_songs, min(3, len(era_songs)))
    
    return {
        "era": era,
        "mood": mood,
        "artist_preference": artist_preference,
        "recommended_songs": selected_songs,
        "listening_tips": [
            "Best enjoyed with a cup of tea in the evening",
            "Close your eyes and let the melodies transport you",
            "Share with family to create bonding moments"
        ],
        "trivia": "These songs represent the golden era of Indian cinema music, when melody was the king and lyrics touched the soul."
    } 