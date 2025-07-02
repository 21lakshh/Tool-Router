"""
Poem Tool - Poetry Generator
=============================

Creates beautiful poems and verses in Hindi, English, or Hinglish.
Perfect for expressing emotions and sharing beautiful poetry.
"""

from typing import Optional

async def get_poem_content(theme: Optional[str] = "love", style: Optional[str] = "romantic", language_preference: Optional[str] = "hinglish") -> dict:
    """
    Creates beautiful poems and verses in Hindi, English, or Hinglish.
    Perfect for expressing emotions and sharing beautiful poetry.
    """
    poems_database = {
        "love": {
            "romantic": {
                "title": "Dil Ki Baat / Heart's Voice",
                "poem": """तेरी आँखों में छुपा है प्यार का एक समंदर,
Your smile brings sunshine to my darkest days.

मेरा दिल कहता है, तू है मेरी जिंदगी का सहारा,
In this beautiful journey, you're my guiding star always.

हर सुबह तेरे ख्यालों से शुरू होती है,
Every evening ends with dreams of you and me.

Love ke इस मौसम में, हम साथ चलेंगे हमेशा,
Together forever, that's our beautiful destiny.""",
                "style": "romantic",
                "language": "hinglish"
            }
        },
        "nature": {
            "descriptive": {
                "title": "Prakriti Ka Geet / Song of Nature", 
                "poem": """हरे-भरे पेड़ों की छांव में,
Green trees dancing in the morning breeze,

चिड़ियों के मधुर गीत सुनाई देते हैं,
Sweet melodies fill the air with peace.

सूरज की किरणें फूलों को छूती हैं,
Sunlight kisses every blooming flower,

प्रकृति का यह नज़ारा है कितना सुंदर,
Nature's beauty shows divine power.""",
                "style": "descriptive",
                "language": "hinglish"
            }
        },
        "friendship": {
            "cheerful": {
                "title": "Dosti Ka Rang / Colors of Friendship",
                "poem": """दोस्तों के साथ हर दिन है खुशियों भरा,
With friends by side, life's a joyful ride.

हंसी-मज़ाक और प्यार से भरा है यह रिश्ता,
True friendship is life's most precious guide.

मुश्किल वक्त में साथ देते हैं ये,
In tough times, they never leave you alone,

दोस्ती का यह प्यार है अनमोल और सच्चा,
Real friends make your heart their home.""",
                "style": "cheerful",
                "language": "hinglish"
            }
        }
    }
    
    # Select poem based on theme and style
    selected_poem = poems_database.get(theme, {}).get(style)
    if not selected_poem:
        selected_poem = poems_database["love"]["romantic"]
    
    return {
        "poem_title": selected_poem["title"],
        "poem_content": selected_poem["poem"],
        "theme": theme,
        "style": style,
        "language_preference": language_preference,
        "recitation_tips": [
            "Read with emotion and proper pauses",
            "Emphasize the rhythm and rhyme",
            "Let the words flow naturally"
        ],
        "sharing_suggestion": "Perfect for sharing with loved ones or on social media"
    } 