"""
Story Tool - Nani Ki Kahaniyan
===============================

Generates moral stories and bedtime tales for children, just like Nani used to tell.
Perfect for bedtime routine and teaching values to kids.
"""

from typing import Optional

async def get_story_content(age_group: Optional[str] = "children", moral_theme: Optional[str] = "honesty", language_preference: Optional[str] = "hinglish") -> dict:
    """
    Generates moral stories and bedtime tales for children, just like Nani used to tell.
    Perfect for bedtime routine and teaching values to kids.
    """
    stories_database = {
        "honesty": {
            "title": "Sach Bolne Wala Rajkumar / The Honest Prince",
            "story": """एक बार एक राज्य में एक छोटा राजकुमार रहता था। वह हमेशा सच बोलता था, चाहे उसे कितनी भी मुश्किल हो।

एक दिन उसने गलती से अपनी माँ का प्रिय गुलदस्ता तोड़ दिया। वह डर गया, लेकिन फिर भी उसने सच बता दिया।

"Mummy, maine galti se aapka favorite vase tod diya hai. I'm very sorry."

उसकी माँ को गुस्सा आने के बजाय खुशी हुई कि उसका बेटा इतना ईमानदार है।

Moral: सच बोलना हमेशा सही रास्ता है। Truth always wins in the end.""",
            "moral": "Honesty is the best policy / सच्चाई सबसे अच्छी नीति है",
            "age_group": "children",
            "duration": "5 minutes"
        },
        "kindness": {
            "title": "Pyaari Chidiya aur Sher / The Kind Bird and Lion",
            "story": """जंगल में एक छोटी चिड़िया रहती थी। वह बहुत दयालु थी और सबकी मदद करती थी।

एक दिन उसने देखा कि एक बड़ा शेर कांटे में फंसा है और बहुत परेशान है।

"Don't worry, Uncle Sher! Main aapki help karungi," चिड़िया ने कहा।

छोटी चिड़िया ने अपनी छोटी चोंच से धीरे-धीरे कांटा निकाल दिया।

शेर बहुत खुश हुआ और बोला, "Thank you, little friend! आज से तुम मेरी सबसे अच्छी दोस्त हो।"

Moral: छोटे हो या बड़े, सबकी मदद करनी चाहिए। Kindness knows no size.""",
            "moral": "Be kind to everyone / सभी के साथ दयालु बनें",
            "age_group": "children", 
            "duration": "5 minutes"
        },
        "perseverance": {
            "title": "Mehnat Karne Wali Chiti / The Hardworking Ant",
            "story": """गर्मियों में एक चींटी दिन भर मेहनत करती थी और अनाज इकट्ठा करती थी।

उसका दोस्त टिड्डा हमेशा गाना गाता और खेलता रहता था।

"Arre yaar, itna kaam kyun karti ho? Come and play with me!" टिड्डा बोला।

चींटी ने कहा, "Sardi aane wali hai, hume prepare karna chahiye."

जब ठंड आई, चींटी के पास खाना था लेकिन टिड्डा भूखा रह गया।

फिर चींटी ने अपना खाना अपने दोस्त के साथ share किया।

Moral: मेहनत का फल हमेशा मीठा होता है। Hard work always pays off.""",
            "moral": "Hard work and preparation are important / मेहनत और तैयारी जरूरी है",
            "age_group": "children",
            "duration": "6 minutes"
        }
    }
    
    # Select story based on theme
    selected_story = stories_database.get(moral_theme, stories_database["honesty"])
    
    return {
        "story_title": selected_story["title"],
        "story_content": selected_story["story"],
        "moral_lesson": selected_story["moral"],
        "age_group": age_group,
        "language_preference": language_preference,
        "estimated_duration": selected_story["duration"],
        "bedtime_tips": [
            "Read in a calm, soothing voice",
            "Pause to ask questions about the moral",
            "Let the child retell the story in their own words"
        ]
    } 