"""
Recipe Tool - Leftover Chef
============================

Recommends recipes based on leftover food ingredients available at home.
Perfect for reducing food waste and creating delicious meals from what you have.
"""

from typing import List, Optional

async def get_recipe_suggestions(leftovers: List[str], cuisine_type: Optional[str] = "Indian", dietary_preferences: Optional[List[str]] = None) -> dict:
    """
    Recommends recipes based on leftover food ingredients available at home.
    Perfect for reducing food waste and creating delicious meals from what you have.
    """
    # Sample recipe recommendations (in real implementation, this would use a recipe API or database)
    recipe_database = {
        ("rice", "dal"): {
            "name": "Dal Chawal Khichdi",
            "description": "Comfort food made with rice and dal, perfect for a quick meal",
            "ingredients": ["rice", "dal", "turmeric", "salt", "ghee"],
            "instructions": "1. Wash rice and dal together\n2. Add turmeric and salt\n3. Cook with 3 cups water\n4. Garnish with ghee",
            "prep_time": "20 minutes",
            "difficulty": "Easy"
        },
        ("roti", "sabzi"): {
            "name": "Roti Roll",
            "description": "Transform leftover roti and sabzi into a delicious wrap",
            "ingredients": ["roti", "leftover sabzi", "onions", "chutney"],
            "instructions": "1. Heat the roti\n2. Add sabzi in center\n3. Add onions and chutney\n4. Roll tightly",
            "prep_time": "10 minutes", 
            "difficulty": "Easy"
        },
        ("bread", "vegetables"): {
            "name": "Bread Upma",
            "description": "South Indian style bread upma with vegetables",
            "ingredients": ["bread", "vegetables", "mustard seeds", "curry leaves"],
            "instructions": "1. Cut bread into pieces\n2. Sauté vegetables\n3. Add bread and mix\n4. Season with spices",
            "prep_time": "15 minutes",
            "difficulty": "Medium"
        }
    }
    
    # Find matching recipes
    leftover_set = set(item.lower() for item in leftovers)
    matching_recipes = []
    
    for ingredients, recipe in recipe_database.items():
        if leftover_set.intersection(set(ingredients)):
            matching_recipes.append(recipe)
    
    # If no exact match, provide general suggestions
    if not matching_recipes:
        matching_recipes = [{
            "name": "Creative Leftover Mix",
            "description": f"Try making a stir-fry or curry with your {', '.join(leftovers)}",
            "ingredients": leftovers + ["spices", "oil", "onions"],
            "instructions": "1. Heat oil in pan\n2. Add onions and spices\n3. Add leftovers and mix\n4. Cook until heated through",
            "prep_time": "15 minutes",
            "difficulty": "Easy"
        }]
    
    return {
        "leftovers_provided": leftovers,
        "cuisine_type": cuisine_type,
        "dietary_preferences": dietary_preferences,
        "recommended_recipes": matching_recipes[:3],  # Return top 3 matches
        "tips": [
            "Always check if leftovers are fresh before cooking",
            "Add fresh spices to enhance flavors",
            "Consider mixing leftovers with fresh ingredients"
        ]
    } 