"""
Pucho Ghar ke Baatein - MCP Tools Package
=========================================

This package contains all the MCP tools for the Puch AI system.
Each tool is organized in its own module for better maintainability.

Available Tools:
- Recipe Tool: Leftover to recipe suggestions
- Story Tool: Moral stories and bedtime tales  
- Poem Tool: Poetry generation in multiple languages
- Music Tool: Nostalgic music recommendations
- Food Location Tool: Restaurant and food place finder
"""

from .recipe_tool import get_recipe_suggestions
from .story_tool import get_story_content
from .poem_tool import get_poem_content
from .music_tool import get_music_recommendations
from .food_location_tool import get_food_locations

__all__ = [
    'get_recipe_suggestions',
    'get_story_content', 
    'get_poem_content',
    'get_music_recommendations',
    'get_food_locations'
] 