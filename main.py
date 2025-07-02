from fastmcp import FastMCP
import asyncio
from dotenv import load_dotenv
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from pydantic import BaseModel
import logging
import os
import json
from typing import List, Optional
from models import *
from router import MultilingualToolRouter
from tools import (
    get_recipe_suggestions,
    get_story_content,
    get_poem_content,
    get_music_recommendations,
    get_food_locations
)

load_dotenv()
TOKEN = os.getenv("TOKEN")
MY_NUMBER = os.getenv("MY_NUMBER")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize multilingual router
router = MultilingualToolRouter()

class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None

# Rich descriptions for all tools
LeftoverChefDescription = RichToolDescription(
    description="👨‍🍳 Leftover Chef - Creative recipe suggestions from whatever ingredients you have at home. Perfect for Indian household cooking with practical, family-friendly recipes.",
    use_when="Use when users ask about cooking with leftovers, ingredients they have, recipe suggestions, or cooking help. Works with inputs like 'ghar mein chawal hai kya banau', 'leftover recipe', 'cooking help', or ingredient lists.",
    side_effects="Provides creative recipe suggestions, cooking instructions, and ingredient substitutions based on available leftovers."
)

NaniKahaniyaDescription = RichToolDescription(
    description="📚 Nani Ki Kahaniyan - Traditional bedtime stories and moral tales in Hindi/English. Perfect for children's storytelling with cultural values and life lessons.",
    use_when="Use when users request stories, tales, moral lessons, bedtime stories, or children's content. Works with inputs like 'story sunao', 'kahani batao', 'bedtime story', or requests for moral tales.",
    side_effects="Generates heartwarming stories with moral lessons, cultural context, and age-appropriate content."
)

PoemGeneratorDescription = RichToolDescription(
    description="🎭 Poem Generator - Beautiful Hindi and English poetry on various themes like love, nature, friendship, and life. Culturally rich verses with emotional depth.",
    use_when="Use when users ask for poems, poetry, verses, or creative writing. Works with inputs like 'poem sunao', 'poetry chahiye', 'kavita', or requests for specific themes.",
    side_effects="Creates original poems with cultural richness, emotional depth, and artistic expression in Hindi, English, or Hinglish."
)

VividhBhartiDescription = RichToolDescription(
    description="🎵 Vividh Bharti Jukebox - Nostalgic music recommendations from the golden era of Indian cinema (1950s-1980s). Perfect for reliving classic Bollywood memories.",
    use_when="Use when users ask for music, songs, nostalgia, classic Bollywood, or entertainment. Works with inputs like 'purane gaane', 'music sunao', 'nostalgic songs', or era-specific requests.",
    side_effects="Provides curated music recommendations with historical context, artist information, and nostalgic value."
)

FoodLocatorDescription = RichToolDescription(
    description="🍽️ Food Locator - Find nearby restaurants, street food, and dining options based on location, budget, and cuisine preferences. Perfect for discovering local food gems.",
    use_when="Use when users ask about restaurants, food places, dining options, or local food recommendations. Works with inputs like 'nearby restaurant', 'food places', 'dining options', or location-based food queries.",
    side_effects="Provides restaurant recommendations, location details, cuisine information, and budget-appropriate dining suggestions."
)

# Intelligent Assistant and Diagnostic tools
IntelligentAssistantDescription = RichToolDescription(
    description="🤖 Main AI assistant that intelligently understands user queries in Hindi, English, or Hinglish and automatically routes to the most appropriate specialized tool. Handles all types of user requests including recipes, stories, poems, music, and food recommendations.",
    use_when="Use this tool for ANY user query or conversation in Hindi/English/Hinglish when you want automatic intelligent routing. This tool handles ALL user inputs and selects the best specialized tool automatically.",
    side_effects="Analyzes user input, automatically selects the best internal specialized tool, and returns results with routing information and confidence scores."
)

RouteInputDescription = RichToolDescription(
    description="🔍 Diagnostic tool that demonstrates multilingual tool routing capabilities by showing how user input gets analyzed and routed to appropriate tools.",
    use_when="Use this tool when you want to understand or debug how the routing system works, see confidence scores, or analyze the decision-making process for user inputs in Hindi/English/Hinglish.",
    side_effects="Returns routing decision metadata including confidence scores, selected tool, reasoning, and language detection without executing the actual tool."
)

EvaluateRoutingDescription = RichToolDescription(
    description="📊 Evaluation tool that measures the accuracy of multilingual tool routing using comprehensive test cases across different languages and intents.",
    use_when="Use this tool when you want to assess the performance of the routing system, get accuracy metrics, or understand how well the system handles different types of user inputs.",
    side_effects="Runs comprehensive evaluation tests and returns detailed accuracy metrics including per-tool and per-language performance statistics."
)

class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="unknown", scopes=[], expires_at=None)
        return None

mcp = FastMCP("Pucho Ghar ke Baatein MCP Tools", auth=SimpleBearerAuthProvider(TOKEN))

@mcp.tool
async def validate() -> str:
    """
    NOTE: This tool must be present in an MCP server used by puch.
    """
    return MY_NUMBER

@mcp.tool(description=RouteInputDescription.description())
async def route_input(user_input: str) -> dict:
    """
    Route user input to appropriate tool and return routing decision with confidence metrics.
    This demonstrates multilingual tool routing capabilities.
    """
    decision = router.route_to_tool(user_input)
    return {
        "selected_tool": decision.selected_tool,
        "confidence_score": decision.confidence_score,
        "reasoning": decision.reasoning,
        "language_detected": decision.language_detected.value,
        "semantic_similarity": decision.semantic_similarity
    }

async def process_user_request(user_input: str) -> dict:
    """
    🤖 CORE INTELLIGENT ASSISTANT - Intelligently routes user input and executes the appropriate tool
    
    This is the core logic for Puch chat interface. It:
    1. Analyzes user input in Hindi/English/Hinglish 
    2. Routes to the best tool using hybrid ML classifier
    3. Executes the tool and returns results
    4. Handles edge cases gracefully
    
    Perfect for: "Ghar mein chawal hai kya banau?", "Story sunao", "Purane gaane", etc.
    """
    
    # Step 1: Route the input using our hybrid system
    decision = router.route_to_tool(user_input)
    
    # Step 2: Handle clarification case
    if decision.selected_tool == "clarification_needed":
        return {
            "status": "clarification_needed",
            "message": "मुझे समझ नहीं आया। कृपया स्पष्ट करें कि आप क्या चाहते हैं। / I didn't understand. Please clarify what you want.",
            "suggestions": [
                "खाना बनाने के लिए - 'leftover se kya banau' या 'recipe batao'",
                "कहानी के लिए - 'story sunao' या 'bacchon ki kahani'", 
                "कविता के लिए - 'poem sunao' या 'poetry chahiye'",
                "संगीत के लिए - 'purane gaane' या 'nostalgic music'",
                "खाने की जगह के लिए - 'nearby restaurant' या 'food places'"
            ],
            "routing_info": {
                "confidence": decision.confidence_score,
                "language": decision.language_detected.value,
                "reasoning": decision.reasoning
            }
        }
    
    # Step 3: Execute the appropriate tool based on routing decision
    try:
        result = None
        
        if decision.selected_tool == "leftover_chef":
            # Extract ingredients from input or use defaults
            # Simple keyword extraction for demo
            common_leftovers = ["rice", "dal", "roti", "sabzi", "bread", "chawal", "दाल", "रोटी"]
            found_leftovers = [item for item in common_leftovers if item.lower() in user_input.lower()]
            if not found_leftovers:
                found_leftovers = ["mixed ingredients"]  # Default
            
            result = await internal_leftover_chef(leftovers=found_leftovers)
            
        elif decision.selected_tool == "nani_kahaniyan":
            # Extract story preferences from input
            moral_theme = "honesty"
            if "kindness" in user_input.lower() or "दयालु" in user_input:
                moral_theme = "kindness"
            elif "hard work" in user_input.lower() or "mehnat" in user_input.lower():
                moral_theme = "perseverance"
                
            result = await internal_nani_kahaniyan(moral_theme=moral_theme)
            
        elif decision.selected_tool == "poem_generator":
            # Extract poem theme from input
            theme = "love"
            if "nature" in user_input.lower() or "prakriti" in user_input.lower():
                theme = "nature"
            elif "friend" in user_input.lower() or "dost" in user_input.lower():
                theme = "friendship"
                
            result = await internal_poem_generator(theme=theme)
            
        elif decision.selected_tool == "vividh_bharti":
            # Extract music preferences from input
            era = "1960s"  # Default to golden era
            mood = "nostalgic"
            if "1950" in user_input:
                era = "1950s"
            elif "1970" in user_input:
                era = "1970s"
                
            result = await internal_vividh_bharti(era=era, mood=mood)
            
        elif decision.selected_tool == "food_locator":
            # Extract location preferences
            food_type = "all"
            budget = "moderate"
            if "cheap" in user_input.lower() or "budget" in user_input.lower():
                budget = "budget"
            elif "expensive" in user_input.lower() or "fine dining" in user_input.lower():
                budget = "expensive"
                
            result = await internal_food_locator(food_type=food_type, budget_range=budget)
        
        # Step 4: Return enhanced response with routing information
        return {
            "status": "success",
            "user_query": user_input,
            "tool_result": result,
            "routing_info": {
                "selected_tool": decision.selected_tool,
                "confidence": decision.confidence_score,
                "method": decision.routing_method,
                "language": decision.language_detected.value,
                "reasoning": decision.reasoning
            },
            "ai_message": f"✅ समझ गया! मैंने आपके लिए {decision.selected_tool} का उपयोग किया है। / Got it! I used {decision.selected_tool} for you."
        }
        
    except Exception as e:
        # Error handling
        return {
            "status": "error", 
            "message": f"माफ करें, कुछ गलत हुआ। / Sorry, something went wrong: {str(e)}",
            "user_query": user_input,
            "routing_info": {
                "selected_tool": decision.selected_tool,
                "confidence": decision.confidence_score,
                "language": decision.language_detected.value
            }
                 }

@mcp.tool(description=IntelligentAssistantDescription.model_dump_json())
async def intelligent_assistant(user_input: str) -> dict:
    """
    🤖 MAIN PUCH AI ASSISTANT - Intelligently routes user input and executes the appropriate tool
    
    This is the primary endpoint for Puch chat interface. It:
    1. Analyzes user input in Hindi/English/Hinglish 
    2. Routes to the best tool using hybrid ML classifier
    3. Executes the tool and returns results
    4. Handles edge cases gracefully
    
    Perfect for: "Ghar mein chawal hai kya banau?", "Story sunao", "Purane gaane", etc.
    """
    return await process_user_request(user_input)

@mcp.tool(description=EvaluateRoutingDescription.model_dump_json())
async def evaluate_routing_accuracy() -> dict:
    """
    Evaluate the accuracy of multilingual tool routing using comprehensive test cases.
    Returns detailed accuracy metrics including per-tool and per-language performance.
    """
    test_cases = router.get_test_dataset()
    metrics = router.evaluate_accuracy(test_cases)
    
    return {
        "overall_accuracy": metrics.accuracy,
        "total_tests": metrics.total_tests,
        "correct_predictions": metrics.correct_predictions,
        "average_confidence": metrics.avg_confidence,
        "precision_per_tool": metrics.precision_per_tool,
        "recall_per_tool": metrics.recall_per_tool,
        "language_accuracy": {lang.value: acc for lang, acc in metrics.language_accuracy.items()},
        "confusion_matrix": metrics.confusion_matrix
    }

# Individual MCP Tools - Now exposed as standalone tools

@mcp.tool(description=LeftoverChefDescription.model_dump_json())
async def leftover_chef(leftovers: List[str], cuisine_type: Optional[str] = "Indian", dietary_preferences: Optional[List[str]] = None) -> dict:
    """
    👨‍🍳 Get creative recipe suggestions from your leftover ingredients.
    Perfect for Indian household cooking with practical, family-friendly recipes.
    """
    return await get_recipe_suggestions(leftovers, cuisine_type, dietary_preferences)

@mcp.tool(description=NaniKahaniyaDescription.model_dump_json())
async def nani_kahaniyan(age_group: Optional[str] = "children", moral_theme: Optional[str] = "honesty", language_preference: Optional[str] = "hinglish") -> dict:
    """
    📚 Traditional bedtime stories and moral tales in Hindi/English.
    Perfect for children's storytelling with cultural values and life lessons.
    """
    return await get_story_content(age_group, moral_theme, language_preference)

@mcp.tool(description=PoemGeneratorDescription.model_dump_json())
async def poem_generator(theme: Optional[str] = "love", style: Optional[str] = "romantic", language_preference: Optional[str] = "hinglish") -> dict:
    """
    🎭 Beautiful Hindi and English poetry on various themes.
    Creates culturally rich verses with emotional depth.
    """
    return await get_poem_content(theme, style, language_preference)

@mcp.tool(description=VividhBhartiDescription.model_dump_json())
async def vividh_bharti(era: Optional[str] = "1900s", mood: Optional[str] = "nostalgic", artist_preference: Optional[str] = None) -> dict:
    """
    🎵 Nostalgic music recommendations from the golden era of Indian cinema.
    Perfect for reliving classic Bollywood memories.
    """
    return await get_music_recommendations(era, mood, artist_preference)

@mcp.tool(description=FoodLocatorDescription.model_dump_json())
async def food_locator(latitude: Optional[float] = None, longitude: Optional[float] = None, food_type: Optional[str] = "all", budget_range: Optional[str] = "moderate") -> dict:
    """
    🍽️ Find nearby restaurants, street food, and dining options.
    Perfect for discovering local food gems based on location and preferences.
    """
    return await get_food_locations(latitude, longitude, food_type, budget_range)

# Internal helper functions for intelligent_assistant routing
async def internal_leftover_chef(leftovers: List[str], cuisine_type: Optional[str] = "Indian", dietary_preferences: Optional[List[str]] = None) -> dict:
    """Internal function for recipe suggestions"""
    return await get_recipe_suggestions(leftovers, cuisine_type, dietary_preferences)

async def internal_nani_kahaniyan(age_group: Optional[str] = "children", moral_theme: Optional[str] = "honesty", language_preference: Optional[str] = "hinglish") -> dict:
    """Internal function for story generation"""
    return await get_story_content(age_group, moral_theme, language_preference)

async def internal_poem_generator(theme: Optional[str] = "love", style: Optional[str] = "romantic", language_preference: Optional[str] = "hinglish") -> dict:
    """Internal function for poem generation"""
    return await get_poem_content(theme, style, language_preference)

async def internal_vividh_bharti(era: Optional[str] = "1900s", mood: Optional[str] = "nostalgic", artist_preference: Optional[str] = None) -> dict:
    """Internal function for music recommendations"""
    return await get_music_recommendations(era, mood, artist_preference)

async def internal_food_locator(latitude: Optional[float] = None, longitude: Optional[float] = None, food_type: Optional[str] = "all", budget_range: Optional[str] = "moderate") -> dict:
    """Internal function for food location suggestions"""
    return await get_food_locations(latitude, longitude, food_type, budget_range)

async def main():
    logger.info("Starting Pucho Ghar ke Baatein MCP Server with multilingual tool routing...")
    logger.info("🍳 Individual Tools: leftover_chef, nani_kahaniyan, poem_generator, vividh_bharti, food_locator")
    logger.info("🤖 Intelligent Router: intelligent_assistant (automatically routes to best tool)")
    logger.info("📊 Diagnostic tools: route_input, evaluate_routing_accuracy")
    logger.info("✅ All tools exposed with RichToolDescription for optimal AI selection")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8085)

if __name__ == "__main__":
    asyncio.run(main())