from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum
import time

class ToolIntent(str, Enum):
    RECIPE_SUGGESTION = "recipe_suggestion"
    STORY_TELLING = "story_telling" 
    POEM_GENERATION = "poem_generation"
    MUSIC_RECOMMENDATION = "music_recommendation"
    FOOD_LOCATION = "food_location"

class Language(str, Enum):
    HINDI = "hindi"
    ENGLISH = "english"
    HINGLISH = "hinglish"
    MIXED = "mixed"

class ToolMetadata(BaseModel):
    name: str
    intent: ToolIntent
    description_en: str
    description_hi: str
    keywords_en: List[str]
    keywords_hi: List[str]
    keywords_hinglish: List[str]
    semantic_context: List[str]
    confidence_threshold: float = 0.7

class RecipeRequest(BaseModel):
    leftovers: List[str]
    dietary_preferences: Optional[List[str]] = None
    cuisine_type: Optional[str] = None

class StoryRequest(BaseModel):
    age_group: Optional[str] = "children"
    moral_theme: Optional[str] = None
    language_preference: Optional[str] = "hinglish"

class PoemRequest(BaseModel):
    theme: Optional[str] = None
    style: Optional[str] = "romantic"
    language_preference: Optional[str] = "hinglish"

class MusicRequest(BaseModel):
    era: Optional[str] = "1900s"
    mood: Optional[str] = "nostalgic"
    artist_preference: Optional[str] = None

class LocationRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    food_type: Optional[str] = None
    budget_range: Optional[str] = None

class RouteDecision(BaseModel):
    selected_tool: str
    confidence_score: float
    reasoning: str
    language_detected: Language
    semantic_similarity: float
    timestamp: float = time.time()
    # New fields for hybrid routing
    routing_method: str = "semantic"  # "classifier" or "semantic" or "fallback"
    classifier_confidence: Optional[float] = None
    predicted_intent: Optional[ToolIntent] = None

class AccuracyTestCase(BaseModel):
    input_text: str
    expected_tool: str
    expected_intent: ToolIntent
    language: Language
    description: str

class EvaluationResult(BaseModel):
    test_case: AccuracyTestCase
    actual_tool: str
    confidence_score: float
    is_correct: bool
    reasoning: str
    execution_time: float

class AccuracyMetrics(BaseModel):
    total_tests: int
    correct_predictions: int
    accuracy: float
    precision_per_tool: Dict[str, float]
    recall_per_tool: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    avg_confidence: float
    language_accuracy: Dict[Language, float]

