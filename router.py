from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import re
import time
from models import *
import logging
import os

logger = logging.getLogger(__name__)

# Lazy import for intent classifier to avoid circular imports
_intent_classifier = None

def get_intent_classifier():
    """Lazy loading of intent classifier"""
    global _intent_classifier
    if _intent_classifier is None:
        try:
            from intent_classifier import MultilingualIntentClassifier
            _intent_classifier = MultilingualIntentClassifier()
            if os.path.exists("./intent_model"):
                _intent_classifier.load_model("./intent_model")
                logger.info("✅ Intent classifier loaded successfully")
            else:
                logger.warning("⚠️  Intent model not found, will use semantic-only routing")
                _intent_classifier = None
        except Exception as e:
            logger.warning(f"⚠️  Failed to load intent classifier: {e}")
            _intent_classifier = None
    return _intent_classifier

class MultilingualToolRouter:
    def __init__(self):
        # Use a multilingual sentence transformer model
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.tools_metadata = self._initialize_tool_metadata()
        self.evaluation_history: List[EvaluationResult] = []
        self.route_history: List[RouteDecision] = []
        
        # Pre-compute embeddings for tool descriptions
        self._precompute_tool_embeddings()
    
    def _initialize_tool_metadata(self) -> Dict[str, ToolMetadata]:
        """Initialize metadata for all available tools"""
        return {
            "leftover_chef": ToolMetadata(
                name="leftover_chef",
                intent=ToolIntent.RECIPE_SUGGESTION,
                description_en="Recommends recipes based on leftover food ingredients available at home",
                description_hi="घर में बचे हुए खाने से रेसिपी बनाने की सलाह देता है",
                keywords_en=["recipe", "cook", "food", "ingredients", "leftover", "meal", "dish"],
                keywords_hi=["रेसिपी", "खाना", "बनाना", "सामग्री", "बचा हुआ", "भोजन"],
                keywords_hinglish=[
                    "recipe batao", "khana banana", "leftover se kya banau", "cooking tips",
                    "ghar mein kya hai", "kuch banana hai", "leftover food se", "recipe suggest",
                    "khana banane ka tarika", "kya cook karu", "bacha hua khana", "meal idea",
                    "cooking help", "recipe chahiye", "dish banao", "khana ready karo",
                    "dal chawal se kya banau", "roti sabzi leftover", "khane ka jugaad"
                ],
                semantic_context=["cooking guidance", "ingredient utilization", "meal preparation", "food waste reduction", "ghar ka khana", "leftover jugaad"],
                confidence_threshold=0.3
            ),
            "nani_kahaniyan": ToolMetadata(
                name="nani_kahaniyan",
                intent=ToolIntent.STORY_TELLING,
                description_en="Generates moral stories and bedtime tales for children",
                description_hi="बच्चों के लिए नैतिक कहानियां और सोने से पहले की कहानियां बनाता है",
                keywords_en=["story", "tale", "moral", "bedtime", "children", "narrative"],
                keywords_hi=["कहानी", "किस्सा", "नैतिक", "सोने से पहले", "बच्चे"],
                keywords_hinglish=[
                    "story sunao", "kahani batao", "bedtime story", "bacchon ke liye",
                    "moral story", "nani ki kahani", "sone se pehle", "bachon ko story",
                    "kahani suna do", "story time", "tale sunao", "good story",
                    "moral wali kahani", "kids story", "bacche story", "kahani chahiye",
                    "interesting story", "story with moral", "sunane ke liye kahani"
                ],
                semantic_context=["storytelling", "moral education", "children entertainment", "bedtime routine", "nani ki yaadein", "bachpan ki kahaniyan"],
                confidence_threshold=0.3
            ),
            "poem_generator": ToolMetadata(
                name="poem_generator",
                intent=ToolIntent.POEM_GENERATION,
                description_en="Creates beautiful poems and verses in Hindi and English",
                description_hi="हिंदी और अंग्रेजी में सुंदर कविताएं और छंद बनाता है",
                keywords_en=["poem", "poetry", "verse", "rhyme", "literature"],
                keywords_hi=["कविता", "शायरी", "छंद", "तुकबंदी", "साहित्य"],
                keywords_hinglish=[
                    "kavita sunao", "poetry batao", "achhi si poem", "koi poem",
                    "shayari sunao", "romantic poetry", "love poem", "kavita likhkar",
                    "poem create karo", "beautiful poetry", "heart touching poem",
                    "emotional kavita", "poetry suggest", "verse sunao", "rhyme banao",
                    "poetry chahiye", "poem sunane ka", "kavita ka mood"
                ],
                semantic_context=["creative writing", "artistic expression", "emotional expression", "literary creation", "dil ki baat", "feelings poetry"],
                confidence_threshold=0.3
            ),
            "vividh_bharti": ToolMetadata(
                name="vividh_bharti",
                intent=ToolIntent.MUSIC_RECOMMENDATION,
                description_en="Recommends nostalgic 1900s classic Indian songs and music",
                description_hi="1900 के दशक के पुराने भारतीय गाने और संगीत की सिफारिश करता है",
                keywords_en=["music", "songs", "classic", "old", "nostalgic", "1900s", "vintage"],
                keywords_hi=["संगीत", "गाने", "पुराने", "क्लासिक", "नॉस्टेल्जिक"],
                keywords_hinglish=[
                    "purane gaane", "old songs", "classic music", "nostalgic songs",
                    "gaane recommend karo", "music batao", "retro songs", "vintage music",
                    "old bollywood", "classic hits", "gaane sunao", "music suggest",
                    "nostalgic feeling", "old melodies", "gaane chahiye", "classic tracks",
                    "purane zamane ke gaane", "golden era songs", "evergreen music"
                ],
                semantic_context=["music recommendation", "nostalgia", "classic entertainment", "vintage music", "purane din", "golden memories"],
                confidence_threshold=0.3
            ),
            "food_locator": ToolMetadata(
                name="food_locator",
                intent=ToolIntent.FOOD_LOCATION,
                description_en="Suggests good food places and restaurants near current location",
                description_hi="वर्तमान स्थान के पास अच्छे खाने की जगह और रेस्टोरेंट सुझाता है",
                keywords_en=["restaurant", "food", "nearby", "location", "eat", "dining"],
                keywords_hi=["रेस्टोरेंट", "खाना", "पास में", "स्थान", "भोजन"],
                keywords_hinglish=[
                    "paas mein khana", "nearby restaurant", "kahan khana milega", "food places",
                    "yahan ke paas dhaba", "restaurant batao", "food location", "khane ki jagah",
                    "nearby food", "koi achha restaurant", "paas ka dhaba", "dining options",
                    "food delivery", "restaurant suggest", "khana order karna", "food spots",
                    "yahan ka food scene", "local restaurant", "food hunt", "achha khana kahan"
                ],
                semantic_context=["location services", "dining recommendations", "local food discovery", "restaurant finder", "khana dhundna", "food exploration"],
                confidence_threshold=0.3
            )
        }
    
    def _precompute_tool_embeddings(self):
        """Pre-compute embeddings for all tool descriptions and keywords"""
        self.tool_embeddings = {}
        
        for tool_name, metadata in self.tools_metadata.items():
            # Combine all descriptions and keywords for comprehensive matching
            # Give extra weight to Hinglish keywords by repeating them
            hinglish_boost = " ".join(metadata.keywords_hinglish * 2)  # Double weight for Hinglish
            
            combined_text = f"{metadata.description_en} {metadata.description_hi} "
            combined_text += " ".join(metadata.keywords_en + metadata.keywords_hi)
            combined_text += f" {hinglish_boost} "  # Enhanced Hinglish presence
            combined_text += " " + " ".join(metadata.semantic_context)
            
            # Add common Hinglish phrases that might appear for this tool
            hinglish_context = []
            if tool_name == "leftover_chef":
                hinglish_context = ["ghar mein khana", "leftover se jugaad", "kya banau", "recipe chahiye"]
            elif tool_name == "nani_kahaniyan":
                hinglish_context = ["bacchon ko story", "kahani sunao", "moral story chahiye", "bedtime tale"]
            elif tool_name == "poem_generator":
                hinglish_context = ["poetry sunao", "kavita chahiye", "poem likho", "shayari batao"]
            elif tool_name == "vividh_bharti":
                hinglish_context = ["purane gaane sunao", "old music chahiye", "nostalgic songs", "classic tracks"]
            elif tool_name == "food_locator":
                hinglish_context = ["yahan khana kahan", "nearby dhaba", "restaurant batao", "food places"]
            
            combined_text += " " + " ".join(hinglish_context)
            
            self.tool_embeddings[tool_name] = self.model.encode(combined_text)
    
    def detect_language(self, text: str) -> Language:
        """Detect the primary language of input text"""
        # Expanded Hinglish words (Roman Hindi and common code-switching)
        hinglish_words = [
            # Basic words
            'ghar', 'mein', 'hai', 'kuch', 'batao', 'karo', 'yaar', 'paas', 'koi', 'achha', 'sunao', 'bacchon', 'gaane', 'purane',
            # Action words
            'banana', 'khana', 'recipe', 'story', 'kavita', 'poem', 'dhaba', 'restaurant', 'nearby',
            # Common expressions
            'kya', 'hai', 'se', 'ka', 'ki', 'ko', 'mera', 'tera', 'aur', 'bhi', 'toh', 'wala', 'wali',
            # Question words
            'kahan', 'kaise', 'kyun', 'kab', 'kitna', 'kaun',
            # Food related
            'dal', 'chawal', 'roti', 'sabzi', 'leftover', 'bacha', 'hua',
            # Entertainment
            'music', 'songs', 'kahani', 'poetry', 'shayari',
            # Common particles
            'bhi', 'toh', 'na', 'ho', 'kar', 'ke', 'liye', 'chahiye', 'milega',
            # Expressions
            'achhi', 'accha', 'bura', 'burra', 'bahut', 'thoda', 'zyada', 'kam'
        ]
        
        text_lower = text.lower()
        
        # Count different character types
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text.replace(' ', ''))
        
        # Count Hinglish words more sophisticated way
        words_in_text = text_lower.split()
        hinglish_word_count = sum(1 for word in words_in_text if any(hw in word for hw in hinglish_words))
        
        # Check for common Hinglish patterns
        hinglish_patterns = [
            r'\b(kya|koi|kuch|kahan|kaise|kyun)\b',  # Question words
            r'\b(hai|hain|tha|thi|the)\b',           # Verbs
            r'\b(aur|bhi|toh|wala|wali)\b',          # Connectors
            r'\b(mein|se|ka|ki|ko|ke|liye)\b',       # Prepositions
            r'\b(batao|karo|sunao|dena|lena)\b'      # Commands
        ]
        
        pattern_matches = sum(1 for pattern in hinglish_patterns if re.search(pattern, text_lower))
        
        if total_chars == 0:
            return Language.ENGLISH
        
        hindi_ratio = hindi_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # Enhanced Hinglish detection
        is_hinglish = (
            hinglish_word_count > 0 or                               # Contains Hinglish words
            pattern_matches > 0 or                                   # Matches Hinglish patterns
            (hindi_ratio > 0.05 and english_ratio > 0.3) or        # Mixed script
            (hindi_ratio > 0.1 and english_ratio > 0.1)            # Any mix of scripts
        )
        
        if is_hinglish:
            return Language.HINGLISH
        elif hindi_ratio > 0.7:
            return Language.HINDI
        elif english_ratio > 0.9:
            return Language.ENGLISH
        else:
            return Language.MIXED
    
    def route_to_tool(self, user_input: str, return_confidence: bool = True) -> RouteDecision:
        """
        Hybrid routing: Intent classifier first, then semantic similarity fallback
        
        Steps:
        1. Try intent classifier with confidence >= 0.55
        2. If classifier confidence < 0.55, fallback to semantic similarity
        3. If semantic similarity also below threshold, return clarification_needed
        """
        start_time = time.time()
        
        # Detect language first
        detected_language = self.detect_language(user_input)
        
        # Initialize variables for tracking
        classifier_confidence = None
        predicted_intent = None
        routing_method = "semantic"  # Default fallback
        selected_tool = None
        confidence_score = 0.0
        reasoning = ""
        
        # Step 1: Try Intent Classifier
        intent_classifier = get_intent_classifier()
        
        if intent_classifier is not None:
            try:
                predicted_intent, classifier_confidence = intent_classifier.predict(user_input)
                
                # Check if classifier confidence is high enough
                if classifier_confidence >= 0.55:
                    # Map intent to tool
                    intent_to_tool_map = {
                        ToolIntent.RECIPE_SUGGESTION: "leftover_chef",
                        ToolIntent.STORY_TELLING: "nani_kahaniyan",
                        ToolIntent.POEM_GENERATION: "poem_generator",
                        ToolIntent.MUSIC_RECOMMENDATION: "vividh_bharti",
                        ToolIntent.FOOD_LOCATION: "food_locator"
                    }
                    
                    selected_tool = intent_to_tool_map.get(predicted_intent)
                    confidence_score = classifier_confidence
                    routing_method = "classifier"
                    reasoning = f"Intent classifier: {predicted_intent.value} (confidence: {classifier_confidence:.3f})"
                    
                    logger.info(f"🎯 Classifier routing: '{user_input}' → {selected_tool} (confidence: {classifier_confidence:.3f})")
                
            except Exception as e:
                logger.warning(f"⚠️  Classifier prediction failed: {e}")
        
        # Step 2: Fallback to Semantic Similarity if classifier didn't work or confidence too low
        if selected_tool is None:
            # Get embedding for user input
            input_embedding = self.model.encode(user_input)
            
            # Calculate similarities with all tools
            similarities = {}
            for tool_name, tool_embedding in self.tool_embeddings.items():
                similarity = np.dot(input_embedding, tool_embedding) / (
                    np.linalg.norm(input_embedding) * np.linalg.norm(tool_embedding)
                )
                similarities[tool_name] = similarity
            
            # Find best match
            best_tool = max(similarities, key=similarities.get)
            best_similarity = similarities[best_tool]
            
            # Check if confidence meets threshold with language-specific adjustments
            tool_metadata = self.tools_metadata[best_tool]
            base_threshold = tool_metadata.confidence_threshold
            
            # Apply language-specific threshold adjustments
            if detected_language == Language.HINGLISH:
                adjusted_threshold = base_threshold * 0.85  # 15% reduction
            elif detected_language == Language.HINDI:
                adjusted_threshold = base_threshold * 0.95  # 5% reduction  
            else:
                adjusted_threshold = base_threshold
            
            meets_threshold = best_similarity >= adjusted_threshold
            
            if meets_threshold:
                selected_tool = best_tool
                confidence_score = best_similarity
                routing_method = "semantic" if intent_classifier is None else "fallback"
                semantic_reason = f"Semantic similarity: {best_similarity:.3f} (threshold: {adjusted_threshold:.3f})"
                
                if classifier_confidence is not None:
                    reasoning = f"Classifier confidence too low ({classifier_confidence:.3f}), fallback to {semantic_reason}"
                else:
                    reasoning = f"Semantic only: {semantic_reason}"
                
                logger.info(f"📊 Semantic routing: '{user_input}' → {selected_tool} (similarity: {best_similarity:.3f})")
            else:
                # Both methods failed - need clarification
                selected_tool = "clarification_needed"
                confidence_score = best_similarity
                routing_method = "clarification"
                
                if classifier_confidence is not None:
                    reasoning = f"Both methods failed: Classifier confidence {classifier_confidence:.3f} < 0.55, Semantic similarity {best_similarity:.3f} < {adjusted_threshold:.3f}"
                else:
                    reasoning = f"Semantic similarity {best_similarity:.3f} below threshold {adjusted_threshold:.3f}"
                
                logger.info(f"❓ Clarification needed: '{user_input}' (low confidence)")
        
        # Create final reasoning with language info
        reasoning += f", Language: {detected_language.value}"
        
        # Calculate semantic similarity for logging even if classifier was used
        if routing_method == "classifier":
            input_embedding = self.model.encode(user_input)
            tool_embedding = self.tool_embeddings[selected_tool]
            semantic_similarity = np.dot(input_embedding, tool_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(tool_embedding)
            )
        else:
            semantic_similarity = confidence_score if routing_method != "clarification" else best_similarity
        
        decision = RouteDecision(
            selected_tool=selected_tool,
            confidence_score=confidence_score,
            reasoning=reasoning,
            language_detected=detected_language,
            semantic_similarity=semantic_similarity,
            timestamp=time.time(),
            routing_method=routing_method,
            classifier_confidence=classifier_confidence,
            predicted_intent=predicted_intent
        )
        
        # Log the decision
        self.route_history.append(decision)
        logger.info(f"🎯 Final decision: '{user_input}' → {selected_tool} via {routing_method} (confidence: {confidence_score:.3f})")
        
        return decision
    
    def evaluate_accuracy(self, test_cases: List[AccuracyTestCase]) -> AccuracyMetrics:
        """Evaluate routing accuracy on a set of test cases"""
        results = []
        
        for test_case in test_cases:
            start_time = time.time()
            
            # Route the test input
            decision = self.route_to_tool(test_case.input_text)
            
            # Check if prediction is correct
            is_correct = decision.selected_tool == test_case.expected_tool
            
            result = EvaluationResult(
                test_case=test_case,
                actual_tool=decision.selected_tool,
                confidence_score=decision.confidence_score,
                is_correct=is_correct,
                reasoning=decision.reasoning,
                execution_time=time.time() - start_time
            )
            
            results.append(result)
        
        # Store results
        self.evaluation_history.extend(results)
        
        # Calculate metrics
        return self._calculate_metrics(results)
    
    def _calculate_metrics(self, results: List[EvaluationResult]) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics"""
        total_tests = len(results)
        correct_predictions = sum(1 for r in results if r.is_correct)
        overall_accuracy = correct_predictions / total_tests if total_tests > 0 else 0.0
        
        # Per-tool metrics
        tool_true_positives = {}
        tool_false_positives = {}
        tool_false_negatives = {}
        
        # Confusion matrix
        confusion_matrix = {}
        
        # Language-specific accuracy
        language_correct = {}
        language_total = {}
        
        for result in results:
            expected = result.test_case.expected_tool
            actual = result.actual_tool
            language = result.test_case.language
            
            # Initialize counters
            if expected not in tool_true_positives:
                tool_true_positives[expected] = 0
                tool_false_negatives[expected] = 0
            if actual not in tool_false_positives:
                tool_false_positives[actual] = 0
                
            if expected not in confusion_matrix:
                confusion_matrix[expected] = {}
            if actual not in confusion_matrix[expected]:
                confusion_matrix[expected][actual] = 0
            
            if language not in language_correct:
                language_correct[language] = 0
                language_total[language] = 0
            
            # Update counters
            confusion_matrix[expected][actual] += 1
            language_total[language] += 1
            
            if result.is_correct:
                tool_true_positives[expected] += 1
                language_correct[language] += 1
            else:
                tool_false_negatives[expected] += 1
                tool_false_positives[actual] += 1
        
        # Calculate precision and recall per tool
        precision_per_tool = {}
        recall_per_tool = {}
        
        for tool in self.tools_metadata.keys():
            tp = tool_true_positives.get(tool, 0)
            fp = tool_false_positives.get(tool, 0)
            fn = tool_false_negatives.get(tool, 0)
            
            precision_per_tool[tool] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_per_tool[tool] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Language accuracy
        language_accuracy = {}
        for lang, total in language_total.items():
            language_accuracy[lang] = language_correct[lang] / total if total > 0 else 0.0
        
        # Average confidence
        avg_confidence = sum(r.confidence_score for r in results) / len(results) if results else 0.0
        
        return AccuracyMetrics(
            total_tests=total_tests,
            correct_predictions=correct_predictions,
            accuracy=overall_accuracy,
            precision_per_tool=precision_per_tool,
            recall_per_tool=recall_per_tool,
            confusion_matrix=confusion_matrix,
            avg_confidence=avg_confidence,
            language_accuracy=language_accuracy
        )
    
    def get_test_dataset(self) -> List[AccuracyTestCase]:
        """Generate comprehensive test dataset for evaluation"""
        return [
            # Recipe Tool Tests
            AccuracyTestCase(input_text="What can I cook with leftover rice and dal?", expected_tool="leftover_chef", expected_intent=ToolIntent.RECIPE_SUGGESTION, language=Language.ENGLISH, description="English recipe query"),
            AccuracyTestCase(input_text="Ghar mein sirf chawal aur dal hai, kuch recipe batao", expected_tool="leftover_chef", expected_intent=ToolIntent.RECIPE_SUGGESTION, language=Language.HINGLISH, description="Hinglish recipe query"),
            AccuracyTestCase(input_text="बचे हुए खाने से क्या बना सकते हैं?", expected_tool="leftover_chef", expected_intent=ToolIntent.RECIPE_SUGGESTION, language=Language.HINDI, description="Hindi recipe query"),
            AccuracyTestCase(input_text="Leftover roti se kya banau?", expected_tool="leftover_chef", expected_intent=ToolIntent.RECIPE_SUGGESTION, language=Language.HINGLISH, description="Leftover roti query"),
            
            # Story Tool Tests  
            AccuracyTestCase(input_text="Tell me a bedtime story", expected_tool="nani_kahaniyan", expected_intent=ToolIntent.STORY_TELLING, language=Language.ENGLISH, description="English bedtime story"),
            AccuracyTestCase(input_text="Bacchon ko sunane ke liye koi achhi kahani batao", expected_tool="nani_kahaniyan", expected_intent=ToolIntent.STORY_TELLING, language=Language.HINGLISH, description="Hinglish children story"),
            AccuracyTestCase(input_text="कोई नैतिक कहानी सुनाइए", expected_tool="nani_kahaniyan", expected_intent=ToolIntent.STORY_TELLING, language=Language.HINDI, description="Hindi moral story"),
            AccuracyTestCase(input_text="Story with moral sunao", expected_tool="nani_kahaniyan", expected_intent=ToolIntent.STORY_TELLING, language=Language.HINGLISH, description="Hinglish moral story"),
            
            # Poem Tool Tests
            AccuracyTestCase(input_text="Write a beautiful poem", expected_tool="poem_generator", expected_intent=ToolIntent.POEM_GENERATION, language=Language.ENGLISH, description="English poem request"),
            AccuracyTestCase(input_text="Koi achhi kavita sunao", expected_tool="poem_generator", expected_intent=ToolIntent.POEM_GENERATION, language=Language.HINGLISH, description="Hinglish poem request"),
            AccuracyTestCase(input_text="प्रेम पर कविता लिखिए", expected_tool="poem_generator", expected_intent=ToolIntent.POEM_GENERATION, language=Language.HINDI, description="Hindi love poem"),
            AccuracyTestCase(input_text="Romantic poetry batao", expected_tool="poem_generator", expected_intent=ToolIntent.POEM_GENERATION, language=Language.HINGLISH, description="Hinglish romantic poem"),
            
            # Music Tool Tests
            AccuracyTestCase(input_text="Suggest some old classic songs", expected_tool="vividh_bharti", expected_intent=ToolIntent.MUSIC_RECOMMENDATION, language=Language.ENGLISH, description="English music request"),
            AccuracyTestCase(input_text="Purane gaane recommend karo", expected_tool="vividh_bharti", expected_intent=ToolIntent.MUSIC_RECOMMENDATION, language=Language.HINGLISH, description="Hinglish music request"),
            AccuracyTestCase(input_text="कुछ पुराने गाने बताइए", expected_tool="vividh_bharti", expected_intent=ToolIntent.MUSIC_RECOMMENDATION, language=Language.HINDI, description="Hindi music request"),
            AccuracyTestCase(input_text="1900s ke nostalgic songs batao", expected_tool="vividh_bharti", expected_intent=ToolIntent.MUSIC_RECOMMENDATION, language=Language.HINGLISH, description="Nostalgic songs request"),
            
            # Food Location Tests
            AccuracyTestCase(input_text="Good restaurants near me", expected_tool="food_locator", expected_intent=ToolIntent.FOOD_LOCATION, language=Language.ENGLISH, description="English restaurant search"),
            AccuracyTestCase(input_text="Yahan ke paas koi achha dhaba hai?", expected_tool="food_locator", expected_intent=ToolIntent.FOOD_LOCATION, language=Language.HINGLISH, description="Hinglish dhaba search"),
            AccuracyTestCase(input_text="पास में खाने की जगह बताइए", expected_tool="food_locator", expected_intent=ToolIntent.FOOD_LOCATION, language=Language.HINDI, description="Hindi food place search"),
            AccuracyTestCase(input_text="Nearby food places batao", expected_tool="food_locator", expected_intent=ToolIntent.FOOD_LOCATION, language=Language.HINGLISH, description="Hinglish food search"),
            
            # Additional challenging Hinglish test cases
            AccuracyTestCase(input_text="Purane gaane recommend karo yaar", expected_tool="vividh_bharti", expected_intent=ToolIntent.MUSIC_RECOMMENDATION, language=Language.HINGLISH, description="Challenging Hinglish music request"),
            AccuracyTestCase(input_text="Koi achhi si kavita sunao na", expected_tool="poem_generator", expected_intent=ToolIntent.POEM_GENERATION, language=Language.HINGLISH, description="Casual Hinglish poem request"),
            AccuracyTestCase(input_text="Bacchon ke liye moral story batao", expected_tool="nani_kahaniyan", expected_intent=ToolIntent.STORY_TELLING, language=Language.HINGLISH, description="Hinglish children story request"),
            AccuracyTestCase(input_text="Dal chawal se kya banana hai", expected_tool="leftover_chef", expected_intent=ToolIntent.RECIPE_SUGGESTION, language=Language.HINGLISH, description="Specific Hinglish recipe query"),
            AccuracyTestCase(input_text="Yahan ka food scene kaisa hai", expected_tool="food_locator", expected_intent=ToolIntent.FOOD_LOCATION, language=Language.HINGLISH, description="Complex Hinglish food inquiry"),
        ] 