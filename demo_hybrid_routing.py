#!/usr/bin/env python3
"""
Demo: Hybrid Routing Approach
=============================

This script demonstrates the hybrid routing approach using both:
1. Semantic Embedding similarity (Primary)
2. Intent Classification Model (Backup)

Run this after training the intent model to see both approaches in action.

Prerequisites:
1. Run data.py to generate training data JSON
2. Run intent_classifier.py to train the DistilBERT model
3. Run this script to see hybrid routing performance
"""

import json
import os
from router import MultilingualToolRouter
from intent_classifier import MultilingualIntentClassifier
from models import Language, ToolIntent, AccuracyTestCase
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_intent_model():
    """Load the trained intent classification model if available"""
    try:
        if os.path.exists('intent_model/'):
            logger.info("✅ Found trained intent model in intent_model/ directory")
            intent_classifier = MultilingualIntentClassifier()
            intent_classifier.load_model('intent_model/')
            return intent_classifier
        else:
            logger.warning("❌ No trained intent model found. Run intent_classifier.py first!")
            return None
    except Exception as e:
        logger.error(f"Error loading intent model: {e}")
        return None

def test_semantic_routing(router, test_cases):
    """Test semantic embedding routing approach"""
    logger.info("\n🔍 TESTING SEMANTIC EMBEDDING ROUTING")
    logger.info("=" * 50)
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        decision = router.route_to_tool(case.input_text)
        is_correct = decision.selected_tool == case.expected_tool
        
        status = "✅" if is_correct else "❌"
        logger.info(f"{status} Test {i}: '{case.input_text}'")
        logger.info(f"   Expected: {case.expected_tool} | Got: {decision.selected_tool}")
        logger.info(f"   Confidence: {decision.confidence_score:.3f} | Language: {decision.language_detected.value}")
        
        if is_correct:
            correct += 1
        
        logger.info("")
    
    accuracy = (correct / total) * 100
    logger.info(f"📊 SEMANTIC ROUTING ACCURACY: {accuracy:.1f}% ({correct}/{total})")
    return accuracy

def test_intent_classification(intent_classifier, test_cases):
    """Test intent classification model approach"""
    logger.info("\n🧠 TESTING INTENT CLASSIFICATION MODEL")
    logger.info("=" * 50)
    
    if not intent_classifier:
        logger.warning("❌ No intent classifier available. Skipping this test.")
        return 0
    
    correct = 0
    total = len(test_cases)
    
    # Tool mapping from intent to tool name
    intent_to_tool = {
        ToolIntent.RECIPE_SUGGESTION: "leftover_chef",
        ToolIntent.STORY_TELLING: "nani_kahaniyan", 
        ToolIntent.POEM_GENERATION: "poem_generator",
        ToolIntent.MUSIC_RECOMMENDATION: "vividh_bharti",
        ToolIntent.FOOD_LOCATION: "food_locator"
    }
    
    for i, case in enumerate(test_cases, 1):
        try:
            # Predict intent and confidence
            predicted_intent, confidence = intent_classifier.predict(case.input_text)
            predicted_tool = intent_to_tool.get(predicted_intent, "clarification_needed")
            
            is_correct = predicted_tool == case.expected_tool
            
            status = "✅" if is_correct else "❌"
            logger.info(f"{status} Test {i}: '{case.input_text}'")
            logger.info(f"   Expected: {case.expected_tool} | Got: {predicted_tool}")
            logger.info(f"   Intent: {predicted_intent.value} | Confidence: {confidence:.3f}")
            
            if is_correct:
                correct += 1
                
        except Exception as e:
            logger.error(f"❌ Error predicting for test {i}: {e}")
            logger.info(f"   Input: '{case.input_text}'")
        
        logger.info("")
    
    accuracy = (correct / total) * 100
    logger.info(f"📊 INTENT CLASSIFICATION ACCURACY: {accuracy:.1f}% ({correct}/{total})")
    return accuracy

def test_hybrid_approach(router, intent_classifier, test_cases):
    """Test hybrid approach: semantic first, intent model as backup"""
    logger.info("\n🔀 TESTING HYBRID ROUTING APPROACH")
    logger.info("=" * 50)
    
    if not intent_classifier:
        logger.warning("❌ No intent classifier available. Using semantic-only routing.")
        return test_semantic_routing(router, test_cases)
    
    correct = 0
    total = len(test_cases)
    semantic_used = 0
    intent_used = 0
    
    # Tool mapping
    intent_to_tool = {
        ToolIntent.RECIPE_SUGGESTION: "leftover_chef",
        ToolIntent.STORY_TELLING: "nani_kahaniyan",
        ToolIntent.POEM_GENERATION: "poem_generator", 
        ToolIntent.MUSIC_RECOMMENDATION: "vividh_bharti",
        ToolIntent.FOOD_LOCATION: "food_locator"
    }
    
    for i, case in enumerate(test_cases, 1):
        # Step 1: Try semantic routing
        semantic_decision = router.route_to_tool(case.input_text)
        
        # Step 2: If semantic routing is uncertain, try intent classification
        if semantic_decision.selected_tool == "clarification_needed":
            try:
                predicted_intent, intent_confidence = intent_classifier.predict(case.input_text)
                
                # Use intent model if it's confident enough
                if intent_confidence > 0.7:  # Threshold for intent model
                    final_tool = intent_to_tool.get(predicted_intent, "clarification_needed")
                    final_confidence = intent_confidence
                    method_used = "Intent Model"
                    intent_used += 1
                else:
                    final_tool = "clarification_needed"
                    final_confidence = max(semantic_decision.confidence_score, intent_confidence)
                    method_used = "Clarification (both low confidence)"
            except Exception as e:
                final_tool = "clarification_needed"
                final_confidence = semantic_decision.confidence_score
                method_used = f"Clarification (intent error: {e})"
        else:
            final_tool = semantic_decision.selected_tool
            final_confidence = semantic_decision.confidence_score
            method_used = "Semantic Embedding"
            semantic_used += 1
        
        is_correct = final_tool == case.expected_tool
        
        status = "✅" if is_correct else "❌"
        logger.info(f"{status} Test {i}: '{case.input_text}'")
        logger.info(f"   Expected: {case.expected_tool} | Got: {final_tool}")
        logger.info(f"   Method: {method_used} | Confidence: {final_confidence:.3f}")
        logger.info(f"   Language: {case.language.value}")
        
        if is_correct:
            correct += 1
        
        logger.info("")
    
    accuracy = (correct / total) * 100
    logger.info(f"📊 HYBRID ROUTING ACCURACY: {accuracy:.1f}% ({correct}/{total})")
    logger.info(f"🎯 Method Usage: Semantic={semantic_used}, Intent={intent_used}, Other={total-semantic_used-intent_used}")
    return accuracy

def create_demo_test_cases():
    """Create a set of demo test cases for comparison"""
    return [
        # Hinglish cases (challenging)
        AccuracyTestCase(
            input_text="Ghar mein sirf chawal aur dal hai, kuch recipe batao",
            expected_tool="leftover_chef",
            expected_intent=ToolIntent.RECIPE_SUGGESTION,
            language=Language.HINGLISH,
            description="Hinglish recipe request with leftovers"
        ),
        AccuracyTestCase(
            input_text="Koi achhi kavita sunao nature ke bare mein",
            expected_tool="poem_generator",
            expected_intent=ToolIntent.POEM_GENERATION,
            language=Language.HINGLISH,
            description="Hinglish poetry request about nature"
        ),
        AccuracyTestCase(
            input_text="Purane gaane recommend karo 1960s ke",
            expected_tool="vividh_bharti",
            expected_intent=ToolIntent.MUSIC_RECOMMENDATION,
            language=Language.HINGLISH,
            description="Hinglish music request for old songs"
        ),
        AccuracyTestCase(
            input_text="Yahan ke paas koi achha dhaba hai",
            expected_tool="food_locator",
            expected_intent=ToolIntent.FOOD_LOCATION,
            language=Language.HINGLISH,
            description="Hinglish restaurant location query"
        ),
        AccuracyTestCase(
            input_text="Bacchon ko sunane ke liye story chahiye",
            expected_tool="nani_kahaniyan",
            expected_intent=ToolIntent.STORY_TELLING,
            language=Language.HINGLISH,
            description="Hinglish story request for children"
        ),
        
        # English cases
        AccuracyTestCase(
            input_text="What can I cook with leftover rice?",
            expected_tool="leftover_chef",
            expected_intent=ToolIntent.RECIPE_SUGGESTION,
            language=Language.ENGLISH,
            description="English recipe request with leftovers"
        ),
        AccuracyTestCase(
            input_text="Tell me a bedtime story",
            expected_tool="nani_kahaniyan",
            expected_intent=ToolIntent.STORY_TELLING,
            language=Language.ENGLISH,
            description="English bedtime story request"
        ),
        AccuracyTestCase(
            input_text="Write a romantic poem",
            expected_tool="poem_generator",
            expected_intent=ToolIntent.POEM_GENERATION,
            language=Language.ENGLISH,
            description="English romantic poetry request"
        ),
        AccuracyTestCase(
            input_text="Suggest classic bollywood songs",
            expected_tool="vividh_bharti",
            expected_intent=ToolIntent.MUSIC_RECOMMENDATION,
            language=Language.ENGLISH,
            description="English Bollywood music request"
        ),
        AccuracyTestCase(
            input_text="Good restaurants near me",
            expected_tool="food_locator",
            expected_intent=ToolIntent.FOOD_LOCATION,
            language=Language.ENGLISH,
            description="English restaurant location query"
        ),
        
        # Hindi cases
        AccuracyTestCase(
            input_text="बचे हुए खाने से कुछ बना सकते हैं",
            expected_tool="leftover_chef",
            expected_intent=ToolIntent.RECIPE_SUGGESTION,
            language=Language.HINDI,
            description="Hindi recipe request with leftovers"
        ),
        AccuracyTestCase(
            input_text="कोई अच्छी कहानी सुनाओ",
            expected_tool="nani_kahaniyan",
            expected_intent=ToolIntent.STORY_TELLING,
            language=Language.HINDI,
            description="Hindi story request"
        ),
        AccuracyTestCase(
            input_text="प्रेम की कविता लिखो",
            expected_tool="poem_generator",
            expected_intent=ToolIntent.POEM_GENERATION,
            language=Language.HINDI,
            description="Hindi love poetry request"
        ),
        
        # Edge cases
        AccuracyTestCase(
            input_text="Music",
            expected_tool="clarification_needed",
            expected_intent=ToolIntent.MUSIC_RECOMMENDATION,  # Intent unclear but likely music
            language=Language.ENGLISH,
            description="Ambiguous English music query"
        ),
        AccuracyTestCase(
            input_text="Food",
            expected_tool="clarification_needed",
            expected_intent=ToolIntent.FOOD_LOCATION,  # Intent unclear but likely food
            language=Language.ENGLISH,
            description="Ambiguous English food query"
        ),
    ]

def main():
    """Main demo function"""
    logger.info("🚀 HYBRID ROUTING APPROACH DEMO")
    logger.info("=" * 60)
    logger.info("Testing both semantic embedding and intent classification approaches")
    logger.info("")
    
    # Initialize components
    logger.info("⚙️ Initializing components...")
    router = MultilingualToolRouter()
    intent_classifier = load_intent_model()
    test_cases = create_demo_test_cases()
    
    logger.info(f"📝 Created {len(test_cases)} test cases across 3 languages")
    logger.info("")
    
    # Run all tests
    results = {}
    
    # Test 1: Semantic Embedding Only
    results['semantic'] = test_semantic_routing(router, test_cases)
    
    # Test 2: Intent Classification Only  
    results['intent'] = test_intent_classification(intent_classifier, test_cases)
    
    # Test 3: Hybrid Approach
    results['hybrid'] = test_hybrid_approach(router, intent_classifier, test_cases)
    
    # Summary
    logger.info("\n📈 FINAL COMPARISON RESULTS")
    logger.info("=" * 50)
    logger.info(f"🔍 Semantic Embedding:     {results['semantic']:.1f}%")
    logger.info(f"🧠 Intent Classification:  {results['intent']:.1f}%")
    logger.info(f"🔀 Hybrid Approach:        {results['hybrid']:.1f}%")
    
    # Determine best approach
    best_method = max(results, key=results.get)
    best_accuracy = results[best_method]
    
    logger.info(f"\n🏆 BEST PERFORMING METHOD: {best_method.upper()} ({best_accuracy:.1f}%)")
    
    # Recommendations
    logger.info("\n💡 RECOMMENDATIONS:")
    if results['hybrid'] >= max(results['semantic'], results['intent']):
        logger.info("✅ Hybrid approach performs best - use semantic with intent backup")
    elif results['semantic'] > results['intent']:
        logger.info("✅ Semantic embedding performs best - use as primary method")
    else:
        logger.info("✅ Intent classification performs best - consider using as primary")
    
    if intent_classifier is None:
        logger.info("\n⚠️  To see full hybrid performance:")
        logger.info("   1. Run: python data.py")
        logger.info("   2. Run: python intent_classifier.py")
        logger.info("   3. Run: python demo_hybrid_routing.py")

if __name__ == "__main__":
    main() 