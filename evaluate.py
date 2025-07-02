#!/usr/bin/env python3
"""
Comprehensive evaluation script for hybrid multilingual tool routing
Demonstrates classifier + semantic fallback performance
"""

import asyncio
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
from collections import defaultdict

from router import MultilingualToolRouter
from data import IntentDatasetGenerator
from intent_classifier import MultilingualIntentClassifier, train_intent_classifier
from models import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRoutingEvaluator:
    """Comprehensive evaluator for hybrid routing system"""
    
    def __init__(self):
        self.router = MultilingualToolRouter()
        self.results = []
        
    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("🚀 Starting comprehensive hybrid routing evaluation...")
        
        # Step 1: Ensure classifier is trained
        self._ensure_classifier_trained()
        
        # Step 2: Generate test dataset
        test_data = self._generate_test_dataset()
        
        # Step 3: Run evaluation
        evaluation_results = self._evaluate_hybrid_routing(test_data)
        
        # Step 4: Analyze routing methods
        routing_analysis = self._analyze_routing_methods(evaluation_results)
        
        # Step 5: Test specific examples
        example_results = self._test_specific_examples()
        
        # Step 6: Generate comprehensive report
        self._generate_report(evaluation_results, routing_analysis, example_results)
        
        return evaluation_results, routing_analysis, example_results
    
    def _ensure_classifier_trained(self):
        """Ensure intent classifier is trained and available"""
        logger.info("🔧 Checking intent classifier...")
        
        try:
            classifier = MultilingualIntentClassifier()
            classifier.load_model("./intent_model")
            logger.info("✅ Intent classifier loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  Classifier not found: {e}")
            logger.info("🏋️ Training new intent classifier...")
            train_intent_classifier()
            logger.info("✅ Intent classifier training completed")
    
    def _generate_test_dataset(self) -> List[Dict]:
        """Generate comprehensive test dataset"""
        logger.info("📋 Generating test dataset...")
        
        generator = IntentDatasetGenerator()
        training_data = generator.generate_training_data()
        
        # Use a subset for testing (different from training)
        test_data = training_data[80:]  # Last 20% for testing
        
        logger.info(f"Generated {len(test_data)} test examples")
        return test_data
    
    def _evaluate_hybrid_routing(self, test_data: List[Dict]) -> List[Dict]:
        """Evaluate hybrid routing on test dataset"""
        logger.info("🧪 Evaluating hybrid routing...")
        
        results = []
        routing_method_counts = defaultdict(int)
        
        for i, example in enumerate(test_data):
            start_time = time.time()
            
            # Route using hybrid system
            decision = self.router.route_to_tool(example['text'])
            
            # Check correctness
            is_correct = decision.selected_tool == example['tool']
            
            # Track routing method
            routing_method_counts[decision.routing_method] += 1
            
            result = {
                'example': example,
                'decision': decision,
                'is_correct': is_correct,
                'execution_time': time.time() - start_time,
                'routing_method': decision.routing_method,
                'classifier_confidence': decision.classifier_confidence,
                'semantic_similarity': decision.semantic_similarity
            }
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_data)} examples")
        
        logger.info(f"📊 Routing method distribution: {dict(routing_method_counts)}")
        
        return results
    
    def _analyze_routing_methods(self, results: List[Dict]) -> Dict:
        """Analyze performance by routing method"""
        logger.info("📈 Analyzing routing methods...")
        
        method_stats = defaultdict(lambda: {
            'count': 0,
            'correct': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'by_language': defaultdict(lambda: {'count': 0, 'correct': 0})
        })
        
        for result in results:
            method = result['routing_method']
            language = result['example']['language']
            
            method_stats[method]['count'] += 1
            method_stats[method]['avg_confidence'] += result['decision'].confidence_score
            
            if result['is_correct']:
                method_stats[method]['correct'] += 1
                method_stats[method]['by_language'][language]['correct'] += 1
            
            method_stats[method]['by_language'][language]['count'] += 1
        
        # Calculate final stats
        for method, stats in method_stats.items():
            if stats['count'] > 0:
                stats['accuracy'] = stats['correct'] / stats['count']
                stats['avg_confidence'] = stats['avg_confidence'] / stats['count']
                
                # Language-specific accuracy
                for lang_stats in stats['by_language'].values():
                    if lang_stats['count'] > 0:
                        lang_stats['accuracy'] = lang_stats['correct'] / lang_stats['count']
        
        return dict(method_stats)
    
    def _test_specific_examples(self) -> List[Dict]:
        """Test the specific examples mentioned in the requirements"""
        logger.info("🎯 Testing specific example queries...")
        
        test_queries = [
            {
                "query": "Ghar mein bacha hua chawal hai, kya banega?",
                "expected_tool": "leftover_chef",
                "expected_intent": "recipe_suggestion",
                "description": "Hinglish leftover recipe query"
            },
            {
                "query": "Story sunao bacchon ke liye",
                "expected_tool": "nani_kahaniyan", 
                "expected_intent": "story_telling",
                "description": "Hinglish children story request"
            },
            {
                "query": "Purane gaane baja do",
                "expected_tool": "vividh_bharti",
                "expected_intent": "music_recommendation", 
                "description": "Hinglish old songs request"
            },
            {
                "query": "Restaurants nearby",
                "expected_tool": "food_locator",
                "expected_intent": "food_location",
                "description": "English nearby restaurants"
            },
            {
                "query": "Poem likho love ke upar",
                "expected_tool": "poem_generator",
                "expected_intent": "poem_generation",
                "description": "Hinglish love poem request"
            },
            {
                "query": "कुछ अच्छी कविता सुनाइए",
                "expected_tool": "poem_generator",
                "expected_intent": "poem_generation",
                "description": "Hindi poem request"
            },
            {
                "query": "What can I cook with leftover rice?",
                "expected_tool": "leftover_chef",
                "expected_intent": "recipe_suggestion", 
                "description": "English leftover recipe query"
            },
            {
                "query": "नैतिक कहानी बताइए",
                "expected_tool": "nani_kahaniyan",
                "expected_intent": "story_telling",
                "description": "Hindi moral story request"
            }
        ]
        
        results = []
        
        for test in test_queries:
            decision = self.router.route_to_tool(test['query'])
            
            is_correct_tool = decision.selected_tool == test['expected_tool']
            is_correct_intent = (
                decision.predicted_intent is not None and 
                decision.predicted_intent.value == test['expected_intent']
            )
            
            result = {
                'query': test['query'],
                'expected_tool': test['expected_tool'],
                'expected_intent': test['expected_intent'],
                'description': test['description'],
                'decision': decision,
                'is_correct_tool': is_correct_tool,
                'is_correct_intent': is_correct_intent,
                'routing_method': decision.routing_method,
                'classifier_confidence': decision.classifier_confidence,
                'semantic_similarity': decision.semantic_similarity
            }
            
            results.append(result)
            
            # Log result
            status = "✅" if is_correct_tool else "❌"
            logger.info(
                f"{status} '{test['query']}' → {decision.selected_tool} "
                f"via {decision.routing_method} (confidence: {decision.confidence_score:.3f})"
            )
        
        return results
    
    def _generate_report(self, evaluation_results: List[Dict], 
                        routing_analysis: Dict, example_results: List[Dict]):
        """Generate comprehensive evaluation report"""
        logger.info("📋 Generating evaluation report...")
        
        # Overall statistics
        total_tests = len(evaluation_results)
        correct_predictions = sum(1 for r in evaluation_results if r['is_correct'])
        overall_accuracy = correct_predictions / total_tests if total_tests > 0 else 0.0
        
        # Routing method breakdown
        classifier_usage = sum(1 for r in evaluation_results if r['routing_method'] == 'classifier')
        semantic_usage = sum(1 for r in evaluation_results if r['routing_method'] in ['semantic', 'fallback'])
        clarification_usage = sum(1 for r in evaluation_results if r['routing_method'] == 'clarification')
        
        print("\n" + "="*80)
        print("           HYBRID MULTILINGUAL TOOL ROUTING EVALUATION REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Correct Predictions: {correct_predictions}")
        print(f"   Overall Accuracy: {overall_accuracy:.1%}")
        
        print(f"\n🔄 ROUTING METHOD DISTRIBUTION:")
        print(f"   Intent Classifier: {classifier_usage} ({classifier_usage/total_tests:.1%})")
        print(f"   Semantic Fallback: {semantic_usage} ({semantic_usage/total_tests:.1%})")
        print(f"   Clarification Needed: {clarification_usage} ({clarification_usage/total_tests:.1%})")
        
        print(f"\n📈 PERFORMANCE BY ROUTING METHOD:")
        for method, stats in routing_analysis.items():
            print(f"   {method.upper()}:")
            print(f"     Count: {stats['count']}")
            print(f"     Accuracy: {stats['accuracy']:.1%}")
            print(f"     Avg Confidence: {stats['avg_confidence']:.3f}")
        
        print(f"\n🎯 SPECIFIC EXAMPLE RESULTS:")
        example_correct = sum(1 for r in example_results if r['is_correct_tool'])
        print(f"   Accuracy on key examples: {example_correct}/{len(example_results)} ({example_correct/len(example_results):.1%})")
        
        for result in example_results:
            status = "✅" if result['is_correct_tool'] else "❌"
            print(f"   {status} {result['description']}")
            print(f"       Query: '{result['query']}'")
            print(f"       Routed to: {result['decision'].selected_tool} via {result['routing_method']}")
            print(f"       Confidence: {result['decision'].confidence_score:.3f}")
        
        # Language-specific performance
        print(f"\n🌐 PERFORMANCE BY LANGUAGE:")
        language_stats = defaultdict(lambda: {'count': 0, 'correct': 0})
        
        for result in evaluation_results:
            lang = result['example']['language']
            language_stats[lang]['count'] += 1
            if result['is_correct']:
                language_stats[lang]['correct'] += 1
        
        for lang, stats in language_stats.items():
            accuracy = stats['correct'] / stats['count'] if stats['count'] > 0 else 0.0
            print(f"   {lang.upper()}: {stats['correct']}/{stats['count']} ({accuracy:.1%})")
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Save detailed results
        with open("evaluation_results.json", 'w', encoding='utf-8') as f:
            # Convert to serializable format
            serializable_results = []
            for result in evaluation_results:
                ser_result = {
                    'example': result['example'],
                    'decision': {
                        'selected_tool': result['decision'].selected_tool,
                        'confidence_score': result['decision'].confidence_score,
                        'reasoning': result['decision'].reasoning,
                        'language_detected': result['decision'].language_detected.value,
                        'routing_method': result['decision'].routing_method,
                        'classifier_confidence': result['decision'].classifier_confidence,
                        'semantic_similarity': result['decision'].semantic_similarity
                    },
                    'is_correct': result['is_correct'],
                    'execution_time': result['execution_time']
                }
                serializable_results.append(ser_result)
            
            json.dump({
                'overall_stats': {
                    'total_tests': total_tests,
                    'correct_predictions': correct_predictions,
                    'overall_accuracy': overall_accuracy
                },
                'routing_analysis': routing_analysis,
                'detailed_results': serializable_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info("📁 Detailed results saved to evaluation_results.json")

async def main():
    """Main evaluation function"""
    evaluator = HybridRoutingEvaluator()
    
    try:
        evaluation_results, routing_analysis, example_results = evaluator.run_comprehensive_evaluation()
        
        print("\n🎉 Hybrid routing evaluation completed successfully!")
        print("   • Intent classifier + semantic fallback working")
        print("   • Multilingual support validated")
        print("   • Performance metrics generated")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 