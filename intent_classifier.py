#!/usr/bin/env python3
"""
Multilingual Intent Classifier for MCP Tool Routing
Uses transformers for training and inference on Hindi+English+Hinglish data
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Dict, Tuple
import logging
from models import ToolIntent
from data import IntentDatasetGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentDataset(Dataset):
    """Custom dataset for intent classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MultilingualIntentClassifier:
    """Multilingual intent classifier for tool routing"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_to_intent = {}
        self.intent_to_label = {}
        self.trainer = None
        
        # Initialize intent mapping
        self._initialize_label_mapping()
    
    def _initialize_label_mapping(self):
        """Initialize mapping between labels and intents"""
        intents = [
            ToolIntent.RECIPE_SUGGESTION,
            ToolIntent.STORY_TELLING,
            ToolIntent.POEM_GENERATION,
            ToolIntent.MUSIC_RECOMMENDATION,
            ToolIntent.FOOD_LOCATION
        ]
        
        for i, intent in enumerate(intents):
            self.label_to_intent[i] = intent
            self.intent_to_label[intent] = i
        
        logger.info(f"Initialized {len(intents)} intent classes")
    
    def prepare_training_data(self, training_data: List[Dict]) -> Tuple[List[str], List[int]]:
        """Prepare training data for the model"""
        texts = []
        labels = []
        
        for example in training_data:
            texts.append(example['text'])
            intent_enum = ToolIntent(example['intent'])
            label = self.intent_to_label[intent_enum]
            labels.append(label)
        
        logger.info(f"Prepared {len(texts)} training examples")
        return texts, labels
    
    def train(self, training_data: List[Dict], output_dir: str = "./intent_model", 
              test_size: float = 0.2, epochs: int = 3, batch_size: int = 16):
        """Train the intent classifier"""
        
        # Prepare data
        texts, labels = self.prepare_training_data(training_data)
        
        # Split into train/validation
        split_idx = int(len(texts) * (1 - test_size))
        train_texts, val_texts = texts[:split_idx], texts[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        logger.info(f"Training split: {len(train_texts)} train, {len(val_texts)} validation")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_to_intent)
        )
        
        # Create datasets
        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = IntentDataset(val_texts, val_labels, self.tokenizer)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,  # Reduced warmup for smaller dataset
            weight_decay=0.01,
            learning_rate=2e-5,  # Better learning rate for fine-tuning
            logging_dir=f"{output_dir}/logs",
            logging_steps=5,  # More frequent logging
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=3,  # Keep only best 3 models
            dataloader_drop_last=False,  # Don't drop incomplete batches
            fp16=False,  # Disable for stability on smaller datasets
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        # Train
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save model
        self.save_model(output_dir)
        
        # Evaluate
        logger.info("Evaluating model...")
        eval_results = self.trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        return eval_results
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    def save_model(self, output_dir: str):
        """Save trained model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mapping
        with open(f"{output_dir}/label_mapping.json", 'w') as f:
            mapping = {
                "label_to_intent": {str(k): v.value for k, v in self.label_to_intent.items()},
                "intent_to_label": {k.value: v for k, v in self.intent_to_label.items()}
            }
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """Load trained model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        # Load label mapping
        with open(f"{model_dir}/label_mapping.json", 'r') as f:
            mapping = json.load(f)
            self.label_to_intent = {int(k): ToolIntent(v) for k, v in mapping["label_to_intent"].items()}
            self.intent_to_label = {ToolIntent(k): v for k, v in mapping["intent_to_label"].items()}
        
        # Set to evaluation mode
        self.model.eval()
        logger.info(f"Model loaded from {model_dir}")
    
    def predict(self, text: str) -> Tuple[ToolIntent, float]:
        """Predict intent for a single text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get prediction and confidence
            predicted_label = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_label].item()
            
            predicted_intent = self.label_to_intent[predicted_label]
            
        return predicted_intent, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[ToolIntent, float]]:
        """Predict intents for multiple texts"""
        results = []
        for text in texts:
            intent, confidence = self.predict(text)
            results.append((intent, confidence))
        return results
    
    def evaluate_on_test_data(self, test_data: List[Dict]) -> Dict:
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        texts, true_labels = self.prepare_training_data(test_data)
        
        # Get predictions
        predictions = []
        confidences = []
        
        for text in texts:
            intent, confidence = self.predict(text)
            pred_label = self.intent_to_label[intent]
            predictions.append(pred_label)
            confidences.append(confidence)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        # Classification report
        target_names = [intent.value for intent in self.label_to_intent.values()]
        report = classification_report(
            true_labels, predictions, 
            target_names=target_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        results = {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": predictions,
            "confidences": confidences
        }
        
        logger.info(f"Test accuracy: {accuracy:.3f}, Avg confidence: {avg_confidence:.3f}")
        
        return results

def train_intent_classifier():
    """Train the intent classifier with saved training data"""
    logger.info("🚀 Starting intent classifier training...")
    
    # Load training data from saved JSON file
    logger.info("📋 Loading training data from training_data.json...")
    try:
        with open("training_data_enhanced.json", 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        logger.info(f"✅ Loaded {len(training_data)} training examples from file")
    except FileNotFoundError:
        logger.warning("⚠️ training_data.json not found, generating fresh data...")
        generator = IntentDatasetGenerator()
        training_data = generator.generate_training_data()
        logger.info(f"Generated {len(training_data)} training examples")
    
    # Print dataset statistics
    intent_counts = {}
    language_counts = {}
    for example in training_data:
        intent = example['intent']
        language = example['language']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        language_counts[language] = language_counts.get(language, 0) + 1
    
    logger.info("📊 Training Dataset Statistics:")
    logger.info("Intent Distribution:")
    for intent, count in intent_counts.items():
        logger.info(f"  {intent}: {count} examples")
    logger.info("Language Distribution:")
    for language, count in language_counts.items():
        logger.info(f"  {language}: {count} examples")
    
    # Initialize classifier
    classifier = MultilingualIntentClassifier()
    
    # Train model with better parameters for our dataset size
    logger.info("🏋️ Training model...")
    eval_results = classifier.train(
        training_data,
        output_dir="./intent_model",
        epochs=10,  # More epochs for better learning
        batch_size=8,  # Small batch size for our dataset
        test_size=0.15  # Keep more data for training
    )
    
    logger.info("✅ Training completed!")
    logger.info(f"📈 Final validation accuracy: {eval_results.get('eval_accuracy', 'N/A'):.3f}")
    
    return classifier, eval_results

def test_classifier_predictions():
    """Test the trained classifier with sample inputs"""
    logger.info("🧪 Testing classifier predictions...")
    
    # Load trained model
    classifier = MultilingualIntentClassifier()
    
    try:
        classifier.load_model("./intent_model")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Training new model...")
        classifier, _ = train_intent_classifier()
    
    # Test examples
    test_examples = [
        "Ghar mein bacha hua chawal hai, kya banega?",
        "Story sunao bacchon ke liye",
        "Purane gaane baja do",
        "Restaurants nearby",
        "Poem likho love ke upar",
        "कुछ अच्छी कविता सुनाइए",
        "What can I cook with leftover rice?",
        "नैतिक कहानी बताइए"
    ]
    
    logger.info("🎯 Testing sample queries:")
    for text in test_examples:
        intent, confidence = classifier.predict(text)
        logger.info(f"'{text}' → {intent.value} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    # Train classifier
    train_intent_classifier()
    
    # Test predictions
    test_classifier_predictions() 