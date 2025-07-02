# Pucho Ghar ke Baatein - Multilingual MCP Tool Routing System

## 🎯 Project Overview

This project demonstrates a **comprehensive solution to measure the accuracy of correct MCP tool selection when dealing with multilingual inputs (Hindi + English text)**. It was built to address the core interview question from Puch AI about handling tool routing in multilingual scenarios.

---

## 🏗️ System Architecture

### Complete Tool Ecosystem

```mermaid
graph TD
    A[User Input<br/>Hindi/English/Hinglish] --> B[MultilingualToolRouter]
    B --> C{Routing Method}
    C -->|Primary| E[Intent Classification<br/>Model]
    C -->|Backup| D[Semantic Embedding<br/>Similarity]
    D --> F[Confidence Check]
    E --> F
    F --> G{Confidence > Threshold?}
    G -->|Yes| H[Execute Selected Tool]
    G -->|No| I[Request Clarification]
    H --> J[5 Specialized Tools]
    J --> K[Return Results + Metrics]
    K --> L[Log for Accuracy Analysis]
```

### Dual Routing Architecture

1. **Primary: Intent Classification Model**
   - Trained paraphrase-multilingual-MiniLM-L12-v2 model on custom dataset
   - 540+ training examples across 5 intents
   - Stored in `intent_model/` with full checkpoints

2. **Secondary: Semantic Embedding Approach**
   - Uses `paraphrase-multilingual-MiniLM-L12-v2`
   - Pre-computed tool embeddings in Hindi, English, Hinglish
   - Cosine similarity matching with confidence threshold

---

## 🛠️ All Available MCP Tools

### Individual Specialized Tools

#### 1. 🍽️ Leftover Chef (`leftover_chef`)
- **Purpose**: Creative recipe suggestions from leftover ingredients
- **Perfect for**: Indian household cooking scenarios
- **Examples**: 
  - `"Ghar mein sirf chawal aur dal hai, kuch recipe batao"`
  - `"What can I cook with leftover rice and vegetables?"`
- **Parameters**: `leftovers`, `cuisine_type`, `dietary_preferences`

#### 2. 📖 Nani Ki Kahaniyan (`nani_kahaniyan`) 
- **Purpose**: Traditional bedtime stories and moral tales
- **Perfect for**: Children's storytelling with cultural values
- **Examples**:
  - `"Bacchon ko sunane ke liye koi achhi kahani batao"`
  - `"Tell me a moral story about honesty"`
- **Parameters**: `age_group`, `moral_theme`, `language_preference`

#### 3. 🎭 Poem Generator (`poem_generator`)
- **Purpose**: Beautiful Hindi/English poetry on various themes
- **Perfect for**: Creative expression and cultural poetry
- **Examples**:
  - `"Koi achhi kavita sunao about nature"`
  - `"Write a romantic poem in Hinglish"`
- **Parameters**: `theme`, `style`, `language_preference`

#### 4. 🎵 Vividh Bharti Jukebox (`vividh_bharti`)
- **Purpose**: Nostalgic music recommendations from golden era
- **Perfect for**: 1950s-1980s classic Bollywood nostalgia
- **Examples**:
  - `"Purane gaane recommend karo 1960s ke"`
  - `"Suggest some nostalgic songs by Lata Mangeshkar"`
- **Parameters**: `era`, `mood`, `artist_preference`

#### 5. 📍 Food Locator (`food_locator`)
- **Purpose**: Nearby restaurant and street food recommendations
- **Perfect for**: Local dining discovery with cultural preferences
- **Examples**:
  - `"Yahan ke paas koi achha dhaba hai?"`
  - `"Budget-friendly restaurants near me"`
- **Parameters**: `latitude`, `longitude`, `food_type`, `budget_range`

### Intelligent Routing Tools

#### 6. 🤖 Intelligent Assistant (`intelligent_assistant`)
- **Purpose**: Automatically routes user input to the best specialized tool
- **Perfect for**: When you want the system to decide the best tool automatically
- **Examples**: Any natural language query in Hindi/English/Hinglish
- **Returns**: Tool result + routing confidence + decision reasoning

#### 7. 🔍 Route Input (`route_input`)
- **Purpose**: Diagnostic tool showing routing decision process
- **Perfect for**: Understanding how the system makes routing decisions
- **Returns**: Selected tool, confidence score, reasoning, language detection

#### 8. 📊 Evaluate Routing Accuracy (`evaluate_routing_accuracy`)
- **Purpose**: Comprehensive accuracy evaluation across all test cases
- **Perfect for**: Measuring system performance and identifying improvement areas
- **Returns**: Detailed accuracy metrics, confusion matrix, per-tool performance

---

## 🔬 Technical Implementation Deep Dive

### Model Training Approach

We explored two approaches for multilingual tool routing:

#### 1. Intent Classification Model (Primary)
```bash
# Training details
Model: paraphrase-multilingual-MiniLM-L12-v2 (fine-tuned for sequence classification)
Dataset: 540+ examples across 5 intents in 3 languages
Training epochs: 3
Final training loss: 0.10
Validation accuracy: 95%+
Location: intent_model/ directory with full checkpoints
```

![Intent Classification Training Results](public/training_intent.png)
*Intent classification model training showing loss reduction to 0.10 and high validation accuracy*

**Training Data Structure:**
```json
{
  "text": "Ghar mein leftover rice hai, kuch banau?",
  "intent": "RECIPE_SUGGESTION", 
  "language": "hinglish"
}
```

#### 2. Semantic Embedding Approach (Backup)
```python
# Model details
Transformer: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (same base model)
Approach: Pre-compute tool embeddings, match user input via cosine similarity
Thresholds: 0.4 across all tools (used as fallback when intent model is uncertain)
```

**Why Intent Classification is Primary:**
- ✅ **Trained on our specific use case** - Learns exact patterns for our 5 tools
- ✅ **Higher accuracy** - 0.10 loss with 95%+ validation accuracy
- ✅ **Cultural context understanding** - Trained on Hindi/English/Hinglish examples
- ✅ **Confident predictions** - Provides probability scores for each intent

### Multilingual Processing Pipeline

```python
class MultilingualToolRouter:
    def route_to_tool(self, user_input: str) -> RoutingDecision:
        # 1. Language Detection (Hindi, English, Hinglish)
        language = self._detect_language(user_input)
        
        # 2. Semantic Encoding
        input_embedding = self.model.encode(user_input)
        
        # 3. Similarity Calculation
        similarities = cosine_similarity([input_embedding], self.tool_embeddings)[0]
        
        # 4. Confidence-based Selection
        max_similarity = max(similarities)
        if max_similarity > self.confidence_thresholds[selected_tool]:
            return selected_tool
        else:
            return "clarification_needed"
```

![Routing Testing Results](public/testing_routing.png)
*Comprehensive routing testing results showing accuracy across different languages and tools*

### Evaluation Framework Architecture

```python
@dataclass
class AccuracyTestCase:
    input_text: str          # "Ghar mein chawal hai kya banau?"
    expected_tool: str       # "leftover_chef"
    intent: ToolIntent       # ToolIntent.RECIPE_SUGGESTION
    language: Language       # Language.HINGLISH
    confidence_min: float    # 0.6

@dataclass
class AccuracyMetrics:
    accuracy: float                              # Overall accuracy
    precision_per_tool: Dict[str, float]         # Per-tool precision
    recall_per_tool: Dict[str, float]           # Per-tool recall  
    language_accuracy: Dict[Language, float]     # Per-language accuracy
    confusion_matrix: Dict[str, Dict[str, int]]  # Confusion analysis
    confidence_distribution: Dict[str, List[float]]  # Confidence scores
```

### Test Dataset Examples

Our evaluation covers realistic multilingual scenarios:

```python
test_cases = [
    # Hinglish (challenging cases)
    AccuracyTestCase("Ghar mein sirf chawal aur dal hai", "leftover_chef", ...),
    AccuracyTestCase("Koi achhi kavita sunao nature ke bare mein", "poem_generator", ...),
    
    # Pure Hindi
    AccuracyTestCase("बच्चों को सुनाने के लिए कोई अच्छी कहानी बताओ", "nani_kahaniyan", ...),
    
    # Pure English  
    AccuracyTestCase("Tell me a bedtime story", "nani_kahaniyan", ...),
    
    # Edge cases
    AccuracyTestCase("Music", "clarification_needed", ...), # Ambiguous
]
```

---

## 🚀 Quick Start Guide

### Installation & Setup

```bash
# 1. Clone and navigate
git clone <your-repo>
cd "Pucho Ghar ke Baatein"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Environment setup
echo "TOKEN=your_mcp_token" > .env  
echo "MY_NUMBER=your_phone_number" >> .env


```

### Running the Complete System

```bash
# Start MCP server with all tools
python main.py

# Output:
# 🍳 Individual Tools: leftover_chef, nani_kahaniyan, poem_generator, vividh_bharti, food_locator  
# 🤖 Intelligent Router: intelligent_assistant (automatically routes to best tool)
# 📊 Diagnostic tools: route_input, evaluate_routing_accuracy
# ✅ All tools exposed with RichToolDescription for optimal AI selection
# Server running on http://0.0.0.0:8085
```

### Using the System

#### Option 1: Direct Tool Access
```python
# Call specific tools directly
await leftover_chef(leftovers=["rice", "dal"], cuisine_type="Indian")
await nani_kahaniyan(moral_theme="honesty", language_preference="hinglish") 
```

#### Option 2: Intelligent Routing
```python
# Let the system choose the best tool
result = await intelligent_assistant("Ghar mein leftover rice hai kuch banau?")

# Returns:
{
    "status": "success",
    "tool_result": {...},  # Recipe suggestions
    "routing_info": {
        "selected_tool": "leftover_chef",
        "confidence": 0.87,
        "language": "hinglish",
        "reasoning": "High similarity to recipe intent patterns"
    }
}
```

#### Option 3: Diagnostic Analysis
```python
# Understand routing decisions  
decision = await route_input("Koi achhi kavita sunao")

# Evaluate system performance
metrics = await evaluate_routing_accuracy()
```

---

### Key Technical Insights Demonstrated

1. **Intent-Based Routing > Keyword Matching**
   - "Recipe batao" and "खाना बनाना है" both route to `leftover_chef`
   - Semantic understanding transcends literal translation

2. **Confidence-Based Decision Making**
   - System knows when it doesn't know
   - "Music" → clarification rather than wrong guess

3. **Cultural Nuance Preservation**
   - "Nani ki kahaniyan" ≠ "Tell story" (different cultural contexts)
   - Hinglish code-switching handled naturally

4. **Comprehensive Evaluation**
   - Beyond simple accuracy: precision, recall, confusion analysis
   - Language-specific performance measurement
   - Real-world test scenarios

---

## 📚 Step-by-Step Training & Testing Guide

### Complete Workflow for Intent Model Training and Hybrid Routing

To fully experience both routing approaches and see the trained intent model in action:

#### Step 1: Generate Training Data
```bash
python data.py
```
**What this does:**
- Creates comprehensive training dataset in JSON format
- Generates 540+ examples across 5 intents in 3 languages
- Saves to `training_data_enhanced.json`
- Includes Hindi, English, and Hinglish examples for each tool

#### Step 2: Train Intent Classification Model
```bash
python intent_classifier.py
```
**What this does:**
- Loads the training data from JSON
- Fine-tunes paraphrase-multilingual-MiniLM-L12-v2 for sequence classification
- Achieves **0.10 training loss** and 95%+ validation accuracy
- Saves complete model in `intent_model/` directory
- Creates checkpoints and tokenizer files

#### Step 3: Test Hybrid Routing Performance
```bash
python demo_hybrid_routing.py
```
**What this does:**
- Compares intent classification (primary) vs semantic embedding (backup) approaches
- Tests both methods on identical test cases
- Shows confidence scores and decision reasoning
- Demonstrates how the primary intent model works
- Provides performance comparison and recommendations

### Expected Output Example

```
🚀 HYBRID ROUTING APPROACH DEMO
============================================================

🧠 TESTING INTENT CLASSIFICATION MODEL
==================================================
✅ Test 1: 'Ghar mein sirf chawal aur dal hai, kuch recipe batao'
   Expected: leftover_chef | Got: leftover_chef
   Intent: RECIPE_SUGGESTION | Confidence: 0.912

🔍 TESTING SEMANTIC EMBEDDING ROUTING
==================================================
✅ Test 1: 'Ghar mein sirf chawal aur dal hai, kuch recipe batao'
   Expected: leftover_chef | Got: leftover_chef
   Confidence: 0.823 | Language: hinglish

📈 FINAL COMPARISON RESULTS
==================================================
🧠 Intent Classification (Primary):  86.7%
🔍 Semantic Embedding (Backup):     73.3%

🏆 BEST PERFORMING METHOD: INTENT (86.7%)
```

### Training Results

The intent classification model achieves excellent performance:
- **Training Loss**: 0.10 (very low, indicating good learning)
- **Base Model**: paraphrase-multilingual-MiniLM-L12-v2 (fine-tuned)
- **Training Time**: ~5min minutes on standard hardware

---

## 🎯 Key Achievements & Learning Outcomes

### What We Built

1. **Complete MCP Tool Ecosystem** - 8 functional tools with rich descriptions
2. **Dual Routing Architecture** - Semantic + classification approaches  
3. **Multilingual Competency** - Natural Hindi/English/Hinglish handling
4. **Production Evaluation Framework** - Comprehensive accuracy measurement
5. **Real-World Testing** - 20+ realistic multilingual test cases

### Technical Innovations

- **No-Translation Approach**: Direct semantic understanding vs translation pipeline
- **Confidence-Based Routing**: Know when to ask for clarification
- **Cultural Context Preservation**: "Ghar ke baatein" vs generic home conversations
- **Hybrid Architecture**: Multiple routing methods with intelligent fallbacks

## 🚧 Future Improvements

### Immediate Focus: Improving Hinglish Accuracy

Current issue: "Ghar mein leftover rice hai kuch banau?" sometimes routes incorrectly

**Planned Solutions:**
1. **Enhanced Hinglish Training Data**: More code-switching examples
2. **Improved Language Detection**: Better Roman Hindi word recognition  
3. **Context-Aware Embeddings**: Fine-tune on Indian conversation patterns
4. **Threshold Optimization**: Per-language confidence thresholds
---

## 🏆 Project Impact

This project provides a **complete, working solution** to the Puch AI interview challenge:

✅ **Demonstrates tool routing mastery** with 8 functional MCP tools  
✅ **Handles multilinguality naturally** with semantic understanding  
✅ **Measures accuracy comprehensively** with detailed evaluation framework  
✅ **Production-ready architecture** with confidence scoring and monitoring  
✅ **Real-world applicability** for Indian household conversation scenarios  

**Key Achievement**: Built a system that can intelligently route "Ghar mein chawal hai kya banau?" to recipe suggestions while measuring and improving its accuracy across languages.

---