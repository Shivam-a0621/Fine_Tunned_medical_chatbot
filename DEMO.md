# Medical Chatbot - Demo & Examples

## Overview

This document demonstrates how to use the Medical Chatbot with examples from all three interfaces.

---

## Interface 1: CLI (Command-Line Interface)

### Running the CLI
```bash
python run_cli.py
```

### Example Session

```
================================================================================
🏥 MEDICAL CHATBOT - FINE-TUNED LLAMA-3.2-1B
================================================================================
Type your medical questions below. Use /help for commands.
================================================================================

You: What is type 2 diabetes?

Chatbot: Type 2 diabetes is a metabolic disorder characterized by elevated
blood glucose levels due to insulin resistance. The body cannot effectively
use the insulin it produces. Risk factors include obesity, sedentary lifestyle,
family history, and age. Management involves lifestyle modifications, medications
like metformin, and monitoring blood glucose levels.

You: /history

================================================================================
CONVERSATION HISTORY
================================================================================

[1] Q:
What is type 2 diabetes?

A:
Type 2 diabetes is a metabolic disorder...

================================================================================

You: /save diabetes_conversation

✓ Conversation saved to /home/shivam/pikky/conversations/diabetes_conversation.json

You: /settings

Current Generation Settings:
----------------------------------------
  max_length: 256
  temperature: 0.7
  top_p: 0.95
----------------------------------------

You: What are the complications?

Chatbot: Diabetes complications include cardiovascular disease, kidney damage,
diabetic neuropathy, retinopathy leading to blindness, diabetic foot ulcers,
and increased infection risk. Long-term management is crucial to prevent these
complications.

You: /clear

✓ Conversation history cleared

You: /exit

Goodbye! 👋
```

### Available CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show help menu | `/help` |
| `/clear` | Clear conversation | `/clear` |
| `/history` | Show chat history | `/history` |
| `/save <name>` | Save conversation | `/save my_chat` |
| `/load <name>` | Load saved conversation | `/load my_chat` |
| `/settings` | View/configure settings | `/settings` |
| `/session-new` | Create new session | `/session-new` |
| `/session-list` | List all sessions | `/session-list` |
| `/session-load <id>` | Load session by ID | `/session-load abc123` |
| `/session-save` | Save current session | `/session-save` |
| `/exit` | Exit chatbot | `/exit` |

---

## Interface 2: Web UI (Streamlit)

### Running the Web Interface
```bash
streamlit run run_web.py
```

Opens browser at: `http://localhost:8501`

### Web UI Features

#### Chat Screen
```
┌─────────────────────────────────────────────────────────────┐
│  🏥 Medical Chatbot                                    [⚙️]  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 👤 You                                                   │ │
│  │ What is hypertension?                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 🤖 Chatbot                                              │ │
│  │ Hypertension, or high blood pressure, occurs when      │ │
│  │ the force of blood against artery walls is too high... │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Ask a medical question:                                 │ │
│  │ [What causes high cholesterol?                        ] │ │
│  │                                                          │ │
│  │ [📤 Send Question]  [🔄 Reload]                        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

Sidebar:
┌──────────────────┐
│ ⚙️  SETTINGS      │
├──────────────────┤
│ 🏥 Model Info    │
│ ├─ Llama-3.2-1B  │
│ ├─ Device: cuda  │
│ ├─ Status: ✓     │
│                  │
│ 🎛️  Parameters   │
│ ├─ Max Length: 256
│ ├─ Temp: 0.7    │
│ └─ Top P: 0.95  │
│                  │
│ 🗑️  [Clear]      │
│ 📊 [Stats]       │
│ 📥 [Export]      │
└──────────────────┘
```

### Web UI Example

**User asks:** "What is the treatment for asthma?"

**Chatbot responds:**
"Asthma treatment depends on severity. Quick-relief medications include albuterol inhalers for acute symptoms. Long-term control medications include inhaled corticosteroids (fluticasone, budesonide) and long-acting beta-agonists. Severe asthma may require biologic medications like omalizumab. Treatment should include an action plan for attacks."

**Features:**
- Real-time response generation
- Adjustable parameters via sliders
- View conversation statistics
- Export conversation as TXT or JSON
- Clear history button
- Modern, responsive design

---

## Interface 3: REST API

### Starting the API Server

```bash
python run_api.py --host 0.0.0.0 --port 5000
```

Server starts at: `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET http://localhost:5000/api/health

Response:
{
  "status": "healthy",
  "timestamp": "2025-11-27T10:30:45.123456",
  "model_loaded": true
}
```

#### 2. Single Question
```bash
POST http://localhost:5000/api/chat
Content-Type: application/json

Request:
{
  "question": "What is the difference between type 1 and type 2 diabetes?",
  "max_length": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "include_context": true
}

Response:
{
  "status": "success",
  "response": "Type 1 diabetes is an autoimmune condition where the pancreas does not produce insulin. It usually appears in children and young adults. Type 2 diabetes involves insulin resistance where the body cannot effectively use insulin. It develops gradually in adults. Type 1 requires insulin therapy, while type 2 can often be managed with lifestyle changes and oral medications.",
  "timestamp": "2025-11-27T10:30:50.654321",
  "message_count": 1
}
```

#### 3. Batch Chat (Multiple Questions)
```bash
POST http://localhost:5000/api/batch-chat
Content-Type: application/json

Request:
{
  "questions": [
    "What causes high blood pressure?",
    "How do you treat high blood pressure?"
  ],
  "max_length": 256,
  "temperature": 0.7
}

Response:
{
  "status": "success",
  "responses": [
    {
      "question": "What causes high blood pressure?",
      "answer": "Hypertension can be caused by... [response]"
    },
    {
      "question": "How do you treat high blood pressure?",
      "answer": "Treatment includes... [response]"
    }
  ],
  "count": 2
}
```

#### 4. Get Conversation History
```bash
GET http://localhost:5000/api/history

Response:
{
  "status": "success",
  "messages": [
    {
      "role": "user",
      "content": "What is diabetes?"
    },
    {
      "role": "assistant",
      "content": "Diabetes is a metabolic disorder..."
    }
  ],
  "total_messages": 2,
  "user_messages": 1,
  "assistant_messages": 1
}
```

#### 5. Clear History
```bash
DELETE http://localhost:5000/api/history

Response:
{
  "status": "success",
  "message": "Conversation history cleared"
}
```

#### 6. Model Information
```bash
GET http://localhost:5000/api/model-info

Response:
{
  "status": "success",
  "model_info": {
    "model_path": "/home/shivam/pikky/models/merged_model",
    "device": "cuda",
    "dtype": "torch.float16",
    "model_type": "llama",
    "vocab_size": 128256,
    "is_training": false
  }
}
```

### Python Client Example

```python
import requests
import json

API_URL = "http://localhost:5000/api"

# Single question
def ask_question(question):
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "question": question,
            "max_length": 256,
            "temperature": 0.7
        }
    )
    return response.json()

# Example usage
result = ask_question("What are the symptoms of pneumonia?")
print(result["response"])

# Batch questions
def ask_batch(questions):
    response = requests.post(
        f"{API_URL}/batch-chat",
        json={"questions": questions}
    )
    return response.json()

# Example usage
batch_result = ask_batch([
    "What is asthma?",
    "How is asthma treated?"
])

for item in batch_result["responses"]:
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}\n")
```

### cURL Examples

```bash
# Single question
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is myocardial infarction?",
    "max_length": 256,
    "temperature": 0.5
  }'

# Get history
curl http://localhost:5000/api/history

# Clear history
curl -X DELETE http://localhost:5000/api/history

# Health check
curl http://localhost:5000/api/health
```

---

## Configuration Examples

### CLI Configuration

Adjust parameters in chat:
```
You: /settings set
Max length [256]: 512
Temperature [0.7]: 0.5
Top P [0.95]: 0.95
✓ Settings updated
```

### Web UI Configuration

Use sidebar sliders:
- **Max Length:** 50-512 (default 256)
- **Temperature:** 0.0-1.0 (default 0.7)
  - Lower (0.3-0.5): More factual
  - Higher (0.8-1.0): More creative

### API Configuration

Pass parameters in request:
```json
{
  "question": "Medical question",
  "max_length": 512,        // Longer responses
  "temperature": 0.3,       // More factual
  "top_p": 0.9             // Stricter sampling
}
```

---

## Sample Medical Questions

The chatbot excels with specific medical questions:

### Diseases
- "What are the symptoms of diabetes?"
- "What causes asthma?"
- "Explain hypertension."

### Treatments
- "How is pneumonia treated?"
- "What medications treat depression?"
- "Describe chemotherapy."

### Anatomy & Physiology
- "Explain the circulatory system."
- "How does the immune system work?"
- "Describe the process of digestion."

### Pharmacology
- "What are the side effects of penicillin?"
- "How does aspirin work?"
- "Explain antihistamines."

### Clinical Questions
- "What is the differential diagnosis for chest pain?"
- "How do you diagnose diabetes?"
- "Explain the ECG findings in MI."

---

## Expected Response Quality

**Good Response Example:**
```
Q: What are the signs and symptoms of myocardial infarction?

A: Myocardial infarction (heart attack) presents with:
- Chest pain or pressure (often described as crushing)
- Pain radiating to arms, neck, or jaw
- Shortness of breath
- Nausea and diaphoresis
- Palpitations and anxiety

Symptoms may differ in women and elderly patients. It's a medical emergency
requiring immediate hospitalization and ECG/troponin testing.
```

**Temperature Effect:**

**Low Temperature (0.3)** - More factual:
- Precise medical terminology
- Consistent responses
- Suitable for clinical decision-making

**Medium Temperature (0.7)** - Balanced (default):
- Natural language
- Some variation
- Good for educational purposes

**High Temperature (0.9)** - More creative:
- Varied expressions
- More discussion-like
- Less suitable for critical decisions

---

## Troubleshooting

### Slow Responses
- Reduce `max_length` (256 → 128)
- Use GPU instead of CPU
- Reduce concurrent requests

### Memory Issues
- Enable 4-bit quantization (default)
- Use smaller batch size
- Reduce context window

### Poor Quality Answers
- Lower temperature (0.7 → 0.5)
- Ask more specific questions
- Provide context via conversation history

### API Connection Issues
```bash
# Check if server is running
curl http://localhost:5000/api/health

# Check firewall
sudo ufw allow 5000

# Use different port
python run_api.py --port 5001
```

---

## Performance Tips

1. **For CLI:** Use temperature 0.5 for factual medical information
2. **For Web:** Adjust sliders based on needs (lower for facts, higher for discussion)
3. **For API:** Batch related questions together to maintain context
4. **General:** Keep questions specific and concise

---

## Disclaimer

This chatbot is for **educational purposes only**. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.

---

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Merge model: `python src/model_merger.py`
3. Choose interface and start chatting!
   - CLI: `python run_cli.py`
   - Web: `streamlit run run_web.py`
   - API: `python run_api.py`
