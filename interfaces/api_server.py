"""
Flask REST API for Medical Chatbot
RESTful API endpoints for programmatic access to the chatbot.
"""

import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medical_chatbot_engine import ConversationalChatbot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global chatbot engine
chatbot_engine: Optional[ConversationalChatbot] = None


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model."""
    question: str
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    include_context: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the symptoms of diabetes?",
                "max_length": 256,
                "temperature": 0.7,
                "include_context": True,
            }
        }


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    timestamp: str
    message_count: int
    status: str = "success"


class MessageData(BaseModel):
    """Individual message data."""
    role: str
    content: str


class HistoryResponse(BaseModel):
    """History response model."""
    messages: list
    total_messages: int
    user_messages: int
    assistant_messages: int
    status: str = "success"


class ModelInfo(BaseModel):
    """Model information."""
    model_type: str
    device: str
    vocab_size: int
    is_loaded: bool
    status: str = "success"


# Request validation
def validate_request(data: Dict[str, Any], model_class: BaseModel) -> tuple[bool, Any]:
    """
    Validate request data against model.

    Args:
        data: Request data dictionary
        model_class: Pydantic model class

    Returns:
        Tuple of (is_valid, data_or_error)
    """
    try:
        validated = model_class(**data)
        return True, validated
    except ValidationError as e:
        return False, e.errors()


# Error handlers
@app.errorhandler(400)
def bad_request(e):
    """Handle bad requests."""
    return jsonify({
        "status": "error",
        "message": "Bad request",
        "details": str(e)
    }), 400


@app.errorhandler(404)
def not_found(e):
    """Handle not found errors."""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal error: {str(e)}")
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500


# API Routes
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.

    Returns:
        JSON response with server status
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": chatbot_engine is not None
    }), 200


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """
    Get model information.

    Returns:
        JSON response with model metadata
    """
    if chatbot_engine is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 503

    try:
        info = chatbot_engine.get_model_info()
        return jsonify({
            "status": "success",
            "model_info": info
        }), 200
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Generate response to medical question.

    Expected JSON:
        {
            "question": "Medical question",
            "max_length": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "include_context": true
        }

    Returns:
        JSON response with generated answer
    """
    if chatbot_engine is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 503

    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "Empty request body"
            }), 400

        # Validate request
        is_valid, result = validate_request(data, ChatRequest)
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": "Invalid request",
                "details": result
            }), 400

        chat_req = result

        # Generate response
        response = chatbot_engine.chat(
            question=chat_req.question,
            max_length=chat_req.max_length,
            temperature=chat_req.temperature,
        )

        # Prepare response
        history = chatbot_engine.get_history()

        return jsonify({
            "status": "success",
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(history),
        }), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Get conversation history.

    Returns:
        JSON response with conversation history
    """
    if chatbot_engine is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 503

    try:
        messages = chatbot_engine.get_history()
        user_count = sum(1 for m in messages if m["role"] == "user")
        assistant_count = sum(1 for m in messages if m["role"] == "assistant")

        return jsonify({
            "status": "success",
            "messages": messages,
            "total_messages": len(messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
        }), 200

    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """
    Clear conversation history.

    Returns:
        JSON response confirming deletion
    """
    if chatbot_engine is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 503

    try:
        chatbot_engine.clear_history()
        return jsonify({
            "status": "success",
            "message": "Conversation history cleared"
        }), 200

    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/batch-chat", methods=["POST"])
def batch_chat():
    """
    Generate responses to multiple questions.

    Expected JSON:
        {
            "questions": ["Question 1", "Question 2", ...],
            "max_length": 256,
            "temperature": 0.7
        }

    Returns:
        JSON response with multiple answers
    """
    if chatbot_engine is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 503

    try:
        data = request.get_json()
        if not data or "questions" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'questions' field"
            }), 400

        questions = data.get("questions", [])
        max_length = data.get("max_length", 256)
        temperature = data.get("temperature", 0.7)

        if not isinstance(questions, list):
            return jsonify({
                "status": "error",
                "message": "'questions' must be a list"
            }), 400

        # Generate responses
        responses = chatbot_engine.batch_generate(
            questions=questions,
            max_length=max_length,
            temperature=temperature,
        )

        return jsonify({
            "status": "success",
            "responses": [
                {"question": q, "answer": a}
                for q, a in zip(questions, responses)
            ],
            "count": len(responses),
        }), 200

    except Exception as e:
        logger.error(f"Error in batch-chat endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Root endpoint
@app.route("/", methods=["GET"])
def root():
    """Root endpoint with API documentation."""
    return jsonify({
        "name": "Medical Chatbot API",
        "version": "1.0.0",
        "description": "REST API for fine-tuned medical chatbot",
        "endpoints": {
            "GET /api/health": "Health check",
            "GET /api/model-info": "Get model information",
            "POST /api/chat": "Send a question and get response",
            "GET /api/history": "Get conversation history",
            "DELETE /api/history": "Clear conversation history",
            "POST /api/batch-chat": "Generate responses to multiple questions",
        },
        "documentation": "See /api/docs",
    }), 200


def initialize_chatbot(model_path: str = "/home/shivam/pikky/models/merged_model"):
    """
    Initialize the chatbot engine.

    Args:
        model_path: Path to merged model
    """
    global chatbot_engine

    logger.info("Initializing chatbot engine...")
    chatbot_engine = ConversationalChatbot(model_path=model_path)

    if chatbot_engine.load_model():
        logger.info("✓ Chatbot engine loaded successfully")
        return True
    else:
        logger.error("✗ Failed to load chatbot engine")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Medical Chatbot REST API Server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    parser.add_argument(
        "--model-path",
        default="/home/shivam/pikky/models/merged_model",
        help="Path to merged model"
    )

    args = parser.parse_args()

    # Initialize chatbot
    if initialize_chatbot(args.model_path):
        logger.info(f"Starting server on {args.host}:{args.port}")
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=args.debug,
        )
    else:
        logger.error("Failed to start server: Model loading failed")
        sys.exit(1)
