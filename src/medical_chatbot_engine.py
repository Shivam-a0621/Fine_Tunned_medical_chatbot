"""
Medical Chatbot Inference Engine
Core engine for loading the fine-tuned model and generating responses.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalChatbotEngine:
    """Core inference engine for medical chatbot."""

    def __init__(
        self,
        model_path: str = "/home/shivam/pikky/models-20251126T202424Z-1-001/models/medical_chatbot_llama2_qlora",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the medical chatbot engine.

        Args:
            model_path: Path to merged model
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            dtype: Data type for model (torch.float16 or torch.float32)
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

        logger.info(f"Medical Chatbot Engine initialized")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Data type: {dtype}")

    def load_model(self) -> bool:
        """
        Load the model and tokenizer.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model path does not exist: {self.model_path}")
                return False

            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                device_map="auto" if self.device == "cuda" else self.device,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )

            self.model.eval()
            logger.info("✓ Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def generate_response(
        self,
        question: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate a response to a medical question.

        Args:
            question: Medical question from user
            max_length: Maximum length of response in tokens
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            context: Optional conversation context

        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Format prompt with optional context
        if context:
            prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.device
            )

            # Validate input length
            if inputs.shape[1] > 2048:
                logger.warning(f"Input prompt truncated (too long: {inputs.shape[1]})")
                inputs = inputs[:, -2000:]

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer part
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response

            return answer

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: Unable to generate response. {str(e)}"

    def batch_generate(
        self,
        questions: list,
        max_length: int = 256,
        temperature: float = 0.7,
        context: Optional[str] = None,
    ) -> list:
        """
        Generate responses for multiple questions.

        Args:
            questions: List of medical questions
            max_length: Maximum length per response
            temperature: Sampling temperature
            context: Optional conversation context

        Returns:
            List of generated responses
        """
        responses = []
        for question in questions:
            response = self.generate_response(
                question=question,
                max_length=max_length,
                temperature=temperature,
                context=context,
            )
            responses.append(response)
        return responses

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {"status": "Model not loaded"}

        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "dtype": str(self.dtype),
            "model_type": self.model.config.model_type if hasattr(self.model, "config") else "Unknown",
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "is_training": self.model.training,
        }

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")


class ConversationalChatbot(MedicalChatbotEngine):
    """Extended chatbot with conversation memory."""

    def __init__(self, *args, **kwargs):
        """Initialize conversational chatbot."""
        super().__init__(*args, **kwargs)
        self.conversation_history = []
        self.max_history = 10

    def add_to_history(self, role: str, content: str) -> None:
        """
        Add message to conversation history.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.conversation_history.append({"role": role, "content": content})

        # Keep only recent messages
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[
                -(self.max_history * 2) :
            ]

    def get_context_string(self) -> str:
        """
        Get conversation context as formatted string.

        Returns:
            Formatted context for model input
        """
        if not self.conversation_history:
            return ""

        context_parts = []
        for msg in self.conversation_history:
            prefix = "Q:" if msg["role"] == "user" else "A:"
            context_parts.append(f"{prefix} {msg['content']}")

        return "\n\n".join(context_parts)

    def chat(
        self,
        question: str,
        max_length: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate response with conversation history.

        Args:
            question: User question
            max_length: Max response length
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        # Get context from history
        context = self.get_context_string()

        # Generate response
        response = self.generate_response(
            question=question,
            max_length=max_length,
            temperature=temperature,
            context=context,
        )

        # Add to history
        self.add_to_history("user", question)
        self.add_to_history("assistant", response)

        return response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> list:
        """Get conversation history."""
        return self.conversation_history.copy()


if __name__ == "__main__":
    # Test the engine
    print("Testing Medical Chatbot Engine...")
    print("=" * 80)

    # Initialize engine
    engine = MedicalChatbotEngine()

    # Try to load model (will fail if merged model doesn't exist yet)
    if engine.load_model():
        print("Model loaded successfully!")

        # Test generation
        test_question = "What are the symptoms of diabetes?"
        print(f"\nQuestion: {test_question}")
        response = engine.generate_response(test_question)
        print(f"Answer: {response}")
    else:
        print("Model not found. Please merge the model first using model_merger.py")
