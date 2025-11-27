"""
Conversation History Manager
Manages conversation sessions and history for the chatbot.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import uuid


class ConversationHistory:
    """Manages conversation history and context."""

    def __init__(self, session_id: Optional[str] = None, max_history: int = 10):
        """
        Initialize conversation history.

        Args:
            session_id: Unique session identifier (auto-generated if None)
            max_history: Maximum number of message pairs to keep in context
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Args:
            role: Either 'user' or 'assistant'
            content: Message content
        """
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be 'user' or 'assistant'")

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_context(self) -> str:
        """
        Get recent conversation context formatted for model input.

        Returns:
            Formatted conversation context string
        """
        if not self.messages:
            return ""

        # Get last max_history message pairs
        recent_messages = self.messages[-(self.max_history * 2) :]

        context_parts = []
        for msg in recent_messages:
            role_prefix = "Q:" if msg["role"] == "user" else "A:"
            context_parts.append(f"{role_prefix} {msg['content']}")

        return "\n\n".join(context_parts)

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in conversation."""
        return self.messages.copy()

    def get_recent_messages(self, count: int = 10) -> List[Dict[str, str]]:
        """
        Get most recent messages.

        Args:
            count: Number of recent messages to return

        Returns:
            List of message dictionaries
        """
        return self.messages[-count:]

    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages = []
        self.updated_at = datetime.now()

    def get_summary(self) -> Dict:
        """
        Get conversation summary.

        Returns:
            Dictionary with session metadata
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m["role"] == "user"),
            "assistant_messages": sum(
                1 for m in self.messages if m["role"] == "assistant"
            ),
        }

    def save_to_file(self, file_path: str) -> None:
        """
        Save conversation to JSON file.

        Args:
            file_path: Path to save conversation
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": self.messages,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str) -> "ConversationHistory":
        """
        Load conversation from JSON file.

        Args:
            file_path: Path to conversation file

        Returns:
            ConversationHistory instance
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        history = cls(session_id=data["session_id"])
        history.messages = data["messages"]
        history.created_at = datetime.fromisoformat(data["created_at"])
        history.updated_at = datetime.fromisoformat(data["updated_at"])

        return history

    def export_to_text(self) -> str:
        """
        Export conversation as formatted text.

        Returns:
            Formatted conversation text
        """
        if not self.messages:
            return "No conversation history."

        text_parts = [
            "=" * 80,
            f"Session ID: {self.session_id}",
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Updated: {self.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Messages: {len(self.messages)}",
            "=" * 80,
            "",
        ]

        for i, msg in enumerate(self.messages, 1):
            role = "QUESTION" if msg["role"] == "user" else "ANSWER"
            timestamp = msg.get("timestamp", "").split("T")[1].split(".")[0]
            text_parts.append(f"[{i}] {role} ({timestamp})")
            text_parts.append(msg["content"])
            text_parts.append("-" * 80)

        return "\n".join(text_parts)

    def __len__(self) -> int:
        """Get number of messages in conversation."""
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"ConversationHistory(session_id={self.session_id}, "
            f"messages={len(self.messages)})"
        )


class SessionManager:
    """Manages multiple conversation sessions."""

    def __init__(self, sessions_dir: str = "/home/shivam/pikky/sessions"):
        """
        Initialize session manager.

        Args:
            sessions_dir: Directory to store session files
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[ConversationHistory] = None

    def create_session(self, session_id: Optional[str] = None) -> ConversationHistory:
        """
        Create a new conversation session.

        Args:
            session_id: Optional specific session ID

        Returns:
            New ConversationHistory instance
        """
        self.current_session = ConversationHistory(session_id=session_id)
        return self.current_session

    def load_session(self, session_id: str) -> ConversationHistory:
        """
        Load existing session.

        Args:
            session_id: Session ID to load

        Returns:
            Loaded ConversationHistory instance
        """
        file_path = self.sessions_dir / f"{session_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        self.current_session = ConversationHistory.load_from_file(str(file_path))
        return self.current_session

    def save_current_session(self) -> None:
        """Save current session to disk."""
        if self.current_session is None:
            raise ValueError("No active session")

        file_path = self.sessions_dir / f"{self.current_session.session_id}.json"
        self.current_session.save_to_file(str(file_path))

    def list_sessions(self) -> List[str]:
        """
        List all available sessions.

        Returns:
            List of session IDs
        """
        json_files = self.sessions_dir.glob("*.json")
        return sorted([f.stem for f in json_files])

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self.sessions_dir / f"{session_id}.json"

        if not file_path.exists():
            return False

        file_path.unlink()
        return True
