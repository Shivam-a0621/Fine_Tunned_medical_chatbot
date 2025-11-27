"""
CLI Chatbot Interface
Interactive command-line interface for the medical chatbot.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medical_chatbot_engine import ConversationalChatbot
from conversation_history import SessionManager
import json
from datetime import datetime


class CLIChatbot:
    """Interactive CLI chatbot interface."""

    def __init__(self, model_path: str = "/home/shivam/pikky/models/merged_model"):
        """
        Initialize CLI chatbot.

        Args:
            model_path: Path to merged model
        """
        self.engine = ConversationalChatbot(model_path=model_path)
        self.session_manager = SessionManager()
        self.current_session = None
        self.generation_params = {
            "max_length": 256,
            "temperature": 0.7,
            "top_p": 0.95,
        }

    def print_header(self) -> None:
        """Print welcome header."""
        print("\n" + "=" * 80)
        print("🏥 MEDICAL CHATBOT - FINE-TUNED LLAMA-3.2-1B")
        print("=" * 80)
        print("Type your medical questions below. Use /help for commands.")
        print("=" * 80 + "\n")

    def print_help(self) -> None:
        """Print help menu."""
        help_text = """
Available Commands:
  /help              - Show this help message
  /clear             - Clear conversation history
  /history           - Show conversation history
  /save <filename>   - Save conversation to file
  /load <filename>   - Load conversation from file
  /settings          - Show/configure generation settings
  /session-new       - Start a new session
  /session-list      - List all saved sessions
  /session-load <id> - Load a saved session
  /session-save      - Save current session
  /exit              - Exit the chatbot
  /quit              - Exit the chatbot

Generation Settings (adjustable via /settings):
  - max_length: Maximum response length (default: 256)
  - temperature: Sampling temperature (default: 0.7)
  - top_p: Nucleus sampling parameter (default: 0.95)

Tips:
  - Ask specific medical questions for better answers
  - Use /clear between unrelated topics
  - Save important conversations with /save
"""
        print(help_text)

    def print_settings(self) -> None:
        """Print current generation settings."""
        print("\nCurrent Generation Settings:")
        print("-" * 40)
        for key, value in self.generation_params.items():
            print(f"  {key}: {value}")
        print("-" * 40)

    def configure_settings(self) -> None:
        """Configure generation settings."""
        print("\nConfigure Generation Settings")
        print("-" * 40)

        try:
            max_len = input(f"Max length [{self.generation_params['max_length']}]: ").strip()
            if max_len:
                self.generation_params["max_length"] = int(max_len)

            temp = input(f"Temperature [{self.generation_params['temperature']}]: ").strip()
            if temp:
                temp_val = float(temp)
                if 0.0 <= temp_val <= 1.0:
                    self.generation_params["temperature"] = temp_val
                else:
                    print("Temperature must be between 0.0 and 1.0")

            top_p = input(f"Top P [{self.generation_params['top_p']}]: ").strip()
            if top_p:
                top_p_val = float(top_p)
                if 0.0 <= top_p_val <= 1.0:
                    self.generation_params["top_p"] = top_p_val
                else:
                    print("Top P must be between 0.0 and 1.0")

            print("✓ Settings updated")

        except ValueError:
            print("✗ Invalid input. Settings not changed.")

    def show_history(self) -> None:
        """Display conversation history."""
        history = self.engine.get_history()

        if not history:
            print("\nNo conversation history yet.")
            return

        print("\n" + "=" * 80)
        print("CONVERSATION HISTORY")
        print("=" * 80)

        for i, msg in enumerate(history, 1):
            role = "Q:" if msg["role"] == "user" else "A:"
            prefix = f"[{i}] {role}"
            print(f"\n{prefix}")
            print(msg["content"][:200] + ("..." if len(msg["content"]) > 200 else ""))

        print("\n" + "=" * 80)

    def save_conversation(self, filename: str) -> None:
        """
        Save conversation to file.

        Args:
            filename: Output filename
        """
        history = self.engine.get_history()
        if not history:
            print("No conversation to save.")
            return

        try:
            save_dir = Path("/home/shivam/pikky/conversations")
            save_dir.mkdir(parents=True, exist_ok=True)

            file_path = save_dir / f"{filename}.json"

            data = {
                "timestamp": datetime.now().isoformat(),
                "messages": history,
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"✓ Conversation saved to {file_path}")

        except Exception as e:
            print(f"✗ Error saving conversation: {str(e)}")

    def load_conversation(self, filename: str) -> None:
        """
        Load conversation from file.

        Args:
            filename: Input filename
        """
        try:
            save_dir = Path("/home/shivam/pikky/conversations")
            file_path = save_dir / f"{filename}.json"

            if not file_path.exists():
                print(f"✗ File not found: {file_path}")
                return

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.engine.clear_history()
            for msg in data["messages"]:
                self.engine.add_to_history(msg["role"], msg["content"])

            print(f"✓ Loaded {len(data['messages'])} messages from {file_path}")

        except Exception as e:
            print(f"✗ Error loading conversation: {str(e)}")

    def handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Args:
            command: User input command

        Returns:
            False if exit command, True otherwise
        """
        parts = command.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self.print_help()
        elif cmd == "/clear":
            self.engine.clear_history()
            print("✓ Conversation history cleared")
        elif cmd == "/history":
            self.show_history()
        elif cmd == "/save":
            if arg:
                self.save_conversation(arg)
            else:
                print("Usage: /save <filename>")
        elif cmd == "/load":
            if arg:
                self.load_conversation(arg)
            else:
                print("Usage: /load <filename>")
        elif cmd == "/settings":
            if arg.lower() == "set":
                self.configure_settings()
            else:
                self.print_settings()
        elif cmd == "/session-new":
            self.current_session = self.session_manager.create_session()
            print(f"✓ New session created: {self.current_session.session_id}")
        elif cmd == "/session-list":
            sessions = self.session_manager.list_sessions()
            if sessions:
                print("\nAvailable sessions:")
                for sid in sessions:
                    print(f"  - {sid}")
            else:
                print("No saved sessions.")
        elif cmd == "/session-load":
            if arg:
                try:
                    self.current_session = self.session_manager.load_session(arg)
                    print(f"✓ Session loaded: {arg}")
                except FileNotFoundError:
                    print(f"✗ Session not found: {arg}")
            else:
                print("Usage: /session-load <session-id>")
        elif cmd == "/session-save":
            if self.current_session:
                self.session_manager.save_current_session()
                print(f"✓ Session saved: {self.current_session.session_id}")
            else:
                print("No active session to save.")
        elif cmd in ["/exit", "/quit"]:
            print("\nGoodbye! 👋")
            return False

        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")

        return True

    def run(self) -> None:
        """Run the interactive CLI chatbot."""
        # Load model
        print("\nLoading medical chatbot engine...")
        if not self.engine.load_model():
            print("\n✗ Failed to load model.")
            print("Please ensure the merged model exists at: /home/shivam/pikky/models/merged_model")
            print("You can create it by running: python /home/shivam/pikky/src/model_merger.py")
            return

        self.print_header()

        try:
            while True:
                try:
                    # Get user input
                    user_input = input("You: ").strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        if not self.handle_command(user_input):
                            break
                        continue

                    # Generate response
                    print("\nChatbot: ", end="", flush=True)
                    response = self.engine.chat(
                        user_input,
                        max_length=self.generation_params["max_length"],
                        temperature=self.generation_params["temperature"],
                    )
                    print(response)
                    print()

                except KeyboardInterrupt:
                    print("\n\nInterrupted. Type /exit to quit.")
                except Exception as e:
                    print(f"\n✗ Error: {str(e)}")
                    print("Please try again.\n")

        except EOFError:
            print("\nGoodbye! 👋")


def main():
    """Main entry point."""
    try:
        chatbot = CLIChatbot()
        chatbot.run()
    except Exception as e:
        print(f"✗ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
