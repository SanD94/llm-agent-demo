import os
import sys
import argparse
import textwrap
from typing import List, Dict, Any, Optional
from huggingface_hub import InferenceClient



class LlamaChat:
    def __init__(self, api_token: Optional[str] = None):
        """Initialize the Llama chat interface with Hugging Face API token."""
        # Load environment variables from .env file
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        if not self.api_token:
            raise ValueError("Hugging Face API token is required. Set HF_API_TOKEN environment variable or pass it as an argument.")
        
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.client = InferenceClient(token=self.api_token, provider="novita")
        self.conversation_history: List[Dict[str, str]] = []
    
    def format_message(self, role: str, content: str) -> Dict[str, str]:
        """Format a message for the conversation history."""
        return {"role": role, "content": content}
    
    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(self.format_message(role, content))
    
    def send_message_stream(self, message: str) -> None:
        # Add user message to history
        self.add_to_history("user", message)
        
        full_response = ""
        try:
            # Stream the response
            stream = self.client.chat_completion(
                model=self.model_id,
                messages=self.conversation_history,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stream=True
            )
            
            print("\nLlama:", end="", flush=True)
            
            # Process the streaming response
            for response in stream:
                chunk = response.choices[0].delta.content
                if chunk:
                    print(chunk, end="", flush=True)
                    full_response += chunk
            
            print("\n")  # Add newline at the end
            
            # Add the complete response to history
            self.add_to_history("assistant", full_response)
            
        except Exception as e:
            print(f"\nError communicating with the API: {str(e)}")

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []


def print_wrapped(text: str, width: int = 80) -> None:
    """Print text with wrapping for better readability."""
    for line in text.split('\n'):
        print('\n'.join(textwrap.wrap(line, width=width)))


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Chat with Meta-Llama-3-8B-Instruct model")
    parser.add_argument("--token", "-t", type=str, help="Hugging Face API token (optional, can use HF_API_TOKEN env var)")
    parser.add_argument("--width", "-w", type=int, default=80, help="Output width for text wrapping (default: 80)")
    args = parser.parse_args()
    
    try:
        # Initialize the chat interface
        chat = LlamaChat(api_token=args.token)
        print("\nLlama 8B-Instruct Chat CLI")
        print("Type 'exit', 'quit', or Ctrl+C to end the chat")
        print("Type 'clear' to start a new conversation\n")
        
        while True:
            # Get user input
            try:
                user_input = input("\n> ")
            except KeyboardInterrupt:
                print("\nExiting chat...")
                break
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break
            
            # Check for clear command
            if user_input.lower() == "clear":
                chat.clear_history()
                print("Conversation history cleared.")
                continue
            
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # Get response from the model
            print("\nLlama is thinking...", flush=True)
            response = chat.send_message_stream(user_input)
            
            # Print the response
            print("\nLlama:", flush=True)
            print_wrapped(response, width=args.width)
            print()  # Extra line for readability
    
    except KeyboardInterrupt:
        print("\nExiting chat...")
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())