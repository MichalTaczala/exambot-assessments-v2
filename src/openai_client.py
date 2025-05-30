import os
from dotenv import load_dotenv
import openai


class OpenAIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or self.load_api_key()
        self.client = openai
        if self.api_key:
            self.client.api_key = self.api_key
        else:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment or .env file.")

    @staticmethod
    def load_api_key():
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or .env file.")
        return api_key

    def test_connection(self):
        # This does not make a real API call, just checks if the client is set up
        return self.client is not None and (self.api_key is not None)

    def chat_completion(self, prompt, system_message=None, model="gpt-4.1-nano", **kwargs):
        """
        Send a prompt to the OpenAI chat completion API and return the response text.

        Args:
            prompt (str): The user prompt to send.
            system_message (str, optional): System message for context.
            model (str, optional): Model name to use. Defaults to "gpt-4.1-nano".
            **kwargs: Additional parameters for openai.ChatCompletion.create.

        Returns:
            str: The response text from the model.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            # For openai>=1.0, response.choices[0].message.content
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAIClient] Error during chat completion: {e}")
            return None


# Example usage (for development/testing only)
if __name__ == "__main__":
    # Make sure to set OPENAI_API_KEY in your .env or environment before running
    try:
        client = OpenAIClient()
        print("Client initialized:", client.test_connection())
        # Test chat completion if a real API key is set
        prompt = "What is Knowledge Representation and Reasoning?"
        response = client.chat_completion(prompt)
        print("Chat completion response:", response)
    except Exception as e:
        print("Error initializing OpenAIClient or making chat completion:", e)
