import os
from .schema import PromoItem
from dotenv import load_dotenv
load_dotenv()

DEFAULT_MODEL = os.getenv("PRIMARY_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

class LLM:
    def __init__(self):
        # Initializes the Anthropic client. Make sure ANTHROPIC_API_KEY is in your .env or environment
        if Anthropic:
            self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        else:
            self.client = None


    def chat(self, prompt, **kwargs):
        """
        Calls the Claude API using a simple string prompt.
        """
        if not self.client:
            return f"[Mocked CHAT response - anthropic not installed] length={len(prompt)}"
            
        response = self.client.messages.create(
            model=kwargs.pop("model", DEFAULT_MODEL),
            max_tokens=kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS),
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return next((b.text for b in response.content if b.type == "text"), "")


    def call(self, messages, **kwargs):
        """
        Calls the Claude API using a list of message dicts (e.g., [{"role": "user", "content": "..."}]).
        """
        if not self.client:
            return f"[Mocked CALL response - anthropic not installed] messages_count={len(messages)}"
            
        response = self.client.messages.create(
            model=kwargs.pop("model", DEFAULT_MODEL),
            max_tokens=kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS),
            messages=messages,
            **kwargs
        )
        return next((b.text for b in response.content if b.type == "text"), "")


    def structured_call(self, messages, output_format=None, **kwargs):
        """
        Calls the Claude API using a list of message dicts (e.g., [{"role": "user", "content": "..."}]).
        """
        if not self.client:
            return f"[Mocked CALL response - anthropic not installed] messages_count={len(messages)}"

        response = self.client.messages.parse(
            model=kwargs.pop("model", DEFAULT_MODEL),
            max_tokens=kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS),
            messages=messages,
            output_format=output_format,
            **kwargs
        )

        return response.parsed_output


    def structured_chat(self, prompt, schema, tool_name="extract_data", **kwargs):
        """
        Returns structured output guaranteed to match `schema` (a JSON Schema dict).
        Uses Anthropic tool-use: defines a dummy tool with the given schema,
        forces the model to 'call' it, then returns the parsed dict.
        """
        if not self.client:
            return {"_mock": True, "prompt_length": len(prompt)}

        tool_def = {
            "name": tool_name,
            "description": "Extract structured data from the analysis.",
            "input_schema": schema,
        }

        response = self.client.messages.create(
            model=kwargs.pop("model", DEFAULT_MODEL),
            max_tokens=kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS),
            messages=[{"role": "user", "content": prompt}],
            tools=[tool_def],
            tool_choice={"type": "tool", "name": tool_name},
            **kwargs,
        )

        # The model is forced to produce a tool_use block matching our schema
        for block in response.content:
            if block.type == "tool_use":
                return block.input

        return {}


# Global instance for easy import
llm = LLM()

if __name__ == "__main__":
    print(llm.chat("how are you"))
