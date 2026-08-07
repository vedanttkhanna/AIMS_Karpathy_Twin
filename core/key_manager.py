import os
import re
from groq import Groq
from config import GROQ_MODEL

class QuotaExceededError(Exception):
    pass

class InvalidApiKeyError(Exception):
    pass

class GroqResponseWrapper:
    def __init__(self, text: str):
        self.text = text

class KeyManager:
    def __init__(self):
        self.keys = []
        self.current_idx = 0

    def set_keys(self, raw_input: str):
        """Parse single or multiple Groq API keys (comma or newline separated)."""
        if not raw_input:
            return
        
        parsed = [k.strip() for k in re.split(r'[\n,\s]+', raw_input) if k.strip()]
        if parsed:
            self.keys = parsed
            self.current_idx = 0
            os.environ["GROQ_API_KEY"] = self.keys[0]

    def get_active_key(self) -> str:
        if not self.keys:
            env_key = os.getenv("GROQ_API_KEY", "").strip()
            if env_key:
                self.keys = [env_key]
            else:
                raise ValueError("No Groq API key available. Please enter a key in settings.")
        return self.keys[self.current_idx % len(self.keys)]

    def generate_content(self, contents, model: str = GROQ_MODEL, **kwargs):
        """
        Executes Groq chat completion with Round-Robin key rotation and 
        automatic failover if a key hits rate limits or quota errors.
        """
        if not self.keys:
            self.get_active_key()

        attempts = 0
        total_keys = len(self.keys)
        errors = []

        while attempts < total_keys:
            key = self.keys[self.current_idx % total_keys]
            try:
                client = Groq(api_key=key)
                
                messages = []
                if isinstance(contents, str):
                    messages = [{"role": "user", "content": contents}]
                elif isinstance(contents, list):
                    messages = contents
                else:
                    messages = [{"role": "user", "content": str(contents)}]

                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                )
                
                text = completion.choices[0].message.content or ""
                
                # Success! Rotate index to next key for round-robin balancing
                self.current_idx = (self.current_idx + 1) % total_keys
                return GroqResponseWrapper(text)

            except Exception as e:
                err_str = str(e)
                err_lower = err_str.lower()
                print(f"[KeyManager] Groq Key {key[:8]}... attempt {attempts+1}/{total_keys} failed: {err_str}")
                errors.append(f"Key {key[:8]}...: {err_str}")

                if any(term in err_lower for term in ["invalid_api_key", "invalid api key", "unauthorized", "401"]):
                    self.current_idx = (self.current_idx + 1) % total_keys
                    attempts += 1
                    if attempts >= total_keys:
                        raise InvalidApiKeyError(f"Invalid Groq API key: {err_str}")

                elif any(term in err_lower for term in ["429", "rate_limit_exceeded", "quota", "too many requests"]):
                    self.current_idx = (self.current_idx + 1) % total_keys
                    attempts += 1
                    if attempts >= total_keys:
                        raise QuotaExceededError("Your Groq API key rate limit is reached. Please enter a fresh Groq API key.")

                else:
                    # Try next key on general API errors
                    self.current_idx = (self.current_idx + 1) % total_keys
                    attempts += 1

        raise Exception("All provided Groq API keys failed: " + " | ".join(errors))

# Global KeyManager instance
key_manager = KeyManager()
