import requests
import json
import logging
from typing import List, Dict, Any, Optional
from core.config import config
from core.llm_prompter import LlmPrompterManager

logger = logging.getLogger("qpyt-ui")

class LlmAssistantManager:
    @staticmethod
    def _clean_response(text: str) -> str:
        if not text:
            return ""
            
        # 1. Remove <think> blocks
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # 2. Common preambles to strip
        preambles = [
            "Here is the prompt:", "Voici le prompt :", "Enhanced prompt:", 
            "The user asks:", "Sure, here's a prompt", "Okay, here is",
            "I will generate", "Let's create something like:", "We'll output just the prompt."
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that look like reasoning or conversational noise
            lower_line = line.lower().strip()
            if any(p.lower() in lower_line for p in preambles) and len(line) < 200:
                continue
            if lower_line.startswith("the user wants") or lower_line.startswith("i should"):
                continue
            cleaned_lines.append(line)
            
        result = "\n".join(cleaned_lines).strip()
        
        # 3. Strip quotes if the whole thing is quoted
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1].strip()
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1].strip()
            
        # 4. Remove Markdown bold/italic formatting (e.g., **word**)
        result = re.sub(r'\*\*|__', '', result)
            
        return result

    @classmethod
    def get_response(cls, provider: str, messages: List[Dict[str, str]], model_name: Optional[str] = None, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        # Logic_Zone: RISK | Delta_Initial: 0.7 | Resolution: convergent
        
        # Prepend system prompt if provided
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        raw_content = ""
        if provider == "local":
            raw_content = LlmPrompterManager.generate_chat_response(
                messages=full_messages,
                temperature=temperature
            )

        elif provider == "ollama":
            url = config.get("OLLAMA_URL", "http://localhost:11434/api/chat")
            model = model_name or config.get("OLLAMA_MODEL", "llama3")
            payload = {"model": model, "messages": full_messages, "stream": False, "options": {"temperature": temperature}}
            try:
                response = requests.post(url, json=payload, timeout=config.get("LLM_TIMEOUT", 120))
                response.raise_for_status()
                raw_content = response.json().get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                return f"Error contacting Ollama: {str(e)}"

        elif provider == "lm-studio":
            url = config.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
            model = model_name or "local-model"
            payload = {"model": model, "messages": full_messages, "temperature": temperature, "stream": False}
            try:
                response = requests.post(url, json=payload, timeout=config.get("LLM_TIMEOUT", 120))
                response.raise_for_status()
                raw_content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"LM Studio error: {e}")
                return f"Error contacting LM Studio: {str(e)}"

        elif provider == "gemini":
            api_key = config.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                return "Error: Gemini API Key not found in config.local.json"
            model = (model_name or "gemini-2.5-flash-lite").strip()
            # Use v1beta for newest models like Gemini 2.0 Flash
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            contents = []
            for msg in full_messages:
                role = "user" if msg["role"] in ["user", "system"] else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            payload = {"contents": contents, "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048}}
            try:
                response = requests.post(url, json=payload, timeout=config.get("LLM_TIMEOUT", 120))
                response.raise_for_status()
                raw_content = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                return f"Error contacting Gemini: {str(e)}"

        elif provider == "openai":
            api_key = config.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                return "Error: OpenAI API Key not found in config.local.json"
            url = "https://api.openai.com/v1/chat/completions"
            model = (model_name or "gpt-4o-mini").strip()
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": full_messages, "temperature": temperature}
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=config.get("LLM_TIMEOUT", 120))
                response.raise_for_status()
                raw_content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                return f"Error contacting OpenAI: {str(e)}"

        elif provider == "claude":
            api_key = config.get("CLAUDE_API_KEY", "").strip()
            if not api_key:
                return "Error: Claude API Key not found in config.local.json"
            url = "https://api.anthropic.com/v1/messages"
            model = (model_name or "claude-3-5-sonnet-20240620").strip()
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            # Anthropic has a separate system param
            chat_messages = [m for m in full_messages if m["role"] != "system"]
            sys_msg = next((m["content"] for m in full_messages if m["role"] == "system"), "")
            payload = {
                "model": model,
                "messages": chat_messages,
                "system": sys_msg,
                "max_tokens": 2048,
                "temperature": temperature
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=config.get("LLM_TIMEOUT", 120))
                response.raise_for_status()
                raw_content = response.json().get("content", [{}])[0].get("text", "")
            except Exception as e:
                logger.error(f"Claude error: {e}")
                return f"Error contacting Claude: {str(e)}"

        elif provider == "grok":
            api_key = config.get("GROK_API_KEY", "").strip()
            if not api_key:
                return "Error: Grok API Key not found in config.local.json"
            url = "https://api.x.ai/v1/chat/completions"
            model = (model_name or "grok-beta").strip()
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": full_messages, "temperature": temperature}
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=config.get("LLM_TIMEOUT", 120))
                response.raise_for_status()
                raw_content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"Grok error: {e}")
                return f"Error contacting Grok: {str(e)}"
        else:
            return f"Error: Unknown provider {provider}"

        cleaned = cls._clean_response(raw_content)
        logger.info(f"[LlmAssistant] Raw len: {len(raw_content)} -> Cleaned len: {len(cleaned)}")
        # Print a bit of the cleaned response to see if it's JSON
        print(f"[Debug] Cleaned response start: {cleaned[:100]}...")
        return cleaned
