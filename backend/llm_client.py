import json
import logging
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    MODEL_PRICING,
    MAX_CONTEXT_LENGTH,
)

logger = logging.getLogger(__name__)


class OpenAIChatClient:
    def __init__(
            self,
            model: str = OPENAI_CHAT_MODEL,
            api_key: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000,
    ):
        self.model = model
        self.api_key = api_key or OPENAI_API_KEY
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(api_key=self.api_key, timeout=REQUEST_TIMEOUT)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        self.conversation_history: List[Dict[str, str]] = []

        logger.info(f"Initialized OpenAIChatClient with model: {model}")

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _make_request(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> Any:

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )

            if hasattr(response, 'usage') and response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens

                if self.model in MODEL_PRICING:
                    input_cost = (response.usage.prompt_tokens / 1_000_000) * \
                                 MODEL_PRICING[self.model].get("input", 0)
                    output_cost = (response.usage.completion_tokens / 1_000_000) * \
                                  MODEL_PRICING[self.model].get("output", 0)
                    self.total_cost += input_cost + output_cost

            return response

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    def chat(
            self,
            message: str,
            system_prompt: Optional[str] = None,
            use_history: bool = False,
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if use_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": message})

        response = self._make_request(messages)

        response_text = response.choices[0].message.content

        if use_history:
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            self._trim_history()

        return response_text

    def analyze_resume(
            self,
            resume_text: str,
            query: str,
    ) -> Dict[str, Any]:
        system_prompt = """You are an expert resume analyzer and technical recruiter.
        Analyze the resume and provide structured insights.
        Always respond with valid JSON."""

        user_message = f"""
        Query: {query}

        Resume:
        {resume_text[:3000]}  # Limit to prevent token overflow

        Provide analysis in this JSON format:
        {{
            "relevance_score": 0-10,
            "matching_skills": ["skill1", "skill2"],
            "missing_skills": ["skill1", "skill2"],
            "years_experience": number,
            "summary": "brief summary",
            "recommendation": "hiring recommendation"
        }}
        """

        response = self.chat(user_message, system_prompt=system_prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "summary": response,
                "error": "Failed to parse structured response"
            }

    def select_best_candidate(
            self,
            candidates: List[Dict[str, Any]],
            requirements: str,
    ) -> Dict[str, Any]:
        system_prompt = """You are an expert technical recruiter.
        Evaluate candidates and select the best match.
        Provide clear reasoning for your selection.
        Respond with valid JSON."""

        candidates_text = "\n\n".join([
            f"Candidate {i + 1} (ID: {c.get('id', 'unknown')}):\n"
            f"Title: {c.get('title', 'N/A')}\n"
            f"Years: {c.get('years_experience', 'N/A')}\n"
            f"Skills: {c.get('skills', 'N/A')}\n"
            f"Summary: {c.get('text', '')[:500]}"
            for i, c in enumerate(candidates)
        ])

        user_message = f"""
        Requirements: {requirements}

        Candidates:
        {candidates_text}

        Select the best candidate and respond in this JSON format:
        {{
            "selected_id": "candidate_id",
            "ranking": ["id1", "id2", "id3"],
            "reasoning": "detailed explanation",
            "strengths": ["strength1", "strength2"],
            "concerns": ["concern1", "concern2"],
            "confidence_score": 0-10
        }}
        """

        response = self.chat(user_message, system_prompt=system_prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse response",
                "raw_response": response
            }

    def generate_summary(
            self,
            text: str,
            max_length: int = 200,
    ) -> str:
        prompt = f"""
        Summarize the following text in no more than {max_length} words.
        Focus on key skills, experience, and achievements.

        Text:
        {text[:4000]}
        """

        return self.chat(
            prompt,
            system_prompt="You are a professional resume summarizer."
        )

    def extract_skills(self, text: str) -> List[str]:
        prompt = f"""
        Extract all technical and professional skills from this text.
        Return ONLY a JSON array of skills, nothing else.

        Text:
        {text[:3000]}
        """

        response = self.chat(
            prompt,
            system_prompt="You are a skill extraction expert. Return only JSON."
        )

        try:
            skills = json.loads(response)
            return skills if isinstance(skills, list) else []
        except:
            return [s.strip() for s in response.split(",") if s.strip()]

    def _trim_history(self, max_messages: int = 20):
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def clear_history(self):
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": round(self.total_cost, 4),
            "cost_breakdown": {
                "input": round((self.total_input_tokens / 1_000_000) *
                               MODEL_PRICING.get(self.model, {}).get("input", 0), 4),
                "output": round((self.total_output_tokens / 1_000_000) *
                                MODEL_PRICING.get(self.model, {}).get("output", 0), 4),
            }
        }

    def estimate_cost(
            self,
            message: str,
            expected_response_tokens: int = 500
    ) -> float:
        input_tokens = len(message) // 4

        if self.model in MODEL_PRICING:
            input_cost = (input_tokens / 1_000_000) * MODEL_PRICING[self.model]["input"]
            output_cost = (expected_response_tokens / 1_000_000) * MODEL_PRICING[self.model]["output"]
            return round(input_cost + output_cost, 4)

        return 0.0


def create_chat_client(**kwargs) -> OpenAIChatClient:
    return OpenAIChatClient(**kwargs)


def quick_chat(message: str, system_prompt: Optional[str] = None) -> str:
    client = OpenAIChatClient()
    return client.chat(message, system_prompt=system_prompt)