import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.llms.openai import OpenAI

from backend.llama_config import (
    LLM_MODEL,
    OPENAI_API_KEY,
    QA_PROMPT_TEMPLATE,
    RESUME_ANALYSIS_PROMPT,
    CANDIDATE_SELECTION_PROMPT,
    cost_tracker,
)
from backend.llama_index_store import LlamaIndexStore

logger = logging.getLogger(__name__)


@dataclass
class QueryConfig:
    top_k: int = 8
    response_mode: str = "compact"
    streaming: bool = False
    temperature: float = 0.1
    max_tokens: int = 1000
    use_async: bool = False


class ResumeQueryEngine:
    def __init__(
            self,
            index_store: LlamaIndexStore,
            llm_model: str = LLM_MODEL,
    ):
        self.store = index_store
        self.llm = OpenAI(
            model=llm_model,
            api_key=OPENAI_API_KEY,
            temperature=0.1,
        )

        self.patterns = {
            "years": re.compile(r'(\d+)\+?\s*(?:years?|yrs?)', re.IGNORECASE),
            "skills": re.compile(r'(?:skills?|technologies|tech|knows?):?\s*([^,\n]+)', re.IGNORECASE),
            "role": re.compile(r'(?:role|position|title|job):?\s*([^,\n]+)', re.IGNORECASE),
            "location": re.compile(r'(?:in|from|location|based):?\s*([A-Z][a-zA-Z\s]+)', re.IGNORECASE),
        }

        logger.info(f"Initialized ResumeQueryEngine with model: {llm_model}")

    def parse_query(self, query: str) -> Dict[str, Any]:
        filters = {}

        years_match = self.patterns["years"].search(query)
        if years_match:
            filters["min_years"] = int(years_match.group(1))

        skills_match = self.patterns["skills"].search(query)
        if skills_match:
            skills = [s.strip() for s in skills_match.group(1).split(",")]
            filters["skills"] = skills

        role_match = self.patterns["role"].search(query)
        if role_match:
            filters["role"] = role_match.group(1).strip()

        location_match = self.patterns["location"].search(query)
        if location_match:
            filters["location"] = location_match.group(1).strip()

        return filters

    def build_metadata_filters(
            self,
            parsed_filters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not parsed_filters:
            return None

        filters = {}

        if "min_years" in parsed_filters:
            filters["years_experience"] = {"$gte": parsed_filters["min_years"]}

        if "skills" in parsed_filters:
            pass

        if "role" in parsed_filters:
            filters["roles"] = {"$contains": parsed_filters["role"]}

        return filters if filters else None

    def search(
            self,
            query: str,
            config: Optional[QueryConfig] = None,
    ) -> List[Dict[str, Any]]:
        config = config or QueryConfig()

        parsed_filters = self.parse_query(query)
        metadata_filters = self.build_metadata_filters(parsed_filters)

        results = self.store.search(
            query=query,
            top_k=config.top_k,
            filters=metadata_filters,
            return_metadata=True,
        )

        logger.info(
            f"Search executed: query='{query[:50]}...', "
            f"filters={metadata_filters}, results={len(results)}"
        )

        return results

    def analyze_candidates(
            self,
            query: str,
            requirements: str,
            top_k: int = 5,
    ) -> Dict[str, Any]:
        results = self.search(query, QueryConfig(top_k=top_k))

        if not results:
            return {
                "status": "no_candidates",
                "message": "No matching candidates found",
            }

        candidates_text = "\n\n".join([
            f"Candidate {i + 1} (ID: {r['id']}):\n"
            f"Score: {r['score']:.3f}\n"
            f"Experience: {r['metadata'].get('years_experience', 'N/A')} years\n"
            f"Skills: {r['metadata'].get('skills', 'N/A')}\n"
            f"Summary: {r['text'][:500]}..."
            for i, r in enumerate(results[:top_k])
        ])

        prompt = PromptTemplate(CANDIDATE_SELECTION_PROMPT)
        formatted_prompt = prompt.format(
            context_str=candidates_text,
            query_str=requirements,
        )

        response = self.llm.complete(formatted_prompt)

        input_tokens = len(formatted_prompt.split()) * 1.3
        output_tokens = len(str(response).split()) * 1.3
        cost_tracker.track_llm(int(input_tokens), int(output_tokens))

        return {
            "status": "success",
            "candidates_analyzed": len(results[:top_k]),
            "analysis": str(response),
            "candidate_ids": [r["id"] for r in results[:top_k]],
        }

    def compare_candidates(
            self,
            candidate_ids: List[str],
            criteria: str,
    ) -> Dict[str, Any]:
        candidates_data = []

        for candidate_id in candidate_ids:
            results = self.store.search(
                query="",
                top_k=1,
                filters={"resume_id": candidate_id},
            )

            if results:
                candidates_data.append(results[0])

        if not candidates_data:
            return {
                "status": "error",
                "message": "No candidates found with provided IDs",
            }

        comparison_text = "\n\n".join([
            f"Candidate {c['metadata'].get('resume_id', c['id'])}:\n"
            f"Title: {c['metadata'].get('title', 'N/A')}\n"
            f"Experience: {c['metadata'].get('years_experience', 'N/A')} years\n"
            f"Skills: {c['metadata'].get('skills', 'N/A')}\n"
            f"Summary: {c['text'][:400]}..."
            for c in candidates_data
        ])

        prompt = f"""
        Compare the following candidates based on: {criteria}

        Candidates:
        {comparison_text}

        Provide a detailed comparison including:
        1. Strengths of each candidate
        2. Weaknesses of each candidate
        3. Best fit for different scenarios
        4. Overall ranking with justification
        """

        response = self.llm.complete(prompt)

        return {
            "status": "success",
            "candidates_compared": len(candidates_data),
            "comparison": str(response),
        }

    def generate_insights(
            self,
            query: str,
            insight_type: str = "general",
    ) -> Dict[str, Any]:
        results = self.search(query, QueryConfig(top_k=20))

        if not results:
            return {
                "status": "no_data",
                "message": "No data available for insights",
            }

        years_experience = [
            r["metadata"].get("years_experience", 0)
            for r in results
            if r["metadata"].get("years_experience")
        ]

        skills_mentioned = {}
        for r in results:
            skills = r["metadata"].get("skills", "")
            for skill in skills.split(","):
                skill = skill.strip().lower()
                if skill:
                    skills_mentioned[skill] = skills_mentioned.get(skill, 0) + 1

        if insight_type == "skills":
            top_skills = sorted(
                skills_mentioned.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            insights = {
                "top_skills": top_skills,
                "skill_coverage": f"{len(skills_mentioned)} unique skills found",
            }

        elif insight_type == "experience":
            if years_experience:
                avg_experience = sum(years_experience) / len(years_experience)
                insights = {
                    "average_experience": round(avg_experience, 1),
                    "min_experience": min(years_experience),
                    "max_experience": max(years_experience),
                    "experience_distribution": {
                        "0-3 years": sum(1 for y in years_experience if y <= 3),
                        "4-7 years": sum(1 for y in years_experience if 3 < y <= 7),
                        "8+ years": sum(1 for y in years_experience if y > 7),
                    }
                }
            else:
                insights = {"message": "No experience data available"}

        else:
            insights = {
                "total_matches": len(results),
                "average_score": round(
                    sum(r["score"] for r in results) / len(results), 3
                ),
                "top_candidates": [
                    {
                        "id": r["id"],
                        "score": r["score"],
                        "title": r["metadata"].get("title", "N/A"),
                    }
                    for r in results[:3]
                ],
            }

        return {
            "status": "success",
            "insight_type": insight_type,
            "insights": insights,
            "based_on": f"{len(results)} resumes",
        }

    def execute_custom_query(
            self,
            query: str,
            custom_prompt: Optional[str] = None,
    ) -> str:
        if custom_prompt:
            query_engine = self.store.create_query_engine()
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": PromptTemplate(custom_prompt)}
            )
            response = query_engine.query(query)
        else:
            response = self.store.query(query)

        return str(response)


class SmartResumeAgent:
    def __init__(
            self,
            query_engine: ResumeQueryEngine,
            llm_model: str = LLM_MODEL,
    ):
        self.query_engine = query_engine
        self.llm = OpenAI(
            model=llm_model,
            api_key=OPENAI_API_KEY,
            temperature=0.2,
        )

    def find_best_match(
            self,
            requirements: str,
            constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        understanding_prompt = f"""
        Analyze these job requirements and extract:
        1. Required skills (must-have)
        2. Preferred skills (nice-to-have)
        3. Minimum years of experience
        4. Key responsibilities
        5. Industry/domain preference

        Requirements: {requirements}
        """

        understanding = self.llm.complete(understanding_prompt)

        search_query = f"{requirements} {str(constraints) if constraints else ''}"
        results = self.query_engine.search(search_query, QueryConfig(top_k=10))

        if not results:
            return {
                "status": "no_match",
                "message": "No candidates found matching the requirements",
            }

        evaluation_prompt = f"""
        Based on the requirements analysis:
        {understanding}

        Evaluate these candidates and select the best match:
        {self._format_candidates(results[:5])}

        Provide:
        1. Selected candidate ID
        2. Match score (0-10)
        3. Detailed reasoning
        4. Strengths and gaps
        """

        evaluation = self.llm.complete(evaluation_prompt)

        return {
            "status": "success",
            "requirements_analysis": str(understanding),
            "evaluation": str(evaluation),
            "candidates_considered": len(results),
        }

    def _format_candidates(self, results: List[Dict[str, Any]]) -> str:
        return "\n\n".join([
            f"Candidate {i + 1} (ID: {r['id']}):\n"
            f"Score: {r['score']:.3f}\n"
            f"Years: {r['metadata'].get('years_experience', 'N/A')}\n"
            f"Skills: {r['metadata'].get('skills', 'N/A')}\n"
            f"Text: {r['text'][:300]}..."
            for i, r in enumerate(results)
        ])


def create_resume_query_engine(store: LlamaIndexStore) -> ResumeQueryEngine:
    return ResumeQueryEngine(store)


def create_smart_agent(
        query_engine: ResumeQueryEngine
) -> SmartResumeAgent:
    return SmartResumeAgent(query_engine)
