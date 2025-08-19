import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from backend.config import (
    ROLE_SYNONYMS,
    SKILL_CATEGORIES,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_SCORE,
    SEARCH_WEIGHTS,
)
from backend.vectore_store import VectorStore
from backend.llm_client import OpenAIChatClient

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    raw_query: str
    search_text: str
    min_years: Optional[float]
    max_years: Optional[float]
    required_skills: List[str]
    preferred_skills: List[str]
    roles: List[str]
    location: Optional[str]
    education_level: Optional[str]
    certifications: List[str]
    exclude_terms: List[str]


class QueryParser:
    PATTERNS = {
        "min_years": re.compile(
            r'(?:with |having |minimum |min |at least )?(\d+)\+?\s*(?:years?|yrs?)',
            re.IGNORECASE
        ),
        "year_range": re.compile(
            r'(\d+)\s*(?:-|to)\s*(\d+)\s*(?:years?|yrs?)',
            re.IGNORECASE
        ),
        "location": re.compile(
            r'(?:in |from |based in |located in )([A-Z][a-zA-Z\s]+)(?:\s|,|$)',
            re.IGNORECASE
        ),
        "education": re.compile(
            r'\b(PhD|Ph\.D\.|Doctorate|Masters?|M\.?S\.?|M\.?A\.?|MBA|Bachelors?|B\.?S\.?|B\.?A\.?|BE|BTech|MTech)\b',
            re.IGNORECASE
        ),
        "exclude": re.compile(
            r'(?:not |no |without |exclude |except )([a-zA-Z\s,]+)',
            re.IGNORECASE
        ),
        "senior_level": re.compile(
            r'\b(senior|sr\.?|lead|principal|staff|manager|director|vp|vice president)\b',
            re.IGNORECASE
        ),
        "junior_level": re.compile(
            r'\b(junior|jr\.?|entry[- ]level|intern|fresh|graduate|trainee)\b',
            re.IGNORECASE
        ),
    }

    def parse(self, query: str) -> QueryIntent:
        query_lower = query.lower()

        min_years, max_years = self._extract_years(query)

        location = self._extract_location(query)

        education = self._extract_education(query)

        roles = self._extract_roles(query_lower)

        required_skills, preferred_skills = self._extract_skills(query_lower)

        certifications = self._extract_certifications(query)

        exclude_terms = self._extract_exclude_terms(query)

        search_text = self._clean_search_text(query)

        if self.PATTERNS["senior_level"].search(query) and min_years is None:
            min_years = 5.0
        elif self.PATTERNS["junior_level"].search(query) and max_years is None:
            max_years = 3.0

        return QueryIntent(
            raw_query=query,
            search_text=search_text,
            min_years=min_years,
            max_years=max_years,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            roles=roles,
            location=location,
            education_level=education,
            certifications=certifications,
            exclude_terms=exclude_terms,
        )

    def _extract_years(self, query: str) -> Tuple[Optional[float], Optional[float]]:
        range_match = self.PATTERNS["year_range"].search(query)
        if range_match:
            min_val = float(range_match.group(1))
            max_val = float(range_match.group(2))
            return min_val, max_val

        min_match = self.PATTERNS["min_years"].search(query)
        if min_match:
            return float(min_match.group(1)), None

        return None, None

    def _extract_location(self, query: str) -> Optional[str]:
        match = self.PATTERNS["location"].search(query)
        return match.group(1).strip() if match else None

    def _extract_education(self, query: str) -> Optional[str]:
        match = self.PATTERNS["education"].search(query)
        return match.group(0) if match else None

    def _extract_roles(self, query_lower: str) -> List[str]:
        found_roles = []

        for base_role, synonyms in ROLE_SYNONYMS.items():
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    found_roles.append(base_role)
                    break

        return list(set(found_roles))

    def _extract_skills(self, query_lower: str) -> Tuple[List[str], List[str]]:
        required = []
        preferred = []

        if "must have" in query_lower or "required" in query_lower:
            pattern = re.compile(r'(?:must have|required):?\s*([^,\.\n]+)', re.IGNORECASE)
            match = pattern.search(query_lower)
            if match:
                skills_text = match.group(1)
                required = [s.strip() for s in re.split(r'[,;]', skills_text)]

        for category, skills in SKILL_CATEGORIES.items():
            for skill in skills:
                if skill.lower() in query_lower:
                    if skill not in required:
                        preferred.append(skill)

        return required, preferred

    def _extract_certifications(self, query: str) -> List[str]:
        cert_patterns = [
            r'\b(AWS|Azure|GCP|Google Cloud)\s*(?:Certified|Certification)?\b',
            r'\b(PMP|ITIL|Scrum Master|CISSP|CCNA|CCNP)\b',
            r'\b(CPA|CFA|FRM|ACCA)\b',
        ]

        certs = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            certs.extend(matches)

        return list(set(certs))

    def _extract_exclude_terms(self, query: str) -> List[str]:
        match = self.PATTERNS["exclude"].search(query)
        if match:
            terms = match.group(1).strip()
            return [t.strip() for t in re.split(r'[,;]', terms)]
        return []

    def _clean_search_text(self, query: str) -> str:
        clean = query

        clean = self.PATTERNS["min_years"].sub("", clean)
        clean = self.PATTERNS["year_range"].sub("", clean)

        clean = self.PATTERNS["location"].sub("", clean)

        clean = self.PATTERNS["exclude"].sub("", clean)

        clean = re.sub(r'\s+', ' ', clean).strip()

        return clean


class QueryEngine:
    def __init__(
            self,
            vector_store: VectorStore,
            llm_client: Optional[OpenAIChatClient] = None,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.parser = QueryParser()

        self._cache: Dict[str, List[Dict]] = {}

    def search(
            self,
            query: str,
            top_k: int = DEFAULT_TOP_K,
            use_llm_reranking: bool = False,
            return_intent: bool = False,
    ) -> Dict[str, Any]:
        start_time = datetime.now()

        intent = self.parser.parse(query)
        logger.info(f"Parsed intent: {intent}")

        metadata_filters = self._build_metadata_filters(intent)

        keywords = self._extract_keywords(intent)

        results = self.vector_store.hybrid_search(
            query=intent.search_text or query,
            keywords=keywords,
            top_k=top_k * 2 if use_llm_reranking else top_k,
            filter_metadata=metadata_filters,
        )

        results = self._apply_filters(results, intent)

        if use_llm_reranking and self.llm_client and results:
            results = self._llm_rerank(results, intent, top_k)
        else:
            results = results[:top_k]

        search_time = (datetime.now() - start_time).total_seconds()

        response = {
            "results": results,
            "query": query,
            "total_results": len(results),
            "search_time": search_time,
        }

        if return_intent:
            response["intent"] = {
                "search_text": intent.search_text,
                "min_years": intent.min_years,
                "max_years": intent.max_years,
                "required_skills": intent.required_skills,
                "preferred_skills": intent.preferred_skills,
                "roles": intent.roles,
                "location": intent.location,
                "education": intent.education_level,
                "certifications": intent.certifications,
                "exclude_terms": intent.exclude_terms,
            }

        return response

    def _build_metadata_filters(self, intent: QueryIntent) -> Dict[str, Any]:
        filters = {}

        if intent.min_years is not None:
            filters["years_experience"] = {"$gte": intent.min_years}

        if intent.max_years is not None:
            if "years_experience" in filters:
                filters["years_experience"]["$lte"] = intent.max_years
            else:
                filters["years_experience"] = {"$lte": intent.max_years}

        if intent.roles:
            pass

        return filters

    def _extract_keywords(self, intent: QueryIntent) -> List[str]:
        keywords = []

        keywords.extend(intent.required_skills)

        for role in intent.roles:
            if role in ROLE_SYNONYMS:
                keywords.extend(ROLE_SYNONYMS[role][:3])

        keywords.extend(intent.certifications)

        return keywords

    def _apply_filters(
            self,
            results: List[Dict[str, Any]],
            intent: QueryIntent
    ) -> List[Dict[str, Any]]:
        filtered = []

        for result in results:
            if intent.exclude_terms:
                text_lower = result.get("text", "").lower()
                if any(term.lower() in text_lower for term in intent.exclude_terms):
                    continue

            if intent.required_skills:
                text_lower = result.get("text", "").lower()
                metadata = result.get("metadata", {})
                skills_text = f"{text_lower} {metadata.get('skills', '')}".lower()

                if not all(skill.lower() in skills_text for skill in intent.required_skills):
                    continue

            if intent.education_level:
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                education_text = f"{text} {metadata.get('education', '')}"

                if intent.education_level.lower() not in education_text.lower():
                    continue

            filtered.append(result)

        return filtered

    def _llm_rerank(
            self,
            results: List[Dict[str, Any]],
            intent: QueryIntent,
            top_k: int
    ) -> List[Dict[str, Any]]:
        if not self.llm_client:
            return results[:top_k]

        candidates = []
        for r in results[:min(20, len(results))]:
            candidates.append({
                "id": r["id"],
                "title": r["metadata"].get("title", ""),
                "years": r["metadata"].get("years_experience", 0),
                "skills": r["metadata"].get("skills", ""),
                "text_preview": r["text"][:500],
                "current_score": r.get("combined_score", r.get("similarity", 0)),
            })

        requirements = f"""
        Looking for: {intent.raw_query}
        Min years: {intent.min_years or "Any"}
        Required skills: {", ".join(intent.required_skills) if intent.required_skills else "Any"}
        Preferred skills: {", ".join(intent.preferred_skills) if intent.preferred_skills else "Any"}
        Roles: {", ".join(intent.roles) if intent.roles else "Any"}
        """

        try:
            ranking_result = self.llm_client.select_best_candidate(
                candidates[:top_k],
                requirements
            )

            if "ranking" in ranking_result:
                ranked_ids = ranking_result["ranking"]

                id_to_result = {r["id"]: r for r in results}

                reranked = []
                for ranked_id in ranked_ids[:top_k]:
                    if ranked_id in id_to_result:
                        reranked.append(id_to_result[ranked_id])

                for r in results:
                    if r["id"] not in ranked_ids and len(reranked) < top_k:
                        reranked.append(r)

                return reranked[:top_k]

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")

        return results[:top_k]

    def explain_result(
            self,
            result: Dict[str, Any],
            query: str
    ) -> str:
        if not self.llm_client:
            return "Match based on similarity score."

        prompt = f"""
        Explain why this candidate matches the search query.

        Query: {query}

        Candidate:
        - Title: {result['metadata'].get('title', 'N/A')}
        - Years: {result['metadata'].get('years_experience', 'N/A')}
        - Skills: {result['metadata'].get('skills', 'N/A')}

        Provide a brief 2-3 sentence explanation.
        """

        try:
            explanation = self.llm_client.chat(
                prompt,
                system_prompt="You are a helpful recruiter explaining candidate matches."
            )
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return f"Match score: {result.get('similarity', 0):.3f}"


def create_query_engine(
        vector_store: VectorStore,
        llm_client: Optional[OpenAIChatClient] = None
) -> QueryEngine:
    return QueryEngine(vector_store, llm_client)


def parse_query(query: str) -> QueryIntent:
    parser = QueryParser()
    return parser.parse(query)
