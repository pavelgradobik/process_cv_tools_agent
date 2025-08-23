import logging
from typing import Optional, Dict, Any, List
from llama_index.core.tools import FunctionTool
from backend.llama_index_store import LlamaIndexStore
from backend.llama_query_engine import QueryConfig

logger = logging.getLogger(__name__)


class ResumeRetrievalTool:
    def __init__(self, store: LlamaIndexStore):
        self.store = store
        self.logger = logging.getLogger(self.__class__.__name__)

    def search_resumes(self, query: str, top_k: int = 5, include_details: bool = True) -> str:

        try:
            if not query or not query.strip():
                return "Enpty search query provided"

            top_k = max(1, min(int(top_k), 20))

            self.logger.info(f"Searching resumes: for query = '{query}'..., top_k = {top_k}")

            if not self.store.index:
                return "Error: No resumes have been indexed yet"

            results = self.store.search(
                query=query,
                top_k=top_k,
                return_metadata=True,
            )

            if not results:
                return "No candidates found"

            formatted_results = self._format_search_results(results, include_details)

            self.logger.info(f"Search results: {len(results)} for {query}")

        except Exception as e:
            error_msg = f"Error searching resumes: {e}"
            self.logger.error(error_msg)
            return error_msg

    def _format_search_results(self, results: List[Dict[str, Any]], include_details: bool):

        if not results:
            return "No candidates found"

        formatted_parts = [
            f"Found {len(results)} candidates",
            "=" * 50
        ]

        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})

            candidate_info = [
                f"  Candidate #{i} (ID: {result.get('id', 'Unknown')})",
                f"  Match Score: {result.get('score', 0):.3f}",
                f"  Title: {metadata.get('title', 'Not specified')}",
                f"  Experience: {metadata.get('years_experience', 'Not specified')} years"
            ]

            if include_details:
                skills = metadata.get('skills', 'Not specified')
                if skills and len(skills) > 100:
                    skills = skills[:100] + "..."

                text_preview = result.get("text", "")

                if text_preview:
                    preview = text_preview[:200] + " ... " if len(text_preview) > 200 else text_preview
                    candidate_info.append(f"Preview\n {preview}")

            formatted_parts.extend(candidate_info)

        formatted_parts.append("\n" + "=" * 50)
        formatted_parts.append(f"💡 Tip: Use analyze_candidate(<id>) for detailed analysis of a specific candidate.")

        return "\n".join(formatted_parts)

    def analyze_candidate(self, candidate_id: str, requirements: str) -> str:

        try:
            if not candidate_id or not candidate_id.strip():
                return "Error: No Candidate ID provided or provide a valid candidate ID"

            results = self.store.search(
                query=f"Candidate #{candidate_id}",
                top_k=20,
                return_metadata=True,
            )

            candidate = None

            for result in results:
                if result.get("id") == candidate_id:
                    candidate = result
                    break

            if not candidate:
                return f"Candidate not found with candidate '{candidate_id}'"

            metadata = candidate.get("metadata", {})

            analysis_parts = [
                f"  Detailed Analysis for Candidate {candidate_id}",
                "=" * 50,
                f"  Title: {metadata.get('title', 'Not specified')}",
                f"  Years of Experience: {metadata.get('years_experience', 'Not specified')}",
                f"  Skills: {metadata.get('skills', 'Not specified')}",
                f"  Education: {metadata.get('education', 'Not specified')}",
                f"  Certifications: {metadata.get('certifications', 'Not specified')}",
                f"  Email: {metadata.get('email', 'Not provided')}",
                f"  Phone: {metadata.get('phone', 'Not provided')}",
                f"  LinkedIn: {metadata.get('linkedin', 'Not provided')}",
                f"  GitHub: {metadata.get('github', 'Not provided')}",
                "",
                "Full Resume Content:",
                "-" * 30,
                candidate.get('text', 'No content available')[:1000] + "..."
            ]

            if requirements:
                analysis_parts.extend([
                    "",
                    f"Analysis Against Requirements: '{requirements}'",
                    "=" * 40,
                    "This would require LLM analysis - feature available in the main query engine."
                ])

            return "\n".join(analysis_parts)

        except Exception as e:
            error_msg = f"Error analyzing candidate: {candidate_id}"
            self.logger.error(error_msg)
            return error_msg


def create_resume_tool(store: LlamaIndexStore) -> FunctionTool:
    tool_instance = ResumeRetrievalTool(store)

    analysis_tool = FunctionTool.from_defaults(
        fn=tool_instance.analyze_candidate,
        name="analyze_candidate",
        description=(
            "Analyze a specific candidate in detail. Use this tool when you have a candidate ID "
            "and want to get comprehensive information about them, including full resume content, "
            "contact details, and skills breakdown."
        ),
    )

    return analysis_tool


def create_candidate_analysis_tool(store: LlamaIndexStore) -> FunctionTool:
    tool_instance = ResumeRetrievalTool(store)

    analysis_tool = FunctionTool.from_defaults(
        fn=tool_instance.analyze_candidate,
        name="analyze_candidate",
        description=(
            "Analyze a specific candidate in detail. Use this tool when you have a candidate ID "
            "and want to get comprehensive information about them, including full resume content, "
            "contact details, and skills breakdown."
        )
    )

    return analysis_tool
