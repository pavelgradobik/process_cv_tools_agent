import logging
from typing import Optional, List, Dict, Any
from llama_index.core.tools import FunctionTool

logger = logging.getLogger(__name__)

class GeneralKnowledgeTool:
    """ tool provide general knowledge with DuckDuckGo search"""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_engine = None

    def _get_search_engine(self):
        if self.search_engine is None:
            try:
                from duckduckgo_search import DDGS
                self.search_engine = DDGS()
                self.logger.info("DuckDuckGo search is available")
            except ImportError:
                self.logger.error("DuckDuckGo search is not available")
                raise ImportError((
                    "DuckDuckGo search not available. Please install it with: "
                    "pip install duckduckgo-search>=3.9.6"
                ))
        return self.search_engine

    def search_web(self, query: str, max_results: int = 3) -> str:

        try:
            if not query or not query.strip():
                return "Error Empty query provided"

            max_results = max(1, min(int(max_results), 5))

            self.logger.info(f"Searching web using DuckDuckGo search query={query}, max_results={max_results}")

            search_engine = self._get_search_engine()

            search_results = list(search_engine.text(
                keywords=query,
                max_results=max_results,
                region="wt-wt", #worldwide
                safesearch="moderate"
            ))

            if not search_results:
                return f"Error: no web results for query {query}"

            formatted_results = self._format_web_results(search_results, query)

            self.logger.info(f"Web results for query {query}: {formatted_results}")
            return formatted_results
        except ImportError as ie:
            return str(ie)
        except Exception as e:
            error_msg = f"Error searching web results: {e}"
            self.logger.error(error_msg)
            return error_msg

    def _format_web_results(self, results: List[Dict[str, Any]], query: str) -> str:
        if not results:
            return f"No web results for query {query} found"

        formatted_parts = [
            f"Web Search Results for: '{query}'",
            "=" * 50
        ]

        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No URL')

            if len(body) > 200:
                body = body[:200] + "..."

            result_info = [
                f"  Result #{i}",
                f"  Title: {title}",
                f"  Summary: {body}",
                f"  Source: {href}"
            ]

            formatted_parts.extend(result_info)

        formatted_parts.append("\n" + "=" * 50)
        formatted_parts.append("Note: Information from web search - verify important details from official sources.")

        return "\n".join(formatted_parts)

    def get_definition(self, term: str) -> str:
        try:
            if not term or not term.strip():
                return "Error: Please provide a valid term to define."

            definition_query = f"what is {term} definition explanation"

            return self.search_web(definition_query, max_results=2)

        except Exception as e:
            error_msg = f"Error getting definition for '{term}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def get_current_info(self, topic: str) -> str:
        try:
            if not topic or not topic.strip():
                return "Error: Please provide a valid topic."

            current_query = f"latest {topic} 2025 current trends news"

            return self.search_web(current_query, max_results=3)

        except Exception as e:
            error_msg = f"Error getting current info for '{topic}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg



def create_web_search_tool() -> FunctionTool:
    tool_instance = GeneralKnowledgeTool()

    web_search_tool = FunctionTool.from_defaults(
        fn=tool_instance.search_web,
        name="search_web",
        description=(
            "Search the internet for general information, current events, definitions, "
            "company information, or any topic not related to resumes. Use this tool "
            "when users ask general questions like 'What is AI?', 'Latest tech trends', "
            "or 'How does machine learning work?'. DO NOT use for resume-related queries."
        )
    )

    return web_search_tool


def create_definition_tool() -> FunctionTool:
    tool_instance = GeneralKnowledgeTool()

    definition_tool = FunctionTool.from_defaults(
        fn=tool_instance.get_definition,
        name="get_definition",
        description=(
            "Get the definition and explanation of a specific term or concept. "
            "Use this when users ask 'What is X?' or need explanations of technical terms."
        )
    )

    return definition_tool


def create_current_info_tool() -> FunctionTool:
    tool_instance = GeneralKnowledgeTool()

    current_info_tool = FunctionTool.from_defaults(
        fn=tool_instance.get_current_info,
        name="get_current_info",
        description=(
            "Get current information, news, or trends about a specific topic. "
            "Use this when users ask about 'latest', 'current', 'recent', or 'trending' topics."
        )
    )

    return current_info_tool


