import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from backend.llama_config import LLM_MODEL

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    DEMO = "demo"
    RESEARCH = "research"


@dataclass
class AgentConfig:
    llm_model: str = LLM_MODEL
    temperature: float = 0.1
    max_tokens: int = 1500

    verbose: bool = True
    max_iterations: int = 10
    timeout: float = 120.0

    memory_token_limit: int = 4000

    enable_web_search: bool = True
    enable_calculator: bool = True
    web_search_results: int = 3
    resume_search_results: int = 5

    streaming: bool = True
    async_mode: bool = True

    log_level: str = "INFO"
    trace_reasoning: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'llm_model': self.llm_model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'verbose': self.verbose,
            'max_iterations': self.max_iterations,
            'timeout': self.timeout,
            'memory_token_limit': self.memory_token_limit,
            'enable_web_search': self.enable_web_search,
            'enable_calculator': self.enable_calculator,
            'web_search_results': self.web_search_results,
            'resume_search_results': self.resume_search_results,
            'streaming': self.streaming,
            'async_mode': self.async_mode,
            'log_level': self.log_level,
            'trace_reasoning': self.trace_reasoning,
        }


class AgentConfigManager:
    @staticmethod
    def get_config(mode: AgentMode) -> AgentConfig:
        if mode == AgentMode.DEVELOPMENT:
            return AgentConfigManager._development_config()
        elif mode == AgentMode.PRODUCTION:
            return AgentConfigManager._production_config()
        elif mode == AgentMode.DEMO:
            return AgentConfigManager._demo_config()
        elif mode == AgentMode.RESEARCH:
            return AgentConfigManager._research_config()
        else:
            raise ValueError(f"Unknown agent mode: {mode}")

    @staticmethod
    def _development_config() -> AgentConfig:
        return AgentConfig(
            verbose=True,
            trace_reasoning=True,
            log_level="DEBUG",

            timeout=180.0,
            max_iterations=12,

            temperature=0.2,

            web_search_results=5,
            resume_search_results=8,

            enable_web_search=True,
            enable_calculator=True,
        )

    @staticmethod
    def _production_config() -> AgentConfig:
        return AgentConfig(
            verbose=False,
            trace_reasoning=False,
            log_level="WARNING",

            timeout=60.0,
            max_iterations=8,

            temperature=0.0,

            web_search_results=3,
            resume_search_results=5,

            enable_web_search=True,
            enable_calculator=True,

            memory_token_limit=3000,
            max_tokens=1000,
        )

    @staticmethod
    def _demo_config() -> AgentConfig:
        return AgentConfig(
            verbose=True,
            trace_reasoning=True,
            log_level="INFO",

            timeout=90.0,
            max_iterations=10,

            temperature=0.1,

            web_search_results=3,
            resume_search_results=5,

            enable_web_search=True,
            enable_calculator=True,

            streaming=True,
        )

    @staticmethod
    def _research_config() -> AgentConfig:
        return AgentConfig(
            verbose=True,
            trace_reasoning=True,
            log_level="DEBUG",

            timeout=300.0,
            max_iterations=15,

            temperature=0.3,

            web_search_results=5,
            resume_search_results=10,

            enable_web_search=True,
            enable_calculator=True,

            memory_token_limit=6000,
            max_tokens=2000,
        )

    @staticmethod
    def create_custom_config(**kwargs) -> AgentConfig:
        config = AgentConfigManager._production_config()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        return config


class AgentValidator:
    @staticmethod
    def validate_config(config: AgentConfig) -> Dict[str, Any]:
        issues = []
        warnings = []

        if config.timeout < 30.0:
            warnings.append("Timeout very short - may cause premature failures")
        elif config.timeout > 300.0:
            warnings.append("Timeout very long - may cause poor user experience")

        if config.max_iterations < 3:
            issues.append("Max iterations too low - agent may not complete complex tasks")
        elif config.max_iterations > 20:
            warnings.append("Max iterations very high - may cause long response times")

        if config.temperature < 0.0 or config.temperature > 1.0:
            issues.append("Temperature must be between 0.0 and 1.0")

        if config.memory_token_limit < 1000:
            warnings.append("Memory token limit very low - may lose context quickly")
        elif config.memory_token_limit > 8000:
            warnings.append("Memory token limit very high - may cause performance issues")

        if config.web_search_results < 1 or config.web_search_results > 10:
            warnings.append("Web search results should be between 1-10 for optimal performance")

        if config.resume_search_results < 1 or config.resume_search_results > 20:
            warnings.append("Resume search results should be between 1-20 for optimal performance")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "config_summary": config.to_dict()
        }

    @staticmethod
    def check_system_requirements() -> Dict[str, Any]:
        checks = {
            "openai_available": False,
            "llamaindex_available": False,
            "tools_available": {
                "resume_tools": False,
                "web_search": False,
                "calculator": False,
            },
            "dependencies_missing": []
        }

        try:
            from llama_index.llms.openai import OpenAI
            checks["openai_available"] = True
        except ImportError:
            checks["dependencies_missing"].append("llama-index-llms-openai")

        try:
            from llama_index.core.agent.workflow import ReActAgent
            checks["llamaindex_available"] = True
        except ImportError:
            checks["dependencies_missing"].append("llama-index-core")

        try:
            from backend.tools.resume_tool import create_resume_tool
            checks["tools_available"]["resume_tools"] = True
        except ImportError:
            checks["dependencies_missing"].append("Backend resume tools")

        try:
            from backend.tools.general_tool import create_web_search_tool
            checks["tools_available"]["web_search"] = True
        except ImportError:
            checks["dependencies_missing"].append("duckduckgo-search")

        try:
            from backend.tools.calculator_tool import create_calculator_tool
            checks["tools_available"]["calculator"] = True
        except ImportError:
            checks["dependencies_missing"].append("Calculator tools")

        checks["system_ready"] = (
                checks["openai_available"] and
                checks["llamaindex_available"] and
                any(checks["tools_available"].values())
        )

        return checks


def get_development_agent_config() -> AgentConfig:
    return AgentConfigManager.get_config(AgentMode.DEVELOPMENT)


def get_production_agent_config() -> AgentConfig:
    return AgentConfigManager.get_config(AgentMode.PRODUCTION)


def get_demo_agent_config() -> AgentConfig:
    return AgentConfigManager.get_config(AgentMode.DEMO)


def validate_agent_setup() -> bool:
    system_check = AgentValidator.check_system_requirements()
    return system_check["system_ready"]