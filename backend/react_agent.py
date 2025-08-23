import logging
import asyncio
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from backend.llama_config import LLM_MODEL, cost_tracker
from backend.llama_index_store import LlamaIndexStore
from backend.tools.resume_tool import create_resume_tool, create_candidate_analysis_tool
from backend.tools.general_tool import create_web_search_tool, create_definition_tool
from backend.tools.calculator_tool import create_calculator_tool, create_statistics_tool

logger = logging.getLogger(__name__)


class ReActResumeAgent:
    def __init__(
            self,
            store: LlamaIndexStore,
            llm_model: str = LLM_MODEL,
            verbose: bool = True,
            temperature: float = 0.1,
            max_iterations: int = 10,
            timeout: float = 120.0
    ):
        self.store = store
        self.llm_model = llm_model
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.timeout = timeout

        self.logger = logging.getLogger(self.__class__.__name__)

        self.llm = OpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=1500
        )

        self.tools = self._create_tools()

        self.agent = self._create_agent()

        self.conversation_history = []
        self.session_stats = {
            "total_queries": 0,
            "tool_usage": {},
            "session_stats": datetime.now(),
        }

        self.logger.info(f"ReAct Agent Initialized with {len(self.tools)} tools")

    def _create_tools(self) -> List[FunctionTool]:
        tools = []

        try:
            if self.store.index:
                resume_search_tool = create_resume_tool(self.store)
                tools.append(resume_search_tool)

                candidate_analysis_tool = create_candidate_analysis_tool(self.store)
                tools.append(candidate_analysis_tool)

                self.logger.info("resume related tools loaded successfully")
            else:
                self.logger.warning("No index found, creating - resume tools unavailable")

        except Exception as e:
            self.logger.error(f"Failed to create tools: {e}")

        try:
            web_search_tool = create_web_search_tool()
            tools.append(web_search_tool)

            definition_tool = create_definition_tool()
            tools.append(definition_tool)

            self.logger.info("web search related tools loaded successfully")

        except ImportError:
            self.logger.warning("DuckDuckGo is not available - websearch related tools unavailable")
        except Exception as e:
            self.logger.error(f"Failed to create general tools: {e}")

        try:
            calculator_tool = create_calculator_tool()
            tools.append(calculator_tool)

            statistics_tool = create_statistics_tool()
            tools.append(statistics_tool)

            self.logger.info("math tools loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to create calculator tools: {e}")

        if not tools:
            raise RuntimeError("No tools available! Cannot create agent.")

        self.logger.info(f"Created {len(tools)} tools: {[t.metadata.name for t in tools]}")
        return tools

    def _create_agent(self) -> ReActAgent:
        system_prompt = self._get_system_prompt()

        memory = ChatMemoryBuffer.from_defaults(
            token_limit=4000
        )

        agent = ReActAgent(
            tools=self.tools,
            llm=self.llm,
            memory=memory,
            system_prompt=system_prompt,
            verbose=self.verbose,
            timeout=self.timeout,
            max_iterations=self.max_iterations,
        )

        self.logger.info(f"ReAct agent created successfully")
        return agent

    def _get_system_prompt(self) -> str:
        available_tools = [tool.metadata.name for tool in self.tools]

        system_prompt = f"""You are an intelligent Resume Analysis Assistant powered by ReAct (Reasoning and Acting).

        Your primary expertise is helping with resume and candidate analysis, but you can also assist with general questions and calculations.

        AVAILABLE TOOLS: {', '.join(available_tools)}

        TOOL USAGE GUIDELINES:
        1. **Resume Queries**: Use search_resumes for finding candidates, analyze_candidate for detailed analysis
        2. **General Questions**: Use search_web for current events, definitions, company info, or general knowledge
        3. **Calculations**: Use calculate for math operations, calculate_statistics for data analysis

        RESPONSE STRATEGY:
        - **Be Specific**: When searching resumes, use detailed queries like "Python developer with 5+ years ML experience"
        - **Show Reasoning**: Always explain your thought process before taking actions
        - **Be Helpful**: Provide actionable insights and clear explanations
        - **Stay Focused**: Prioritize resume-related tasks but handle other queries professionally

        IMPORTANT RULES:
        - Use search_resumes for ANY resume/candidate related questions
        - Use search_web ONLY for non-resume general knowledge questions
        - Always explain WHY you chose specific tools
        - Provide clear, actionable responses
        - If no candidates found, suggest alternative search terms

        Remember: You're helping HR professionals and recruiters find the best candidates efficiently."""

        return system_prompt

    async def chat(self, message: str, context: Optional[Context] = None) -> Dict[str, Any]:

        try:
            if not message or not message.strip():
                return {
                    "response": "Please provide a question or request.",
                    "error": "Empty message",
                    "reasoning_trace": [],
                }

            self.logger.info(f"Received message: {message[:100]}")

            if context is None:
                context = Context(self.agent)

            self.session_stats["total_queries"] += 1
            start_time = datetime.now()

            handler = self.agent.run(message, ctx=context)

            reasoning_trace = []
            response_text = ""

            async for event in handler.stream_events():
                if hasattr(event, "delta") and event.delta:
                    response_text += event.delta

                if hasattr(event, "tool_name"):
                    tool_usage = {
                        "tool": event.tool_name,
                        "input": getattr(event, "tool_kwargs", {}),
                        "timestamp": datetime.now().isoformat()
                    }
                    reasoning_trace.append(tool_usage)

                    tool_name = event.tool_name
                    self.session_stats["tool_usage"][tool_name] = \
                        self.session_stats["tool_usage"].get(tool_name, 0) + 1

            final_response = await handler

            response_time = (datetime.now() - start_time).total_seconds()

            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "response": str(final_response),
                "response_time": response_time,
                "tools_used": [trace["tools"] for trace in reasoning_trace],
            }

            self.conversation_history.append(conversation_entry)

            estimated_tokens = len(message.split()) + len(str(final_response).split())
            cost_tracker.track_llm(estimated_tokens // 2, estimated_tokens // 2)

            self.logger.info(f"Response completed: {response_time:.2f}s")

            return {
                "response": str(final_response),
                "reasoning_trace": reasoning_trace,
                "response_time": response_time,
                "tools_used": [trace["tools"] for trace in reasoning_trace],
                "timestamp": start_time.isoformat(),
                "success": True,
            }


        except asyncio.TimeoutError:

            error_msg = f"Agent timed out after {self.timeout}s"

            self.logger.error(error_msg)

            return {
                "response": "Sorry, the request timed out. Please try a simpler query.",
                "error": error_msg,
                "reasoning_trace": [],
                "success": False,
            }


        except Exception as e:

            error_msg = f"Error processing message: {str(e)}"

            self.logger.error(error_msg)

            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "error": error_msg,
                "reasoning_trace": [],
                "success": False,
            }



    def chat_sync(self, message: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.chat(message))


    def get_session_stats(self) -> Dict[str, Any]:

        uptime = datetime.now() - self.session_stats["session_start"]

        return {
            "session_uptime": str(uptime),
            "total_queries": self.session_stats["total_queries"],
            "tools_available": len(self.tools),
            "tool_usage": self.session_stats["tool_usage"],
            "conversation_count": len(self.conversation_history),
            "cost_summary": cost_tracker.get_summary(),
            "agent_config": {
                "llm_model": self.llm_model,
                "max_iterations": self.max_iterations,
                "timeout": self.timeout,
                "verbose": self.verbose,
            }
        }

    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()


    def clear_conversation_history(self):
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared.")


    def get_available_tools(self) -> List[Dict[str, str]]:
        return [
            {
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "parameters": str(tool.metadata.fn_schema) if hasattr(tool.metadata, "fn_schema") else "N/A",
            } for tool in self.tools
        ]


def create_react_agent(
        store: LlamaIndexStore,
        llm_model: str = LLM_MODEL,
        verbose: bool = True,
        **kwargs
) -> ReActResumeAgent:
    return ReActResumeAgent(
        store=store,
        llm_model=llm_model,
        verbose=verbose,
        **kwargs
    )