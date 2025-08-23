"""
Agent Configuration Panel for Streamlit UI
Provides an interactive configuration interface for ReAct agents.
"""

import streamlit as st
from typing import Dict, Any, Optional
from backend.agent_config import AgentConfigManager, AgentMode, AgentValidator


def display_agent_config_panel() -> Dict[str, Any]:
    """
    Display the agent configuration panel and return configuration settings.

    Returns:
        Dictionary containing configuration settings
    """
    st.markdown("### ⚙️ Agent Configuration")

    # Mode selection
    col1, col2 = st.columns([2, 1])

    with col1:
        mode_help = """
        **Development**: Verbose output, longer timeouts, detailed logging
        **Production**: Fast responses, minimal logging, optimized for performance  
        **Demo**: Balanced settings, good for presentations
        **Research**: Maximum capabilities, thorough analysis
        """

        selected_mode = st.selectbox(
            "Agent Mode:",
            options=[mode.value for mode in AgentMode],
            index=2,  # Default to demo
            help=mode_help
        )

        mode = AgentMode(selected_mode)
        base_config = AgentConfigManager.get_config(mode)

    with col2:
        # Quick mode descriptions
        mode_info = {
            "development": {"color": "blue", "icon": "🔧", "desc": "Debug Mode"},
            "production": {"color": "green", "icon": "🚀", "desc": "Performance Mode"},
            "demo": {"color": "orange", "icon": "🎭", "desc": "Presentation Mode"},
            "research": {"color": "purple", "icon": "🔬", "desc": "Analysis Mode"}
        }

        info = mode_info.get(selected_mode, mode_info["demo"])
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 8px;">
            <div style="font-size: 2em;">{info['icon']}</div>
            <div><strong>{info['desc']}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # Advanced configuration (collapsible)
    with st.expander("🔧 Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Performance Settings**")

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(base_config.temperature),
                step=0.1,
                help="Controls randomness in responses. 0.0 = deterministic, 1.0 = creative"
            )

            max_iterations = st.slider(
                "Max Iterations",
                min_value=3,
                max_value=20,
                value=base_config.max_iterations,
                help="Maximum reasoning steps the agent can take"
            )

            timeout = st.slider(
                "Timeout (seconds)",
                min_value=30.0,
                max_value=300.0,
                value=float(base_config.timeout),
                step=10.0,
                help="Maximum time to wait for agent response"
            )

        with col2:
            st.markdown("**Tool Settings**")

            enable_web_search = st.checkbox(
                "Enable Web Search",
                value=base_config.enable_web_search,
                help="Allow agent to search the internet for general questions"
            )

            enable_calculator = st.checkbox(
                "Enable Calculator",
                value=base_config.enable_calculator,
                help="Allow agent to perform mathematical calculations"
            )

            web_search_results = st.slider(
                "Web Search Results",
                min_value=1,
                max_value=10,
                value=base_config.web_search_results,
                help="Number of web search results to retrieve"
            )

            resume_search_results = st.slider(
                "Resume Search Results",
                min_value=1,
                max_value=20,
                value=base_config.resume_search_results,
                help="Number of resume search results to return"
            )

        # Memory and output settings
        st.markdown("**Memory & Output Settings**")
        col1, col2 = st.columns(2)

        with col1:
            memory_token_limit = st.slider(
                "Memory Token Limit",
                min_value=1000,
                max_value=8000,
                value=base_config.memory_token_limit,
                step=500,
                help="Maximum tokens to keep in conversation memory"
            )

        with col2:
            verbose = st.checkbox(
                "Verbose Output",
                value=base_config.verbose,
                help="Show detailed reasoning and debug information"
            )

    # Create configuration dictionary
    config_dict = {
        'mode': selected_mode,
        'llm_model': base_config.llm_model,
        'temperature': temperature,
        'max_iterations': max_iterations,
        'timeout': timeout,
        'enable_web_search': enable_web_search,
        'enable_calculator': enable_calculator,
        'web_search_results': web_search_results,
        'resume_search_results': resume_search_results,
        'memory_token_limit': memory_token_limit,
        'verbose': verbose,
    }

    # Validate configuration
    try:
        # Create a config object for validation
        from backend.agent_config import AgentConfig
        config_obj = AgentConfig(
            temperature=temperature,
            max_iterations=max_iterations,
            timeout=timeout,
            memory_token_limit=memory_token_limit,
            enable_web_search=enable_web_search,
            enable_calculator=enable_calculator,
            web_search_results=web_search_results,
            resume_search_results=resume_search_results,
            verbose=verbose,
        )

        validation = AgentValidator.validate_config(config_obj)

        # Display validation results
        if validation['valid']:
            st.success("✅ Configuration is valid")
        else:
            st.error("❌ Configuration has issues:")
            for issue in validation['issues']:
                st.error(f"• {issue}")

        if validation['warnings']:
            st.warning("⚠️ Configuration warnings:")
            for warning in validation['warnings']:
                st.warning(f"• {warning}")

    except Exception as e:
        st.error(f"Error validating configuration: {str(e)}")

    return config_dict


def display_system_status():
    """Display system status and requirements check."""
    st.markdown("### 🔍 System Status")

    # Check system requirements
    system_check = AgentValidator.check_system_requirements()

    if system_check['system_ready']:
        st.success("✅ System is ready for ReAct Agent deployment")
    else:
        st.error("❌ System is not ready for ReAct Agent")

    # Display detailed status
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Core Components**")

        # OpenAI status
        if system_check['openai_available']:
            st.success("✅ OpenAI LLM available")
        else:
            st.error("❌ OpenAI LLM not available")

        # LlamaIndex status
        if system_check['llamaindex_available']:
            st.success("✅ LlamaIndex core available")
        else:
            st.error("❌ LlamaIndex core not available")

    with col2:
        st.markdown("**Available Tools**")

        tools = system_check.get('tools_available', {})

        # Resume tools
        if tools.get('resume_tools', False):
            st.success("✅ Resume search tools")
        else:
            st.error("❌ Resume search tools")

        # Web search tools
        if tools.get('web_search', False):
            st.success("✅ Web search tools")
        else:
            st.warning("⚠️ Web search tools (optional)")

        # Calculator tools
        if tools.get('calculator', False):
            st.success("✅ Calculator tools")
        else:
            st.error("❌ Calculator tools")

    # Missing dependencies
    missing_deps = system_check.get('dependencies_missing', [])
    if missing_deps:
        st.markdown("**Missing Dependencies**")
        for dep in missing_deps:
            st.error(f"• {dep}")

        st.markdown("**Installation Commands:**")
        if 'duckduckgo-search' in missing_deps:
            st.code("pip install duckduckgo-search>=3.9.6")
        if any('llama-index' in dep for dep in missing_deps):
            st.code("pip install llama-index>=0.11.16")

    return system_check


def display_agent_status(agent: Optional[Any], stats: Dict[str, Any]):
    """
    Display current agent status and statistics.

    Args:
        agent: The ReAct agent instance (if active)
        stats: Agent statistics dictionary
    """
    st.markdown("### 📊 Agent Status")

    if agent is None:
        st.warning("🔴 Agent is not active")
        st.info("Initialize an agent using the configuration above")
        return

    st.success("🟢 Agent is active and ready")

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_queries = stats.get('total_queries', 0)
        st.metric("Total Queries", total_queries)

    with col2:
        tools_available = stats.get('tools_available', 0)
        st.metric("Tools Available", tools_available)

    with col3:
        uptime = stats.get('session_uptime', 'N/A')
        if isinstance(uptime, str) and ':' in uptime:
            # Format uptime nicely
            uptime = uptime.split('.')[0]  # Remove microseconds
        st.metric("Uptime", uptime)

    with col4:
        cost_summary = stats.get('cost_summary', {})
        total_cost = cost_summary.get('total_cost', 0)
        st.metric("Session Cost", f"${total_cost:.4f}")

    # Available tools
    if hasattr(agent, 'get_available_tools'):
        tools = agent.get_available_tools()
        if tools:
            with st.expander("🛠️ Available Tools", expanded=False):
                for tool in tools:
                    st.markdown(f"**{tool['name']}**: {tool['description']}")

    # Tool usage statistics
    tool_usage = stats.get('tool_usage', {})
    if tool_usage:
        st.markdown("**🔧 Tool Usage This Session:**")

        # Create a simple bar chart using metrics
        for tool_name, usage_count in tool_usage.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {tool_name.replace('_', ' ').title()}")
            with col2:
                st.metric("", usage_count)


def reset_agent_session():
    """Reset the agent session and clear all data."""
    if 'react_agent' in st.session_state:
        st.session_state.react_agent = None

    if 'react_chat_history' in st.session_state:
        st.session_state.react_chat_history = []

    if 'agent_stats' in st.session_state:
        st.session_state.agent_stats = {}

    st.success("🔄 Agent session reset successfully!")
    st.rerun()


def display_reset_controls():
    """Display reset and control buttons."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Reset Session", use_container_width=True):
            reset_agent_session()

    with col2:
        if st.button("🧹 Clear Memory", use_container_width=True):
            if st.session_state.get('react_agent'):
                st.session_state.react_agent.clear_conversation_history()
                st.success("Memory cleared!")
            else:
                st.warning("No active agent to clear")

    with col3:
        if st.button("💾 Save Config", use_container_width=True):
            # This would save current configuration
            st.info("Configuration saved! (Feature coming soon)")


def display_agent_help():
    """Display help and tips for using the ReAct agent."""
    with st.expander("❓ How to Use the ReAct Agent", expanded=False):
        st.markdown("""
        ### 🎯 **What can the ReAct Agent do?**

        **Resume & Candidate Search:**
        - *"Find Python developers with 5+ years experience"*
        - *"Search for HR managers with recruiting background"*
        - *"Show me data scientists with PhD degrees"*

        **Data Analysis & Statistics:**
        - *"What's the average experience level of our candidates?"*
        - *"Calculate percentage of candidates with AI skills"*
        - *"Compare frontend vs backend developer skills"*

        **General Knowledge & Research:**
        - *"What are current trends in machine learning?"*
        - *"Explain the difference between React and Angular"*
        - *"What skills are most in-demand for 2025?"*

        ### 🧠 **Understanding ReAct (Reasoning + Acting)**

        The agent follows this pattern:
        1. **Thought**: Analyzes your question
        2. **Action**: Chooses the right tool to use
        3. **Observation**: Processes the tool's results
        4. **Answer**: Provides a comprehensive response

        ### ⚙️ **Configuration Tips**

        - **Development Mode**: Use for testing and debugging
        - **Production Mode**: Use for fast, efficient responses
        - **Demo Mode**: Best for presentations and general use
        - **Research Mode**: Use for complex analysis tasks

        ### 🔧 **Troubleshooting**

        - **Agent timeout**: Increase timeout or reduce complexity
        - **No web results**: Check internet connection
        - **No resume results**: Ensure index is built first
        - **Tool errors**: Check dependencies are installed
        """)