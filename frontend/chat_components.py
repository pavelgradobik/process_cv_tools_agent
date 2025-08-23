"""
Advanced Chat Components for ReAct Agent UI
Provides reusable components for the chat interface.
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Any, Optional


def display_reasoning_trace(trace: List[Dict[str, Any]], expanded: bool = False):
    """
    Display the reasoning trace of the ReAct agent.

    Args:
        trace: List of reasoning steps
        expanded: Whether to expand the trace by default
    """
    if not trace:
        st.info("💭 No reasoning trace available - agent responded directly")
        return

    with st.expander("🧠 Agent Reasoning Process", expanded=expanded):
        st.markdown("**How the agent solved your query:**")

        for i, step in enumerate(trace, 1):
            # Create a visual step indicator
            step_container = st.container()

            with step_container:
                # Step header
                col1, col2 = st.columns([3, 1])

                with col1:
                    tool_name = step.get('tool', 'Unknown Tool')
                    st.markdown(f"**🔧 Step {i}: {tool_name}**")

                with col2:
                    timestamp = step.get('timestamp', '')
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            st.caption(f"⏰ {dt.strftime('%H:%M:%S')}")
                        except:
                            st.caption(f"⏰ {timestamp}")

                # Tool input (if available)
                tool_input = step.get('input', {})
                if tool_input:
                    # Format tool input nicely
                    if isinstance(tool_input, dict):
                        # Show main parameters
                        main_params = []
                        for key, value in tool_input.items():
                            if key in ['query', 'message', 'expression', 'candidate_id']:
                                main_params.append(f"**{key}**: {value}")

                        if main_params:
                            st.markdown("📝 " + " | ".join(main_params))

                        # Show full input in expander if complex
                        if len(tool_input) > 1:
                            with st.expander(f"🔍 Full input for {tool_name}", expanded=False):
                                st.json(tool_input)
                    else:
                        st.markdown(f"📝 Input: {tool_input}")

                # Add visual separator between steps
                if i < len(trace):
                    st.markdown("↓")
                    st.markdown("")


def display_chat_message(chat_entry: Dict[str, Any], show_reasoning: bool = True, message_index: int = 0):
    """
    Display a single chat message with enhanced formatting.

    Args:
        chat_entry: Dictionary containing user message and agent response
        show_reasoning: Whether to show reasoning traces
        message_index: Index of the message in chat history
    """
    user_msg = chat_entry.get('user', '')
    agent_response = chat_entry.get('agent', {})

    # Create a styled container for the chat
    with st.container():
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)

        # User message with custom styling
        st.markdown(
            f"""
            <div style="
                background-color: #f0f2f6; 
                padding: 15px; 
                border-radius: 10px;
                margin: 5px 0;
                border-left: 4px solid #1f77b4;
            ">
                <strong>👤 You:</strong> {user_msg}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Agent response
        if agent_response.get('success', False):
            # Successful response
            response_text = agent_response.get('response', '')

            st.markdown(
                f"""
                <div style="
                    background-color: #e8f5e8; 
                    padding: 15px; 
                    border-radius: 10px;
                    margin: 5px 0;
                    border-left: 4px solid #28a745;
                ">
                    <strong>🤖 Agent:</strong> {response_text}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Response metadata in columns
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

            with col1:
                response_time = agent_response.get('response_time', 0)
                st.caption(f"⏱️ {response_time:.2f}s")

            with col2:
                tools_used = agent_response.get('tools_used', [])
                if tools_used:
                    tool_icons = {"search_resumes": "👥", "search_web": "🌐", "calculate": "🧮", "analyze_candidate": "🔍"}
                    tool_display = " ".join([f"{tool_icons.get(tool, '🔧')}{tool}" for tool in tools_used])
                    st.caption(f"🛠️ {tool_display}")
                else:
                    st.caption("💭 Direct response")

            with col3:
                timestamp = agent_response.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        st.caption(f"🕒 {dt.strftime('%H:%M')}")
                    except:
                        st.caption(f"🕒 {timestamp[:16]}")

            with col4:
                # Quick actions for this message
                if st.button("🔄", key=f"retry_{message_index}", help="Retry this query"):
                    st.session_state.user_input = user_msg
                    st.rerun()

            # Reasoning trace
            if show_reasoning:
                reasoning_trace = agent_response.get('reasoning_trace', [])
                if reasoning_trace:
                    display_reasoning_trace(reasoning_trace, expanded=False)

        else:
            # Error response
            error_msg = agent_response.get('error', 'Unknown error occurred')

            st.markdown(
                f"""
                <div style="
                    background-color: #ffe6e6; 
                    padding: 15px; 
                    border-radius: 10px;
                    margin: 5px 0;
                    border-left: 4px solid #dc3545;
                ">
                    <strong>🤖 Agent Error:</strong> {error_msg}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Separator
        st.markdown("---")


def display_agent_stats_card(stats: Dict[str, Any]):
    """
    Display agent statistics in a compact card format.

    Args:
        stats: Dictionary containing agent statistics
    """
    if not stats:
        st.info("📊 No statistics available yet")
        return

    # Overview metrics in a nice grid
    st.markdown("### 📊 Agent Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_queries = stats.get('total_queries', 0)
        st.metric(
            label="🗣️ Total Queries",
            value=total_queries,
            help="Total number of queries processed"
        )

    with col2:
        tools_available = stats.get('tools_available', 0)
        st.metric(
            label="🛠️ Tools Available",
            value=tools_available,
            help="Number of tools available to the agent"
        )

    with col3:
        uptime = stats.get('session_uptime', 'N/A')
        st.metric(
            label="⏰ Session Uptime",
            value=uptime,
            help="How long the agent has been active"
        )

    with col4:
        cost_summary = stats.get('cost_summary', {})
        total_cost = cost_summary.get('total_cost', 0)
        st.metric(
            label="💰 Total Cost",
            value=f"${total_cost:.4f}",
            help="Total API costs for this session"
        )

    # Tool usage breakdown
    tool_usage = stats.get('tool_usage', {})
    if tool_usage:
        st.markdown("### 🔧 Tool Usage")

        # Create columns for tool usage
        tools_per_row = 3
        tool_items = list(tool_usage.items())

        for i in range(0, len(tool_items), tools_per_row):
            cols = st.columns(tools_per_row)

            for j, (tool_name, usage_count) in enumerate(tool_items[i:i+tools_per_row]):
                with cols[j]:
                    # Add appropriate icons for different tools
                    tool_icons = {
                        'search_resumes': '👥',
                        'analyze_candidate': '🔍',
                        'search_web': '🌐',
                        'get_definition': '📚',
                        'calculate': '🧮',
                        'calculate_statistics': '📊'
                    }

                    icon = tool_icons.get(tool_name, '🔧')
                    display_name = tool_name.replace('_', ' ').title()

                    st.metric(
                        label=f"{icon} {display_name}",
                        value=usage_count,
                        help=f"Number of times {display_name} was used"
                    )


def display_quick_actions():
    """Display quick action buttons for common tasks."""
    st.markdown("### ⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👥 Find Developers", use_container_width=True):
            st.session_state.user_input = "Find Python developers with 3+ years experience"
            return True

    with col2:
        if st.button("📊 Get Statistics", use_container_width=True):
            st.session_state.user_input = "What are the average years of experience for our candidates?"
            return True

    with col3:
        if st.button("🌐 General Query", use_container_width=True):
            st.session_state.user_input = "What are the latest trends in software development?"
            return True

    return False


def export_chat_data(chat_history: List[Dict], agent_stats: Dict, agent_mode: str) -> Dict[str, Any]:
    """
    Prepare chat data for export.

    Args:
        chat_history: List of chat messages
        agent_stats: Agent statistics
        agent_mode: Current agent mode

    Returns:
        Dictionary containing export data
    """
    export_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "agent_mode": agent_mode,
            "total_conversations": len(chat_history),
            "export_version": "1.0"
        },
        "session_stats": agent_stats,
        "conversations": []
    }

    # Process each conversation
    for i, chat in enumerate(chat_history):
        conversation = {
            "id": i + 1,
            "user_message": chat.get('user', ''),
            "agent_response": {
                "text": chat.get('agent', {}).get('response', ''),
                "success": chat.get('agent', {}).get('success', False),
                "response_time": chat.get('agent', {}).get('response_time', 0),
                "tools_used": chat.get('agent', {}).get('tools_used', []),
                "timestamp": chat.get('agent', {}).get('timestamp', ''),
            }
        }

        # Add reasoning trace if available
        reasoning_trace = chat.get('agent', {}).get('reasoning_trace', [])
        if reasoning_trace:
            conversation["reasoning_trace"] = reasoning_trace

        # Add error info if failed
        if not conversation["agent_response"]["success"]:
            conversation["agent_response"]["error"] = chat.get('agent', {}).get('error', '')

        export_data["conversations"].append(conversation)

    return export_data


def create_text_export(export_data: Dict[str, Any]) -> str:
    """
    Create a human-readable text export of the chat.

    Args:
        export_data: Export data dictionary

    Returns:
        Formatted text string
    """
    lines = []

    # Header
    export_info = export_data.get('export_info', {})
    lines.append("=" * 60)
    lines.append("REACT AGENT CHAT EXPORT")
    lines.append("=" * 60)
    lines.append(f"Export Date: {export_info.get('timestamp', 'Unknown')}")
    lines.append(f"Agent Mode: {export_info.get('agent_mode', 'Unknown')}")
    lines.append(f"Total Conversations: {export_info.get('total_conversations', 0)}")
    lines.append("")

    # Session statistics
    session_stats = export_data.get('session_stats', {})
    if session_stats:
        lines.append("SESSION STATISTICS")
        lines.append("-" * 30)
        lines.append(f"Total Queries: {session_stats.get('total_queries', 0)}")
        lines.append(f"Tools Available: {session_stats.get('tools_available', 0)}")
        lines.append(f"Session Uptime: {session_stats.get('session_uptime', 'N/A')}")

        tool_usage = session_stats.get('tool_usage', {})
        if tool_usage:
            lines.append("Tool Usage:")
            for tool, count in tool_usage.items():
                lines.append(f"  - {tool}: {count}")

        lines.append("")

    # Conversations
    lines.append("CONVERSATIONS")
    lines.append("=" * 60)

    conversations = export_data.get('conversations', [])
    for conv in conversations:
        lines.append(f"\nConversation #{conv.get('id', 'Unknown')}")
        lines.append("-" * 40)
        lines.append(f"USER: {conv.get('user_message', '')}")

        agent_resp = conv.get('agent_response', {})
        if agent_resp.get('success', False):
            lines.append(f"AGENT: {agent_resp.get('text', '')}")

            # Add metadata
            response_time = agent_resp.get('response_time', 0)
            tools_used = agent_resp.get('tools_used', [])
            timestamp = agent_resp.get('timestamp', '')

            lines.append(f"Response Time: {response_time:.2f}s")
            if tools_used:
                lines.append(f"Tools Used: {', '.join(tools_used)}")
            if timestamp:
                lines.append(f"Timestamp: {timestamp}")

            # Add reasoning trace if available
            reasoning_trace = conv.get('reasoning_trace', [])
            if reasoning_trace:
                lines.append("\nReasoning Trace:")
                for i, step in enumerate(reasoning_trace, 1):
                    tool_name = step.get('tool', 'Unknown')
                    tool_input = step.get('input', {})
                    lines.append(f"  Step {i}: {tool_name}")
                    if tool_input:
                        lines.append(f"    Input: {tool_input}")
        else:
            lines.append(f"AGENT ERROR: {agent_resp.get('error', 'Unknown error')}")

        lines.append("")

    return "\n".join(lines)


def display_chat_controls():
    """Display chat control buttons and return which action was taken."""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    actions = {}

    with col1:
        actions['send'] = st.button("🚀 Send Message", type="primary", use_container_width=True)

    with col2:
        actions['clear'] = st.button("🧹 Clear Chat", use_container_width=True)

    with col3:
        actions['stats'] = st.button("📊 Stats", use_container_width=True)

    with col4:
        actions['export'] = st.button("📥 Export", use_container_width=True)

    return actions


def display_example_queries():
    """Display clickable example queries and return selected query."""
    st.markdown("### 💡 Example Queries")
    st.caption("Click any example to try it:")

    examples = [
        {
            "category": "👥 Resume Queries",
            "queries": [
                "Find Python developers with machine learning experience",
                "Search for HR managers with 10+ years experience",
                "Show me candidates with full-stack development skills",
                "Find data scientists with PhD degrees"
            ]
        },
        {
            "category": "📊 Analytics Queries",
            "queries": [
                "What is the average years of experience for our candidates?",
                "Calculate the percentage of candidates with Python skills",
                "Compare frontend vs backend developer experience levels",
                "Show statistics for candidates with management experience"
            ]
        },
        {
            "category": "🌐 General Queries",
            "queries": [
                "What are the latest trends in artificial intelligence?",
                "What is the difference between machine learning and deep learning?",
                "What skills are most in-demand for software developers in 2025?",
                "How does remote work affect software development teams?"
            ]
        }
    ]

    selected_query = None

    for category_info in examples:
        with st.expander(category_info["category"], expanded=False):
            for i, query in enumerate(category_info["queries"]):
                if st.button(
                    f"💭 {query}",
                    key=f"example_{category_info['category']}_{i}",
                    use_container_width=True
                ):
                    selected_query = query
                    break

        if selected_query:
            break

    return selected_query


def display_agent_welcome():
    """Display a welcome message for new users."""
    st.markdown("""
    ### 👋 Welcome to the ReAct Agent!
    
    I'm an intelligent assistant that can help you with:
    
    🎯 **Resume & Candidate Analysis**
    - Search for candidates with specific skills
    - Analyze individual candidate profiles
    - Compare multiple candidates
    
    🧮 **Data Analysis & Calculations** 
    - Calculate statistics and percentages
    - Analyze numerical data from resumes
    - Perform mathematical operations
    
    🌐 **General Knowledge & Research**
    - Answer questions about technology trends
    - Provide definitions and explanations
    - Search for current information
    
    ### 🧠 **What makes me special?**
    
    I use **ReAct (Reasoning + Acting)** which means:
    - 💭 **I show my thinking process** - you can see how I solve problems
    - 🛠️ **I choose the right tools** - I pick the best approach for each query
    - 🔄 **I can handle complex requests** - I break down multi-step problems
    
    ### 🚀 **Ready to start?**
    
    Try asking me something like:
    - *"Find Python developers with 5+ years experience"*
    - *"What percentage of our candidates have AI experience?"*
    - *"What are the latest trends in cloud computing?"*
    """)


def format_agent_response(response_text: str) -> str:
    """
    Format agent response text with better styling.

    Args:
        response_text: Raw response text from agent

    Returns:
        Formatted HTML string
    """
    # Basic markdown-like formatting
    formatted = response_text

    # Make headers bold
    formatted = formatted.replace("**", "<strong>").replace("**", "</strong>")

    # Format lists
    lines = formatted.split('\n')
    in_list = False
    result_lines = []

    for line in lines:
        if line.strip().startswith('- ') or line.strip().startswith('• '):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            list_item = line.strip()[2:]  # Remove "- " or "• "
            result_lines.append(f'<li>{list_item}</li>')
        else:
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            result_lines.append(line)

    if in_list:
        result_lines.append('</ul>')

    return '\n'.join(result_lines)


def display_conversation_stats(chat_history: List[Dict]):
    """
    Display statistics about the current conversation.

    Args:
        chat_history: List of chat messages
    """
    if not chat_history:
        st.info("💬 No conversations yet")
        return

    st.markdown("### 📈 Conversation Statistics")

    # Basic stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Messages", len(chat_history))

    with col2:
        successful_responses = sum(1 for chat in chat_history
                                 if chat.get('agent', {}).get('success', False))
        st.metric("Successful Responses", successful_responses)

    with col3:
        avg_response_time = 0
        valid_times = []
        for chat in chat_history:
            response_time = chat.get('agent', {}).get('response_time', 0)
            if response_time > 0:
                valid_times.append(response_time)

        if valid_times:
            avg_response_time = sum(valid_times) / len(valid_times)

        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")

    # Tool usage in conversations
    tool_usage = {}
    for chat in chat_history:
        tools_used = chat.get('agent', {}).get('tools_used', [])
        for tool in tools_used:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

    if tool_usage:
        st.markdown("**🔧 Tools Used in This Session:**")
        for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
            st.caption(f"• {tool.replace('_', ' ').title()}: {count} times")


def create_chat_summary(chat_history: List[Dict]) -> str:
    """
    Create a summary of the chat session.

    Args:
        chat_history: List of chat messages

    Returns:
        Text summary of the chat
    """
    if not chat_history:
        return "No conversations in this session."

    summary_parts = []
    summary_parts.append(f"Chat Session Summary ({len(chat_history)} conversations)")
    summary_parts.append("=" * 50)

    # Analyze query types
    resume_queries = 0
    general_queries = 0
    calculation_queries = 0

    for chat in chat_history:
        user_msg = chat.get('user', '').lower()
        tools_used = chat.get('agent', {}).get('tools_used', [])

        if any(tool in tools_used for tool in ['search_resumes', 'analyze_candidate']):
            resume_queries += 1
        elif any(tool in tools_used for tool in ['search_web', 'get_definition']):
            general_queries += 1
        elif any(tool in tools_used for tool in ['calculate', 'calculate_statistics']):
            calculation_queries += 1

    summary_parts.append(f"Query Types:")
    summary_parts.append(f"  - Resume/Candidate queries: {resume_queries}")
    summary_parts.append(f"  - General knowledge queries: {general_queries}")
    summary_parts.append(f"  - Calculation queries: {calculation_queries}")
    summary_parts.append("")

    # Success rate
    successful = sum(1 for chat in chat_history
                    if chat.get('agent', {}).get('success', False))
    success_rate = (successful / len(chat_history)) * 100 if chat_history else 0

    summary_parts.append(f"Success Rate: {success_rate:.1f}% ({successful}/{len(chat_history)})")

    # Average response time
    response_times = [chat.get('agent', {}).get('response_time', 0)
                     for chat in chat_history
                     if chat.get('agent', {}).get('response_time', 0) > 0]

    if response_times:
        avg_time = sum(response_times) / len(response_times)
        summary_parts.append(f"Average Response Time: {avg_time:.2f}s")

    return "\n".join(summary_parts)


def display_message_actions(message_index: int, user_message: str):
    """
    Display action buttons for a specific message.

    Args:
        message_index: Index of the message
        user_message: The original user message
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Retry", key=f"retry_msg_{message_index}",
                    help="Ask this question again"):
            st.session_state.user_input = user_message
            st.rerun()

    with col2:
        if st.button("📋 Copy", key=f"copy_msg_{message_index}",
                    help="Copy message to clipboard"):
            st.write("💾 Message copied to input field")
            st.session_state.user_input = user_message

    with col3:
        if st.button("🔗 Share", key=f"share_msg_{message_index}",
                    help="Get shareable link"):
            st.info("📤 Share functionality coming soon!")


def validate_chat_data(chat_history: List[Dict]) -> Dict[str, Any]:
    """
    Validate chat history data for consistency.

    Args:
        chat_history: List of chat messages

    Returns:
        Validation results
    """
    validation = {
        "valid": True,
        "issues": [],
        "stats": {
            "total_messages": len(chat_history),
            "valid_messages": 0,
            "corrupted_messages": 0
        }
    }

    for i, chat in enumerate(chat_history):
        try:
            # Check required fields
            if 'user' not in chat:
                validation["issues"].append(f"Message {i+1}: Missing user field")
                validation["valid"] = False
                continue

            if 'agent' not in chat:
                validation["issues"].append(f"Message {i+1}: Missing agent field")
                validation["valid"] = False
                continue

            agent_response = chat['agent']
            required_fields = ['response', 'success', 'response_time', 'tools_used', 'timestamp']

            for field in required_fields:
                if field not in agent_response:
                    validation["issues"].append(f"Message {i+1}: Missing agent.{field}")

            validation["stats"]["valid_messages"] += 1

        except Exception as e:
            validation["issues"].append(f"Message {i+1}: Corrupted data - {str(e)}")
            validation["stats"]["corrupted_messages"] += 1
            validation["valid"] = False

    return validation