"""
Streamlit Frontend for Resume Analysis System
Fully migrated to LlamaIndex for better document processing and retrieval.
"""

import streamlit as st
import pandas as pd
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.llama_config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    cost_tracker,
    initialize_llama_index,
)
from backend.file_processor import ResumeProcessor
from backend.llama_index_store import LlamaIndexStore
from backend.llama_query_engine import (
    ResumeQueryEngine,
    SmartResumeAgent,
    QueryConfig,
)
from backend.react_agent import (
    ReActResumeAgent,
    create_react_agent,
)
from backend.agent_config import (
    AgentConfigManager,
    AgentMode,
    AgentValidator,
    validate_agent_setup,
)
# Page configuration
st.set_page_config(
    page_title="Resume Analysis System - LlamaIndex",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "react_agent" not in st.session_state:
    st.session_state.react_agent = None
if "react_chat_history" not in st.session_state:
    st.session_state.react_chat_history = []
if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = True
if "agent_stats" not in st.session_state:
    st.session_state.agent_stats = {}

# Initialize session state
if "llama_store" not in st.session_state:
    st.session_state.llama_store = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "smart_agent" not in st.session_state:
    st.session_state.smart_agent = None
if "indexed_count" not in st.session_state:
    st.session_state.indexed_count = 0
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None


def initialize_system():
    """Initialize LlamaIndex components."""
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found! Please add it to your .env file.")
        st.stop()

    with st.spinner("Initializing LlamaIndex components..."):
        # Initialize LlamaIndex settings
        initialize_llama_index()

        # Initialize store
        if st.session_state.llama_store is None:
            st.session_state.llama_store = LlamaIndexStore()

            # Try to load existing index
            existing_index = st.session_state.llama_store.load_index()
            if existing_index:
                st.session_state.indexed_count = st.session_state.llama_store.document_count

        # Initialize query engine
        if st.session_state.query_engine is None and st.session_state.llama_store:
            st.session_state.query_engine = ResumeQueryEngine(st.session_state.llama_store)
            st.session_state.smart_agent = SmartResumeAgent(st.session_state.query_engine)


def display_sidebar():
    """Display sidebar with system info and controls."""
    st.sidebar.title("🚀 LlamaIndex Control Panel")

    # System status
    st.sidebar.markdown("### 📊 System Status")

    status_color = "🟢" if st.session_state.indexed_count > 0 else "🔴"
    st.sidebar.info(f"""
    {status_color} **Status**: {"Ready" if st.session_state.indexed_count > 0 else "Not Indexed"}
    📚 **Documents**: {st.session_state.indexed_count}
    🤖 **LLM**: {LLM_MODEL}
    🔤 **Embeddings**: {EMBEDDING_MODEL}
    """)

    # ReAct Agent Status (NEW)
    st.sidebar.markdown("### 🧠 ReAct Agent")

    if st.session_state.react_agent:
        agent_stats = st.session_state.agent_stats or {}
        st.sidebar.success("🟢 Agent Active")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Queries", agent_stats.get('total_queries', 0))
        with col2:
            st.metric("Tools", agent_stats.get('tools_available', 0))

        # Tool usage summary
        tool_usage = agent_stats.get('tool_usage', {})
        if tool_usage:
            st.sidebar.markdown("**Tool Usage:**")
            for tool, count in tool_usage.items():
                st.sidebar.caption(f"• {tool}: {count}")
    else:
        st.sidebar.warning("🔴 Agent Inactive")
        if st.sidebar.button("🚀 Quick Start Agent"):
            try:
                if st.session_state.llama_store:
                    st.session_state.react_agent = quick_agent_setup(st.session_state.llama_store)
                    st.session_state.agent_stats = st.session_state.react_agent.get_session_stats()
                    st.sidebar.success("Agent activated!")
                    st.rerun()
                else:
                    st.sidebar.error("Please build index first")
            except Exception as e:
                st.sidebar.error(f"Failed: {str(e)}")

    # Cost tracking (existing code continues...)
    st.sidebar.markdown("### 💰 Cost Tracking")
    costs = cost_tracker.get_summary()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(
            "Embedding Tokens",
            f"{costs['embedding_tokens']:,}"
        )
    with col2:
        st.metric(
            "LLM Tokens",
            f"{costs['llm_input_tokens'] + costs['llm_output_tokens']:,}"
        )

    st.sidebar.metric(
        "Total Cost",
        f"${costs['total_cost']:.4f}"
    )

    if st.sidebar.button("Reset Costs"):
        cost_tracker.reset()
        st.rerun()

    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        st.slider(
            "Chunk Size",
            min_value=128,
            max_value=1024,
            value=512,
            step=64,
            key="chunk_size",
            help="Size of text chunks for indexing"
        )

        st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=256,
            value=50,
            step=10,
            key="chunk_overlap",
            help="Overlap between chunks"
        )

        st.selectbox(
            "Response Mode",
            ["compact", "tree_summarize", "no_text"],
            key="response_mode",
            help="How to synthesize responses"
        )


# Main UI
st.title("🚀 Resume Analysis System - LlamaIndex Edition")
st.caption("Powered by LlamaIndex, OpenAI, and ChromaDB")

# Initialize system
initialize_system()
display_sidebar()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Index Management", "Smart Search", "AI Agent", "ReAct Agent"])

# ============================
# TAB 1: Index Management
# ============================
with tab1:
    st.header("Document Index Management")

    # File selection
    with st.expander("📁 Data Source", expanded=True):
        source_type = st.radio(
            "Select data source:",
            ["Use Resume.xlsx", "Upload file", "Download from URL"],
            horizontal=True,
        )

        if source_type == "Use Resume.xlsx":
            data_path = Path("data/Resume.xlsx")
            if data_path.exists():
                st.session_state.selected_file = str(data_path)
                st.success(f"✅ Using: {data_path.name}")
            else:
                st.error("Resume.xlsx not found in data directory")

        elif source_type == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file",
                type=["csv", "xlsx", "xls"],
            )

            if uploaded_file:
                save_path = Path("data/uploads") / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.selected_file = str(save_path)
                st.success(f"✅ Uploaded: {uploaded_file.name}")

        else:  # Download from URL
            url = st.text_input(
                "Enter CSV/Excel URL:",
                placeholder="https://example.com/resumes.csv or .xlsx",
                help="Enter the direct link to a CSV or Excel file"
            )

            col1, col2 = st.columns([3, 1])
            with col2:
                download_btn = st.button("Download", type="primary", use_container_width=True)

            if download_btn and url:
                try:
                    import urllib.request
                    from datetime import datetime

                    # Determine file extension from URL
                    file_ext = ".xlsx" if "xlsx" in url.lower() else ".csv"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"downloaded_resume_{timestamp}{file_ext}"
                    save_path = Path("data/uploads") / filename
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download file
                    with st.spinner(f"Downloading from {url}..."):
                        urllib.request.urlretrieve(url, str(save_path))

                    st.session_state.selected_file = str(save_path)
                    st.success(f"✅ Downloaded: {filename}")

                    # Show file info
                    file_size = save_path.stat().st_size / (1024 * 1024)  # MB
                    st.info(f"File size: {file_size:.2f} MB")

                except Exception as e:
                    st.error(f"Download failed: {str(e)}")
                    st.info("Make sure the URL is a direct link to the file.")

    # Indexing controls
    with st.expander("🔧 Build Index", expanded=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            rebuild = st.checkbox(
                "Rebuild index (delete existing)",
                help="Start fresh with new index"
            )

            show_preview = st.checkbox(
                "Show data preview",
                value=True,
                help="Preview data before indexing"
            )

        with col2:
            index_button = st.button(
                "🚀 Build Index",
                type="primary",
                use_container_width=True,
            )

        if index_button:
            if not st.session_state.selected_file:
                st.error("Please select a data source!")
            else:
                # Load and process resumes
                with st.spinner("Loading resumes..."):
                    processor = ResumeProcessor()
                    resumes, stats = processor.process_resumes(
                        st.session_state.selected_file,
                        limit=None  # Process all
                    )

                if not resumes:
                    st.error(f"No resumes loaded. Stats: {stats}")
                else:
                    st.success(f"✅ Loaded {len(resumes)} resumes")

                    # Show preview
                    if show_preview:
                        st.subheader("Data Preview")
                        preview_df = pd.DataFrame([
                            {
                                "ID": r["id"],
                                "Title": r["metadata"].get("title", "N/A"),
                                "Years": r["metadata"].get("years_experience", 0),
                                "Skills": (r["metadata"].get("skills", "")[:100] + "..."),
                            }
                            for r in resumes[:5]
                        ])
                        st.dataframe(preview_df)

                    # Build index
                    with st.spinner("Building LlamaIndex..."):
                        try:
                            # Update chunk settings if changed
                            if "chunk_size" in st.session_state:
                                initialize_llama_index(
                                    chunk_size=st.session_state.chunk_size,
                                    chunk_overlap=st.session_state.chunk_overlap,
                                )

                            # Build index
                            index = st.session_state.llama_store.build_index(
                                resumes,
                                rebuild=rebuild
                            )

                            # Update query engine
                            st.session_state.query_engine = ResumeQueryEngine(
                                st.session_state.llama_store
                            )
                            st.session_state.smart_agent = SmartResumeAgent(
                                st.session_state.query_engine
                            )

                            st.session_state.indexed_count = len(resumes)

                            st.success(f"✅ Successfully indexed {len(resumes)} resumes!")

                            # Show stats
                            stats = st.session_state.llama_store.get_stats()
                            st.json(stats)

                        except Exception as e:
                            st.error(f"Indexing failed: {str(e)}")
                            st.exception(e)

# ============================
# TAB 2: Smart Search
# ============================
with tab2:
    st.header("🔍 Smart Resume Search")

    if st.session_state.indexed_count == 0:
        st.warning("⚠️ No documents indexed. Please build index first.")
    else:
        # Search interface
        search_query = st.text_area(
            "Enter your search query:",
            placeholder="e.g., Senior Python developer with 5+ years experience in machine learning and AWS",
            height=100,
        )

        # Search settings
        with st.expander("🎯 Search Settings"):
            col1, col2, col3 = st.columns(3)

            with col1:
                top_k = st.slider(
                    "Number of results",
                    min_value=1,
                    max_value=50,
                    value=10,
                )

            with col2:
                search_mode = st.selectbox(
                    "Search mode",
                    ["semantic", "hybrid", "keyword"],
                )

            with col3:
                include_analysis = st.checkbox(
                    "Include AI analysis",
                    value=False,
                )

        # Search button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if not search_query:
                st.warning("Please enter a search query!")
            else:
                with st.spinner("Searching..."):
                    try:
                        # Execute search
                        config = QueryConfig(
                            top_k=top_k,
                            response_mode=st.session_state.get("response_mode", "compact"),
                        )

                        results = st.session_state.query_engine.search(
                            search_query,
                            config=config,
                        )

                        st.session_state.last_results = results

                        if results:
                            st.success(f"Found {len(results)} matching resumes")

                            # Display results
                            for i, result in enumerate(results, 1):
                                with st.expander(
                                    f"**#{i}** | ID: {result['id']} | Score: {result['score']:.3f}"
                                ):
                                    metadata = result.get("metadata", {})

                                    # Display metadata
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Years", metadata.get("years_experience", "N/A"))
                                    with col2:
                                        st.metric("Title", metadata.get("title", "N/A"))
                                    with col3:
                                        st.metric("Score", f"{result['score']:.3f}")

                                    # Skills
                                    if metadata.get("skills"):
                                        st.write("**Skills:**", metadata["skills"][:200])

                                    # Text preview
                                    st.write("**Preview:**")
                                    st.text(result["text"][:500] + "...")

                            # Optional analysis
                            if include_analysis:
                                with st.spinner("Generating insights..."):
                                    insights = st.session_state.query_engine.generate_insights(
                                        search_query,
                                        insight_type="general"
                                    )

                                    st.subheader("📊 Search Insights")
                                    st.json(insights)
                        else:
                            st.warning("No results found. Try adjusting your query.")

                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")

# ============================
# TAB 3: AI Agent
# ============================
with tab3:
    st.header("🤖 Smart Resume Agent")

    if st.session_state.indexed_count == 0:
        st.warning("⚠️ No documents indexed. Please build index first.")
    else:
        # Agent mode selection
        agent_mode = st.selectbox(
            "Select agent mode:",
            [
                "Find Best Match",
                "Compare Candidates",
                "Analyze Requirements",
                "Generate Report",
            ]
        )

        if agent_mode == "Find Best Match":
            st.subheader("🎯 Find Best Matching Candidate")

            requirements = st.text_area(
                "Job Requirements:",
                placeholder="Describe the ideal candidate and job requirements...",
                height=150,
            )

            # Additional constraints
            with st.expander("Additional Constraints"):
                min_years = st.number_input("Minimum years experience", 0, 50, 0)
                required_skills = st.text_input(
                    "Required skills (comma-separated)",
                    placeholder="Python, AWS, Docker"
                )

            if st.button("Find Best Match", type="primary"):
                if not requirements:
                    st.warning("Please enter job requirements!")
                else:
                    with st.spinner("Analyzing candidates..."):
                        constraints = {}
                        if min_years > 0:
                            constraints["min_years"] = min_years
                        if required_skills:
                            constraints["skills"] = [s.strip() for s in required_skills.split(",")]

                        result = st.session_state.smart_agent.find_best_match(
                            requirements,
                            constraints
                        )

                        if result["status"] == "success":
                            st.success("✅ Analysis complete!")

                            # Display results
                            st.subheader("Requirements Analysis")
                            st.write(result["requirements_analysis"])

                            st.subheader("Candidate Evaluation")
                            st.write(result["evaluation"])

                            st.info(f"Considered {result['candidates_considered']} candidates")
                        else:
                            st.error(result.get("message", "Analysis failed"))

        elif agent_mode == "Compare Candidates":
            st.subheader("📊 Compare Specific Candidates")

            candidate_ids = st.text_input(
                "Enter candidate IDs (comma-separated):",
                placeholder="12345, 67890, 11111"
            )

            criteria = st.text_area(
                "Comparison Criteria:",
                placeholder="What aspects to compare?",
                height=100
            )

            if st.button("Compare Candidates", type="primary"):
                if not candidate_ids or not criteria:
                    st.warning("Please enter candidate IDs and criteria!")
                else:
                    ids = [id.strip() for id in candidate_ids.split(",")]

                    with st.spinner("Comparing candidates..."):
                        result = st.session_state.query_engine.compare_candidates(
                            ids,
                            criteria
                        )

                        if result["status"] == "success":
                            st.success("✅ Comparison complete!")
                            st.write(result["comparison"])
                        else:
                            st.error(result.get("message", "Comparison failed"))

        elif agent_mode == "Analyze Requirements":
            st.subheader("📋 Analyze Job Requirements")

            requirements = st.text_area(
                "Job Description/Requirements:",
                placeholder="Paste the full job description here...",
                height=200
            )

            if st.button("Analyze", type="primary"):
                if not requirements:
                    st.warning("Please enter requirements!")
                else:
                    with st.spinner("Analyzing requirements..."):
                        analysis = st.session_state.query_engine.analyze_candidates(
                            requirements,
                            requirements,
                            top_k=10
                        )

                        if analysis["status"] == "success":
                            st.success("✅ Analysis complete!")
                            st.write(analysis["analysis"])

                            st.subheader("Top Candidates")
                            for cid in analysis["candidate_ids"][:5]:
                                st.write(f"- Candidate {cid}")
                        else:
                            st.warning(analysis.get("message", "No candidates found"))

        elif agent_mode == "Generate Report":
            st.subheader("📄 Generate Analysis Report")

            report_type = st.selectbox(
                "Report Type:",
                ["Talent Pool Overview", "Skills Gap Analysis", "Market Insights"]
            )

            context = st.text_area(
                "Additional Context:",
                placeholder="Any specific focus areas or requirements?",
                height=100
            )

            if st.button("Generate Report", type="primary"):
                with st.spinner("Generating report..."):
                    # Generate different types of insights
                    query = context or "general analysis"

                    if report_type == "Talent Pool Overview":
                        insights = st.session_state.query_engine.generate_insights(
                            query,
                            insight_type="general"
                        )
                    elif report_type == "Skills Gap Analysis":
                        insights = st.session_state.query_engine.generate_insights(
                            query,
                            insight_type="skills"
                        )
                    else:  # Market Insights
                        insights = st.session_state.query_engine.generate_insights(
                            query,
                            insight_type="experience"
                        )

                    if insights["status"] == "success":
                        st.success("✅ Report generated!")

                        st.subheader(f"📊 {report_type}")
                        st.json(insights["insights"])

                        st.info(f"Analysis based on {insights['based_on']}")
                    else:
                        st.warning(insights.get("message", "No data available"))


# ============================
# TAB 4: 🧠 ReAct Agent (NEW)
# ============================
with tab4:
    st.header("🧠 ReAct Agent - Reasoning + Acting")
    st.caption("AI agent that shows its reasoning process while solving your queries")

    # Check system readiness
    if not validate_agent_setup():
        st.error("⚠️ ReAct Agent system not ready!")

        system_check = AgentValidator.check_system_requirements()
        missing = system_check.get('dependencies_missing', [])

        if missing:
            st.write("**Missing dependencies:**")
            for dep in missing:
                st.write(f"- {dep}")

            st.code("pip install duckduckgo-search>=3.9.6")
        st.stop()

    # Agent initialization section
    with st.expander("🤖 Agent Configuration", expanded=st.session_state.react_agent is None):
        col1, col2 = st.columns([2, 1])

        with col1:
            agent_mode = st.selectbox(
                "Agent Mode:",
                options=[mode.value for mode in AgentMode],
                index=2,  # Default to demo mode
                help="Choose the agent configuration mode"
            )

            st.session_state.show_reasoning = st.checkbox(
                "Show Reasoning Traces",
                value=True,
                help="Display the agent's step-by-step thinking process"
            )

        with col2:
            # Agent status
            if st.session_state.react_agent:
                st.success("🟢 Agent Active")
                agent_info = st.session_state.react_agent.get_session_stats()
                st.metric("Queries Handled", agent_info['total_queries'])
                st.metric("Tools Available", agent_info['tools_available'])
            else:
                st.warning("🔴 Agent Inactive")

        # Initialize or reinitialize agent
        init_col1, init_col2 = st.columns([1, 1])

        with init_col1:
            if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
                with st.spinner("Initializing ReAct Agent..."):
                    try:
                        # Get configuration for selected mode
                        config = AgentConfigManager.get_config(AgentMode(agent_mode))

                        # Create agent
                        agent = create_react_agent(
                            store=st.session_state.llama_store,
                            llm_model=config.llm_model,
                            verbose=config.verbose,
                            temperature=config.temperature,
                            max_iterations=config.max_iterations,
                            timeout=config.timeout,
                        )

                        st.session_state.react_agent = agent
                        st.session_state.agent_stats = agent.get_session_stats()

                        st.success(f"✅ Agent initialized in {agent_mode} mode!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Failed to initialize agent: {str(e)}")

        with init_col2:
            if st.button("🔄 Reset Agent", use_container_width=True):
                st.session_state.react_agent = None
                st.session_state.react_chat_history = []
                st.session_state.agent_stats = {}
                st.success("Agent reset successfully!")
                st.rerun()

    # Main chat interface
    if st.session_state.react_agent is None:
        st.info("👆 Please initialize the ReAct Agent first")
    else:
        # Chat input
        st.markdown("---")
        st.subheader("💬 Chat with ReAct Agent")

        # Example queries
        with st.expander("💡 Example Queries"):
            examples = [
                "Find Python developers with machine learning experience",
                "What is the average years of experience for our candidates?",
                "Search for HR managers with 10+ years experience",
                "What are the latest trends in artificial intelligence?",
                "Calculate the percentage of candidates with Python skills",
                "Compare frontend vs backend developer experience levels"
            ]

            cols = st.columns(2)
            for i, example in enumerate(examples):
                with cols[i % 2]:
                    if st.button(f"💭 {example}", key=f"example_{i}", use_container_width=True):
                        st.session_state.user_input = example

        # Chat input
        user_input = st.text_area(
            "Your Question:",
            value=st.session_state.get('user_input', ''),
            placeholder="Ask me anything about resumes, candidates, or general questions...",
            height=100,
            key="react_chat_input"
        )

        # Chat controls
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            send_message = st.button("🚀 Send Message", type="primary", use_container_width=True)

        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.react_chat_history = []
                st.session_state.react_agent.clear_conversation_history()
                st.success("Chat cleared!")
                st.rerun()

        with col3:
            show_stats = st.button("📊 Stats", use_container_width=True)

        with col4:
            export_chat = st.button("📥 Export", use_container_width=True)

        # Process message
        if send_message and user_input.strip():
            with st.spinner("🤖 Agent thinking..."):
                try:
                    # Get response from agent
                    response = st.session_state.react_agent.chat_sync(user_input)

                    # Add to chat history
                    chat_entry = {
                        "user": user_input,
                        "agent": response,
                        "timestamp": response.get('timestamp', ''),
                    }
                    st.session_state.react_chat_history.append(chat_entry)

                    # Update stats
                    st.session_state.agent_stats = st.session_state.react_agent.get_session_stats()

                    # Clear input
                    st.session_state.user_input = ""
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # Display chat history
        if st.session_state.react_chat_history:
            st.markdown("---")
            st.subheader("💬 Conversation History")

            # Display conversations in reverse order (most recent first)
            for i, chat in enumerate(reversed(st.session_state.react_chat_history)):
                with st.container():
                    # User message
                    st.markdown(f"**👤 You:** {chat['user']}")

                    # Agent response
                    response = chat['agent']

                    if response['success']:
                        # Main response
                        st.markdown(f"**🤖 Agent:** {response['response']}")

                        # Reasoning trace (if enabled)
                        if st.session_state.show_reasoning and response.get('reasoning_trace'):
                            with st.expander("🧠 Reasoning Trace", expanded=False):
                                trace = response['reasoning_trace']

                                if trace:
                                    for j, step in enumerate(trace, 1):
                                        st.markdown(f"**Step {j}: {step.get('tool', 'Unknown')}**")

                                        # Show tool input if available
                                        tool_input = step.get('input', {})
                                        if tool_input:
                                            with st.expander(f"Input for {step.get('tool', 'Tool')}", expanded=False):
                                                st.json(tool_input)

                                        st.caption(f"⏰ {step.get('timestamp', 'N/A')}")

                                        if j < len(trace):
                                            st.markdown("↓")
                                else:
                                    st.info("No reasoning trace available for this response")

                        # Response metadata
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.caption(f"⏱️ Response Time: {response.get('response_time', 0):.2f}s")

                        with col2:
                            tools_used = response.get('tools_used', [])
                            if tools_used:
                                st.caption(f"🛠️ Tools: {', '.join(tools_used)}")
                            else:
                                st.caption("🛠️ Tools: None")

                        with col3:
                            st.caption(f"🕒 {response.get('timestamp', 'N/A')}")

                    else:
                        # Error response
                        st.error(f"**🤖 Agent Error:** {response.get('error', 'Unknown error')}")

                    st.markdown("---")

        # Stats display
        if show_stats and st.session_state.agent_stats:
            st.markdown("---")
            st.subheader("📊 Agent Statistics")

            stats = st.session_state.agent_stats

            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Queries",
                    stats.get('total_queries', 0)
                )

            with col2:
                st.metric(
                    "Tools Available",
                    stats.get('tools_available', 0)
                )

            with col3:
                st.metric(
                    "Session Uptime",
                    stats.get('session_uptime', 'N/A')
                )

            with col4:
                cost_summary = stats.get('cost_summary', {})
                total_cost = cost_summary.get('total_cost', 0)
                st.metric(
                    "Total Cost",
                    f"${total_cost:.4f}"
                )

            # Tool usage breakdown
            tool_usage = stats.get('tool_usage', {})
            if tool_usage:
                st.markdown("**🛠️ Tool Usage:**")

                for tool, count in tool_usage.items():
                    st.metric(tool, count)

            # Agent configuration
            agent_config = stats.get('agent_config', {})
            if agent_config:
                with st.expander("⚙️ Agent Configuration"):
                    st.json(agent_config)

        # Export functionality
        if export_chat and st.session_state.react_chat_history:
            st.markdown("---")
            st.subheader("📥 Export Chat History")

            # Prepare export data
            export_data = {
                "chat_history": st.session_state.react_chat_history,
                "session_stats": st.session_state.agent_stats,
                "export_timestamp": datetime.now().isoformat(),
                "agent_mode": agent_mode,
            }

            # JSON export
            col1, col2 = st.columns(2)

            with col1:
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    "📄 Download as JSON",
                    data=json_data,
                    file_name=f"react_agent_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col2:
                # Create a simplified text version
                text_data = []
                text_data.append(f"ReAct Agent Chat Export")
                text_data.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                text_data.append(f"Agent Mode: {agent_mode}")
                text_data.append("=" * 50)

                for i, chat in enumerate(st.session_state.react_chat_history, 1):
                    text_data.append(f"\nConversation {i}:")
                    text_data.append(f"User: {chat['user']}")
                    text_data.append(f"Agent: {chat['agent']['response']}")

                    if chat['agent'].get('tools_used'):
                        text_data.append(f"Tools Used: {', '.join(chat['agent']['tools_used'])}")

                    text_data.append("-" * 30)

                text_export = "\n".join(text_data)

                st.download_button(
                    "📝 Download as Text",
                    data=text_export,
                    file_name=f"react_agent_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# Export functionality
if st.session_state.last_results:
    st.markdown("---")
    st.subheader("📥 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export to CSV", use_container_width=True):
            df = pd.DataFrame([
                {
                    "ID": r["id"],
                    "Score": r["score"],
                    "Title": r.get("metadata", {}).get("title", ""),
                    "Years": r.get("metadata", {}).get("years_experience", ""),
                    "Skills": r.get("metadata", {}).get("skills", ""),
                    "Preview": r["text"][:200]
                }
                for r in st.session_state.last_results
            ])

            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("Export to JSON", use_container_width=True):
            json_data = json.dumps(st.session_state.last_results, indent=2)
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.caption(
    f"Resume Analysis System v3.0 - LlamaIndex Edition | "
    f"Models: {EMBEDDING_MODEL} & {LLM_MODEL}"
)