"""
Streamlit Frontend for Resume Analysis System
Fully migrated to LlamaIndex for better document processing and retrieval.
"""

import streamlit as st
import pandas as pd
import json
import time
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

# Page configuration
st.set_page_config(
    page_title="Resume Analysis System - LlamaIndex",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    # Cost tracking
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
tab1, tab2, tab3 = st.tabs(["📚 Index Management", "🔍 Smart Search", "🤖 AI Agent"])

# ============================
# TAB 1: Index Management
# ============================
with tab1:
    st.header("Document Index Management")

    # File selection
    with st.expander("📁 Data Source", expanded=True):
        source_type = st.radio(
            "Select data source:",
            ["Use Resume.xlsx", "Upload file"],
            horizontal=True,
        )

        if source_type == "Use Resume.xlsx":
            data_path = Path("data/Resume.xlsx")
            if data_path.exists():
                st.session_state.selected_file = str(data_path)
                st.success(f"✅ Using: {data_path.name}")
            else:
                st.error("Resume.xlsx not found in data directory")

        else:
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
