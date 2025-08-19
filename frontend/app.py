"""
Streamlit Frontend for Resume Analysis System
Simplified to 2 tabs with improved functionality and OpenAI integration.
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import sys
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import (
    DEFAULT_CSV_PATH,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL,
    DEFAULT_TOP_K,
)
from backend.file_processor import ResumeProcessor
from backend.embeddings import OpenAIEmbedder
from backend.vectore_store import VectorStore
from backend.llm_client import OpenAIChatClient

# Page configuration
st.set_page_config(
    page_title="Resume Analysis System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "llm_client" not in st.session_state:
    st.session_state.llm_client = None
if "indexed_resumes" not in st.session_state:
    st.session_state.indexed_resumes = 0
if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = []
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "api_costs" not in st.session_state:
    st.session_state.api_costs = {"embeddings": 0.0, "chat": 0.0}


def initialize_components():
    """Initialize AI components."""
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found! Please add it to your .env file.")
        st.stop()

    with st.spinner("Initializing AI components..."):
        # Initialize embedder
        if st.session_state.embedder is None:
            st.session_state.embedder = OpenAIEmbedder()

        # Initialize vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()

        # Initialize LLM client
        if st.session_state.llm_client is None:
            st.session_state.llm_client = OpenAIChatClient()


def display_costs():
    """Display API usage costs in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 API Usage")

    total_cost = sum(st.session_state.api_costs.values())

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Embeddings", f"${st.session_state.api_costs['embeddings']:.4f}")
    with col2:
        st.metric("Chat", f"${st.session_state.api_costs['chat']:.4f}")

    st.sidebar.metric("Total Cost", f"${total_cost:.4f}")

    if st.sidebar.button("Reset Costs"):
        st.session_state.api_costs = {"embeddings": 0.0, "chat": 0.0}
        if st.session_state.embedder:
            st.session_state.embedder.reset_usage_stats()
        if st.session_state.llm_client:
            st.session_state.llm_client.total_cost = 0.0
        st.rerun()


def display_system_info():
    """Display system information in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ System Info")

    st.sidebar.info(f"""
    **Models:**
    - Embeddings: {OPENAI_EMBEDDING_MODEL}
    - Chat: {OPENAI_CHAT_MODEL}

    **Database:**
    - Indexed: {st.session_state.indexed_resumes} resumes
    - Status: {"🟢 Ready" if st.session_state.vector_store else "🔴 Not initialized"}
    """)


# Main UI
st.title("📚 Resume Analysis System")
st.caption("Powered by OpenAI and ChromaDB")

# Initialize components
initialize_components()

# Sidebar
st.sidebar.title("🎛️ Control Panel")
display_system_info()
display_costs()

# Main tabs
tab1, tab2 = st.tabs(["📊 Index & Search", "🤖 AI Assistant"])

# ============================
# TAB 1: Index & Search
# ============================
with tab1:
    st.header("Resume Index & Search")

    # File selection section
    with st.expander("📁 Data Source", expanded=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            source_type = st.radio(
                "Select data source:",
                ["Use default CSV", "Upload file", "Use URL"],
                horizontal=True,
            )

        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        if source_type == "Use default CSV":
            st.session_state.selected_file = DEFAULT_CSV_PATH
            st.success(f"Using: {Path(DEFAULT_CSV_PATH).name}")

        elif source_type == "Upload file":
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=["csv", "xlsx", "xls"],
                help="Upload a file containing resume data"
            )

            if uploaded_file:
                # Save uploaded file
                save_path = Path("data/uploads") / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.selected_file = str(save_path)
                st.success(f"Uploaded: {uploaded_file.name}")

        elif source_type == "Use URL":
            url = st.text_input(
                "Enter CSV/Excel URL:",
                placeholder="https://example.com/resumes.csv"
            )

            if url and st.button("Download"):
                # Here you would implement URL download
                st.info("URL download will be implemented")

    # Indexing section
    with st.expander("🔧 Build Index", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            max_resumes = st.slider(
                "Maximum resumes to index:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Limit the number of resumes to process"
            )

        with col2:
            rebuild = st.checkbox(
                "Rebuild index",
                help="Delete existing index and create new one"
            )

        with col3:
            if st.button("🚀 Build Index", type="primary", use_container_width=True):
                if not st.session_state.selected_file:
                    st.error("Please select a data source first!")
                else:
                    # Process resumes
                    with st.spinner("Loading and processing resumes..."):
                        processor = ResumeProcessor()
                        resumes, stats = processor.process_resumes(
                            st.session_state.selected_file,
                            limit=max_resumes
                        )

                    if not resumes:
                        st.error(f"No resumes found! Stats: {stats}")
                    else:
                        st.success(f"Loaded {len(resumes)} resumes")

                        # Reset index if requested
                        if rebuild:
                            with st.spinner("Resetting index..."):
                                st.session_state.vector_store.reset_collection()

                        # Generate embeddings
                        with st.spinner(f"Generating embeddings with {OPENAI_EMBEDDING_MODEL}..."):
                            texts = [r["text"] for r in resumes]

                            # Estimate cost
                            cost_estimate = st.session_state.embedder.estimate_cost(texts)
                            st.info(f"Estimated cost: ${cost_estimate['estimated_cost']:.4f}")

                            # Generate embeddings (this is no longer needed with ChromaDB's built-in)
                            # We'll let ChromaDB handle it

                        # Index documents
                        progress_bar = st.progress(0)
                        with st.spinner("Indexing documents..."):
                            added = st.session_state.vector_store.add_documents(
                                resumes,
                                batch_size=50
                            )
                            progress_bar.progress(1.0)

                        # Update stats
                        st.session_state.indexed_resumes = added

                        # Update costs (approximate)
                        if st.session_state.embedder:
                            stats = st.session_state.embedder.get_usage_stats()
                            st.session_state.api_costs["embeddings"] += stats["total_cost"]

                        st.success(f"✅ Successfully indexed {added} resumes!")
                        st.balloons()

    # Search section
    st.markdown("---")
    st.subheader("🔍 Search Resumes")

    col1, col2 = st.columns([4, 1])

    with col1:
        search_query = st.text_input(
            "Search query:",
            placeholder="e.g., Python developer with 5+ years experience in machine learning",
            help="Enter your search criteria"
        )

    with col2:
        top_k = st.number_input(
            "Results:",
            min_value=1,
            max_value=50,
            value=DEFAULT_TOP_K,
            help="Number of results to return"
        )

    # Advanced filters
    with st.expander("🎯 Advanced Filters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            min_years = st.number_input(
                "Minimum years experience:",
                min_value=0,
                max_value=50,
                value=0,
                help="Filter by years of experience"
            )

        with col2:
            required_skills = st.text_input(
                "Required skills (comma-separated):",
                placeholder="Python, Docker, AWS",
                help="Skills that must be present"
            )

        with col3:
            min_similarity = st.slider(
                "Minimum similarity:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum similarity score"
            )

    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not search_query:
            st.warning("Please enter a search query!")
        elif st.session_state.indexed_resumes == 0:
            st.error("No resumes indexed! Please build the index first.")
        else:
            with st.spinner("Searching..."):
                # Build metadata filter
                metadata_filter = {}
                if min_years > 0:
                    metadata_filter["years_experience"] = {"$gte": float(min_years)}

                # Perform search
                keywords = [s.strip() for s in required_skills.split(",")] if required_skills else []

                results = st.session_state.vector_store.hybrid_search(
                    query=search_query,
                    keywords=keywords,
                    top_k=top_k,
                    filter_metadata=metadata_filter,
                )

                # Filter by similarity
                results = [r for r in results if r.get("similarity", 0) >= min_similarity]

                st.session_state.last_search_results = results

            if results:
                st.success(f"Found {len(results)} matching resumes")

                # Display results
                for i, result in enumerate(results, 1):
                    with st.expander(
                            f"**#{i}** | {result['metadata'].get('title', 'Unknown')} | "
                            f"Score: {result.get('combined_score', result['similarity']):.3f}"
                    ):
                        # Metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Years Experience",
                                      result['metadata'].get('years_experience', 'N/A'))
                        with col2:
                            st.metric("Similarity",
                                      f"{result['similarity']:.3f}")
                        with col3:
                            if "keyword_score" in result:
                                st.metric("Keyword Match",
                                          f"{result['keyword_score']:.3f}")

                        # Skills
                        if result['metadata'].get('skills'):
                            st.write("**Skills:**", result['metadata']['skills'][:200])

                        # Text preview
                        st.write("**Preview:**")
                        st.text(result['text'][:500] + "...")

                        # Contact info if available
                        if result['metadata'].get('email') or result['metadata'].get('linkedin'):
                            st.write("**Contact:**")
                            if result['metadata'].get('email'):
                                st.write(f"📧 {result['metadata']['email']}")
                            if result['metadata'].get('linkedin'):
                                st.write(f"💼 {result['metadata']['linkedin']}")
            else:
                st.warning("No results found. Try adjusting your search criteria.")

# ============================
# TAB 2: AI Assistant
# ============================
with tab2:
    st.header("🤖 AI-Powered Resume Analysis")

    if not st.session_state.last_search_results:
        st.info("👈 Please perform a search in the 'Index & Search' tab first to analyze results.")
    else:
        st.success(f"Analyzing {len(st.session_state.last_search_results)} search results")

        # Analysis options
        col1, col2 = st.columns([3, 1])

        with col1:
            analysis_type = st.selectbox(
                "Select analysis type:",
                [
                    "Find best candidate match",
                    "Generate comparative analysis",
                    "Extract key insights",
                    "Create hiring recommendations",
                    "Custom analysis"
                ]
            )

        with col2:
            num_candidates = st.number_input(
                "Analyze top:",
                min_value=1,
                max_value=min(20, len(st.session_state.last_search_results)),
                value=min(5, len(st.session_state.last_search_results)),
                help="Number of candidates to analyze"
            )

        # Custom prompt for custom analysis
        custom_prompt = None
        if analysis_type == "Custom analysis":
            custom_prompt = st.text_area(
                "Enter your analysis requirements:",
                placeholder="What would you like to know about these candidates?",
                height=100
            )

        # Additional context
        job_requirements = st.text_area(
            "Job requirements or additional context (optional):",
            placeholder="Enter specific requirements or context for better analysis",
            height=100
        )

        # Analysis settings
        with st.expander("⚙️ Analysis Settings"):
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider(
                    "Creativity level:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Lower = more focused, Higher = more creative"
                )
            with col2:
                max_tokens = st.slider(
                    "Response length:",
                    min_value=200,
                    max_value=2000,
                    value=800,
                    step=100,
                    help="Maximum length of response"
                )

        # Perform analysis
        if st.button("🔮 Analyze Candidates", type="primary", use_container_width=True):
            # Update LLM client settings
            st.session_state.llm_client.temperature = temperature
            st.session_state.llm_client.max_tokens = max_tokens

            # Get candidates to analyze
            candidates_to_analyze = st.session_state.last_search_results[:num_candidates]

            # Prepare candidates data
            candidates_data = []
            for r in candidates_to_analyze:
                candidates_data.append({
                    "id": r["id"],
                    "title": r["metadata"].get("title", "Unknown"),
                    "years_experience": r["metadata"].get("years_experience", 0),
                    "skills": r["metadata"].get("skills", ""),
                    "text": r["text"][:1000],  # Limit text length
                    "similarity_score": r["similarity"]
                })

            with st.spinner(f"Analyzing with {OPENAI_CHAT_MODEL}..."):
                try:
                    if analysis_type == "Find best candidate match":
                        result = st.session_state.llm_client.select_best_candidate(
                            candidates_data,
                            job_requirements or "Find the most qualified candidate"
                        )

                        # Display results
                        st.markdown("### 🏆 Best Candidate Analysis")

                        if "selected_id" in result:
                            st.success(f"**Selected Candidate ID:** {result['selected_id']}")

                            if "confidence_score" in result:
                                st.metric("Confidence Score", f"{result['confidence_score']}/10")

                            if "reasoning" in result:
                                st.write("**Reasoning:**")
                                st.write(result["reasoning"])

                            if "strengths" in result:
                                st.write("**Key Strengths:**")
                                for strength in result["strengths"]:
                                    st.write(f"✅ {strength}")

                            if "concerns" in result:
                                st.write("**Potential Concerns:**")
                                for concern in result["concerns"]:
                                    st.write(f"⚠️ {concern}")

                            if "ranking" in result:
                                st.write("**Full Ranking:**")
                                for i, candidate_id in enumerate(result["ranking"], 1):
                                    st.write(f"{i}. Candidate {candidate_id}")
                        else:
                            st.error("Failed to parse analysis results")
                            st.json(result)

                    elif analysis_type == "Generate comparative analysis":
                        # Create comparison prompt
                        prompt = f"""
                        Compare these {len(candidates_data)} candidates:

                        Requirements: {job_requirements or "General comparison"}

                        Candidates:
                        {json.dumps(candidates_data, indent=2)}

                        Provide a detailed comparative analysis including:
                        1. Overall comparison matrix
                        2. Strengths and weaknesses of each
                        3. Best fit for different scenarios
                        4. Recommendations
                        """

                        response = st.session_state.llm_client.chat(
                            prompt,
                            system_prompt="You are an expert recruiter providing detailed candidate comparisons."
                        )

                        st.markdown("### 📊 Comparative Analysis")
                        st.markdown(response)

                    elif analysis_type == "Extract key insights":
                        prompt = f"""
                        Extract key insights from these {len(candidates_data)} candidates:

                        {json.dumps(candidates_data, indent=2)}

                        Provide:
                        1. Common skills and patterns
                        2. Experience distribution
                        3. Unique qualifications
                        4. Market trends observed
                        """

                        response = st.session_state.llm_client.chat(
                            prompt,
                            system_prompt="You are a talent analytics expert extracting insights from resume data."
                        )

                        st.markdown("### 💡 Key Insights")
                        st.markdown(response)

                    elif analysis_type == "Create hiring recommendations":
                        prompt = f"""
                        Based on these candidates, provide hiring recommendations:

                        Job Requirements: {job_requirements or "General technical role"}

                        Candidates:
                        {json.dumps(candidates_data, indent=2)}

                        Provide:
                        1. Top 3 recommendations with justification
                        2. Interview focus areas for each
                        3. Potential red flags to investigate
                        4. Salary range expectations
                        5. Final hiring strategy
                        """

                        response = st.session_state.llm_client.chat(
                            prompt,
                            system_prompt="You are a senior hiring manager providing strategic recommendations."
                        )

                        st.markdown("### 📋 Hiring Recommendations")
                        st.markdown(response)

                    elif analysis_type == "Custom analysis":
                        if not custom_prompt:
                            st.error("Please enter your custom analysis requirements!")
                        else:
                            prompt = f"""
                            {custom_prompt}

                            Context: {job_requirements or "N/A"}

                            Candidates:
                            {json.dumps(candidates_data, indent=2)}
                            """

                            response = st.session_state.llm_client.chat(
                                prompt,
                                system_prompt="You are an expert recruiter and analyst."
                            )

                            st.markdown("### 🎯 Custom Analysis Results")
                            st.markdown(response)

                    # Update costs
                    if st.session_state.llm_client:
                        stats = st.session_state.llm_client.get_usage_stats()
                        st.session_state.api_costs["chat"] = stats["total_cost"]

                    # Show cost for this analysis
                    st.info(f"💰 Analysis cost: ${stats['total_cost']:.4f}")

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)

        # Export results
        st.markdown("---")
        st.subheader("📥 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export to CSV", use_container_width=True):
                # Create DataFrame from results
                df = pd.DataFrame([
                    {
                        "ID": r["id"],
                        "Title": r["metadata"].get("title", ""),
                        "Years Experience": r["metadata"].get("years_experience", ""),
                        "Skills": r["metadata"].get("skills", ""),
                        "Email": r["metadata"].get("email", ""),
                        "LinkedIn": r["metadata"].get("linkedin", ""),
                        "Similarity Score": r["similarity"],
                        "Preview": r["text"][:200]
                    }
                    for r in st.session_state.last_search_results
                ])

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"resume_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("📄 Export to JSON", use_container_width=True):
                json_data = json.dumps(st.session_state.last_search_results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"resume_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Footer
st.markdown("---")
st.caption(
    "Resume Analysis System v2.0 | "
    f"Using OpenAI {OPENAI_EMBEDDING_MODEL} & {OPENAI_CHAT_MODEL} | "
    "Powered by ChromaDB"
)