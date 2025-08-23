# 🧠 ReAct Agent Resume Analysis System

A comprehensive AI-powered resume analysis system with ReAct (Reasoning + Acting) agents that can search, analyze, and interact with resume databases intelligently.

## 🚀 Quick Start

### 1. **Environment Setup**

```bash
# Create virtual environment
python -m venv .agentsandtools
source .agentsandtools/bin/activate  # On Windows: .agentsandtools\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install ReAct Agent specific dependencies
pip install duckduckgo-search>=3.9.6
```

### 2. **Configuration**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### 3. **Launch Application**

```bash
streamlit run frontend/app.py
```

The app will open at `http://localhost:8501`

## 📋 **Step-by-Step Usage Guide**

### **Phase 1: Data Preparation (MANDATORY FIRST STEP)**

1. **Go to Tab 1: "📚 Index Management"**
2. **Select Data Source:**
   - Use Resume.xlsx (if you have it in `data/` folder)
   - Upload your own CSV/Excel file
   - Download from URL
3. **Click "🚀 Build Index"** 
   - ✅ Wait for indexing to complete
   - ✅ You should see "Successfully indexed X resumes"

> ⚠️ **IMPORTANT**: You MUST build the index first before using any other features!

### **Phase 2: ReAct Agent Setup**

1. **Go to Tab 4: "🧠 ReAct Agent"**
2. **Configure Agent:**
   - Choose Agent Mode (Demo recommended for first use)
   - Enable "Show Reasoning Traces" to see how the agent thinks
3. **Click "🚀 Initialize Agent"**
   - ✅ Wait for "Agent initialized in demo mode!" message

### **Phase 3: Start Chatting**

Try these example queries:
- `"Find Python developers with machine learning experience"`
- `"What is the average years of experience for our candidates?"`
- `"What are the latest trends in artificial intelligence?"`
- `"Calculate the percentage of candidates with Python skills"`

## 🔧 **Troubleshooting**

### **NLTK SSL Certificate Error**
```
[nltk_data] Error loading stopwords: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Fix:**
```bash
# Run the SSL fix script
python fix_ssl_nltk.py

# OR manually disable SSL for NLTK (temporary)
python -c "
import ssl
import nltk
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')
nltk.download('punkt')
"
```

### **"System not ready" Error**

**Missing Dependencies:**
```bash
pip install duckduckgo-search>=3.9.6
pip install llama-index>=0.11.16
pip install llama-index-llms-openai>=0.2.0
```

### **"No indexed data" Warning**

**Solution:**
1. Go to Tab 1 first
2. Select a data source
3. Click "Build Index"
4. Wait for completion
5. Then go to Tab 4

### **Agent Initialization Fails**

**Common Causes:**
- Missing OpenAI API key in `.env` file
- No indexed data (see above)
- Network connectivity issues

**Fix:**
1. Check `.env` file has correct `OPENAI_API_KEY`
2. Ensure index is built (Tab 1)
3. Try "Reset Agent" and reinitialize

### **Adding New Tools:**
1. Create tool in `backend/tools/`
2. Import in `react_agent.py`
3. Add to `_create_tools()` method

### **ReAct Agent Capabilities:**
- 🧠 **Reasoning Traces**: See step-by-step thinking
- 🔍 **Resume Search**: Find candidates by skills, experience
- 🌐 **Web Search**: Answer general questions
- 🧮 **Calculations**: Perform math and statistics
- 💬 **Conversations**: Context-aware chat interface
- 📊 **Analytics**: Generate insights and reports

### **Export Options:**
- Chat history (JSON/Text)
- Search results (CSV/JSON)
- Session statistics
