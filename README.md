# 💼 LinkedIn Profile Optimizer

> **Transform your LinkedIn profile with AI-powered analysis and optimization recommendations**

A sophisticated yet user-friendly application that combines **LangGraph orchestration** with **CrewAI multi-agent intelligence** to provide personalized LinkedIn profile optimization insights.

## 🧠 Mental Model

### Core Architecture
```
User Input → LangGraph Orchestrator (State Management) → CrewAI Agents → AI Analysis → Actionable Insights
```

**LangGraph**: Acts as the "brain" - orchestrates conversation flow, maintains memory, and coordinates different analysis tasks and state management.

**CrewAI Agents**: Specialized AI agents that work together like a professional team:
- 📊 **Profile Analyzer**: Dissects your current LinkedIn profile
- ✍️ **Content Writer**: Crafts compelling headlines and summaries  
- 🎯 **Career Counselor**: Provides strategic career guidance
- 🔍 **Job Fit Analyzer**: Matches your profile against specific roles
- 🚀 **Optimization Specialist**: Delivers actionable improvement recommendations

**Memory System**: Uses SQLite to remember your conversation history and preferences across sessions

### Model-Agnostic Design
Choose your preferred AI provider based on your needs:
- 🏢 **Azure OpenAI** - Enterprise-grade, predictable pricing
- ☁️ **OpenAI** - Easy setup, pay-per-use
- 🌟 **Google Gemini** - Competitive pricing, Google ecosystem
- 🏠 **Ollama** - Free, local, privacy-focused

## ⚡ Quick Start

### Option 1: Local Development (5 minutes)
```bash
# 1. Clone and navigate
git clone <your-repo>
cd crewai_langgraph_learntube

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your preferred LLM provider settings

# 4. Run the app
streamlit run working_app.py
```

### Option 2: Docker Deployment (2 minutes)
```bash
# Quick deploy with Docker
docker-compose up --build
```

Access the app at: `http://localhost:8504`

## 🔧 Configuration Guide

### 1. Choose Your LLM Provider

Edit `.env` file and set `LLM_PROVIDER` to one of:

**For Free/Local Usage:**
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b  # Small, fast model
```

**For Production/Cloud:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini  # Cost-effective option
```

### 2. Get API Keys (if using cloud providers)

| Provider | Where to Get | Best For |
|----------|-------------|----------|
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | Quick setup, development |
| Azure OpenAI | [Azure Portal](https://portal.azure.com) | Enterprise, production |
| Google Gemini | [AI Studio](https://aistudio.google.com/app/apikey) | Google ecosystem |
| Ollama | [Local installation](https://ollama.ai) | Privacy, no costs |

### 3. Optional: LinkedIn Data Enhancement
```bash
# Add for advanced LinkedIn profile scraping
APIFY_API_TOKEN=your_apify_token_here
```

## 🚀 Usage Examples

### Basic Profile Analysis
1. **Enter LinkedIn URL** or paste profile text
2. **Ask questions** like:
   - "Analyze my profile for a software engineer role"
   - "How can I improve my headline?"
   - "What skills should I highlight for data science?"

### Advanced Optimization
1. **Job-Specific Analysis**: Paste a job description for targeted recommendations
2. **Section Rewriting**: Get AI-crafted content for headlines, summaries, or experience sections
3. **Career Guidance**: Receive strategic advice on profile positioning


## 📁 Project Structure

```
📦 linkedin-profile-optimizer/
├── 🧠 Core Intelligence
│   ├── agents.py          # CrewAI specialist agents
│   ├── graph.py           # LangGraph orchestration and state management
│   └── tools.py           # Utility functions
├── 🖥️ User Interface
│   ├── working_app.py     # Streamlit web app
│   └── .streamlit/        # UI configuration
├── 🐳 Deployment
│   ├── Dockerfile         # Container setup
│   ├── docker-compose.yml # Multi-service deployment
│   └── requirements.txt   # Python dependencies
├── 📚 Documentation
│   ├── README.md          # This guide
│   ├── DEPLOYMENT_GUIDE.md # Detailed deployment instructions
│   └── .env.example       # Configuration template
└── 💾 Data
    └── *.db              # SQLite conversation memory
```

## 🛠️ Key Features

### ✨ AI-Powered Analysis
- **Profile Scoring**: Comprehensive analysis of profile strength
- **Job Matching**: AI assessment of fit for specific roles
- **Content Generation**: Professional headlines, summaries, and descriptions
- **Strategic Insights**: Career positioning and optimization recommendations

### 🔄 Conversation Memory
- **Session Persistence**: Remembers conversation across browser sessions
- **Profile Learning**: Builds understanding of your career goals over time
- **Contextual Responses**: References previous discussions for continuity

### 🌐 Model Flexibility
- **Multi-Provider Support**: Switch between OpenAI, Azure, Gemini, or Ollama
- **Cost Control**: Choose models based on budget and performance needs
- **Fallback System**: Automatic graceful degradation if primary model fails

### 🔒 Privacy Options
- **Local Models**: Use Ollama for complete data privacy
- **Cloud Options**: Leverage powerful cloud models when needed
- **Secure Storage**: Local SQLite database for conversation history

## 🎯 Use Cases

### 🔍 Job Seekers
- Optimize profile for specific job applications
- Get industry-specific content recommendations
- Analyze profile competitiveness

### 💼 Career Changers
- Reposition profile for new industry
- Highlight transferable skills
- Strategic career narrative development

### 📈 Professionals
- Regular profile health checks
- Keep content fresh and engaging
- Stay competitive in your field

## 🚀 Deployment Options

### 🏠 Local Development
Perfect for testing and customization
```bash
streamlit run working_app.py
```

### ☁️ Cloud Deployment
Ready for production deployment on:
- Streamlit Cloud
- Heroku
- AWS/Azure/GCP
- Any Docker-compatible platform

### 🐳 Containerized
Consistent deployment across environments:
```


**Built  using LangGraph + CrewAI + Streamlit**

> *Transform your LinkedIn presence from good to extraordinary with AI-powered insights*
- **State Management**: Maintains conversation context and user data
- **Conditional Routing**: Intelligently routes user queries to appropriate agents
- **Memory System**: Persistent storage using SQLite checkpointer
- **Error Handling**: Graceful handling of API failures and timeouts

## Usage Examples

### Basic Profile Analysis
1. Enter your LinkedIn profile URL
2. Wait for the initial analysis
3. Ask: "Analyze my profile"

### Job Fit Analysis
1. After profile is loaded
2. Ask: "How do I fit for a Data Scientist role?"
3. Get detailed scoring and recommendations

### Content Enhancement
1. Ask: "Rewrite my about section"
2. Or: "Improve my headline for a Senior Developer position"
3. Receive optimized content suggestions

### Career Guidance
1. Ask: "What skills am I missing for a Product Manager role?"
2. Get personalized learning recommendations

## Troubleshooting

### Common Issues


**1. Profile Scraping Fails**
- Verify the LinkedIn URL format
- Check if the profile is public
- Ensure Apify API token has sufficient credits

**2. Slow Performance for local models**
- Consider using a more powerful Ollama model
- Monitor system resources during execution


## 🔮 Future Enhancements

- Support for multiple LinkedIn profiles
- Memory management improvements and production ready postgres backed memory
- Integration with more job boards
- Advanced analytics and reporting
- Resume generation from LinkedIn data
- Interview preparation recommendations
- Networking suggestions via LinkedIn API

