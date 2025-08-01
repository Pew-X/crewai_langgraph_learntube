"""
CrewAI Agents for LinkedIn Profile Optimization
"""
import os
from crewai import Agent, Task, Crew
from tools import get_linkedin_data_tool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinkedInAnalysisAgents:
    """Collection of specialized agents for LinkedIn profile analysis and optimization"""
    
    def __init__(self):
        """Initialize the LLM and tools"""
        # Configure LLM 
        llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        if llm_provider == "azure_openai":
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            
            if not azure_api_key or not azure_endpoint:
                logger.warning("Azure OpenAI credentials missing, falling back to Ollama")
                llm_provider = "ollama"
            else:

                os.environ["AZURE_API_KEY"] = azure_api_key
                os.environ["AZURE_API_BASE"] = azure_endpoint
                os.environ["AZURE_API_VERSION"] = azure_api_version
                

                self.llm = f"azure/{azure_deployment}"
                logger.info(f"Using Azure OpenAI model: {azure_deployment} at {azure_endpoint}")
        
        elif llm_provider == "openai":

            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            
            if not openai_api_key:
                logger.warning("OpenAI API key missing, falling back to Ollama")
                llm_provider = "ollama"
            else:
                
                os.environ["OPENAI_API_KEY"] = openai_api_key
                
                self.llm = f"openai/{openai_model}"
                logger.info(f"Using OpenAI model: {openai_model}")
        
        elif llm_provider == "gemini":
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            
            if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
                logger.warning("No valid Gemini API key found, falling back to Ollama")
                llm_provider = "ollama"
            else:

                os.environ["GEMINI_API_KEY"] = gemini_api_key
                

                self.llm = "gemini/gemini-1.5-flash" 
                logger.info("Using Gemini 1.5 Flash for enhanced conversational quality")
        
        elif llm_provider == "groq":

            groq_api_key = os.getenv("GROQ_API_KEY")
            groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
            
            if not groq_api_key or groq_api_key == "your_groq_api_key_here":
                logger.warning("No valid GROQ API key found, falling back to Ollama")
                llm_provider = "ollama"
            else:

                os.environ["GROQ_API_KEY"] = groq_api_key
                
                # Use GROQ for CrewAI (string format)
                self.llm = f"groq/{groq_model}"
                logger.info(f"Using GROQ model: {groq_model} for ultra-fast inference")
        
        if llm_provider == "ollama":

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            
            
            os.environ["OLLAMA_API_BASE"] = base_url
            
            self.llm = f"ollama/{model_name}"
            logger.info(f"Using Ollama model: {model_name} at {base_url}")
        

        self.linkedin_data_tool = get_linkedin_data_tool()
        
        logger.info("LinkedIn Analysis Agents initialized successfully")
    



#----------------------------------------------------------


    def intelligent_router_agent(self) -> Agent:
        return Agent(
            role="Chief of Staff AI",
            goal="Act as the central nervous system for a career coaching service. Meticulously analyze every user request and the full conversation context to delegate the task to the perfect specialist agent. Your decisions must be precise and efficient.",
            backstory="You are a hyper-efficient Chief of Staff AI, born from the labs of a top-tier executive search firm. You've processed millions of career conversations and have an almost supernatural ability to understand user intent. You don't do the work yourself; you ensure the right expert is always on the case. Your motto is: 'The right task to the right talent, instantly.'",
            llm=self.llm, verbose=True, allow_delegation=False
        )

    def general_chat_agent(self) -> Agent:
        return Agent(
            role="Empathetic AI Career Coach",
            goal="Serve as the primary, friendly interface for the user. Handle greetings, answer general questions, and provide encouragement. When a specialized task is needed, you seamlessly set the stage for a specialist to take over, ensuring the user always feels they are talking to a single, coherent assistant.",
            backstory="You are  an AI Career Coach with a high degree of emotional intelligence. You are warm, encouraging, and an expert at active listening. Your purpose is to build rapport and trust with the user. You can handle general conversation, but you know your limits and are an expert at saying, 'That's a great question, let me bring in my specialist knowledge on that...' before a specialist agent provides a detailed answer.",
            llm=self.llm, verbose=True, allow_delegation=False
        )

    def profile_analyzer_agent(self) -> Agent:
        return Agent(
            role="Senior LinkedIn Strategist & Brand Consultant",
            goal="Conduct a brutally honest, deeply insightful, and highly actionable audit of a user's LinkedIn profile. Your analysis must go beyond generic advice and provide a strategic blueprint for personal branding.",
            backstory="You are a seasoned brand consultant who has worked with C-suite executives at Fortune 500 companies to craft their digital presence. You see a LinkedIn profile not as a resume, but as a strategic asset. You are direct, insightful, and your advice is worth thousands of dollars per hour. You don't just spot typos; you deconstruct the user's entire value proposition and rebuild it to be world-class.",
            llm=self.llm, verbose=True, allow_delegation=False
        )

    def job_fit_analyzer_agent(self) -> Agent:
        return Agent(
            role="Veteran Technical Recruiter & Hiring Manager",
            goal="Perform a meticulous, data-driven analysis of a user's profile against a target job description. Provide a 'hiring-manager-gate' perspective, giving a realistic score and a concrete action plan to bridge the gap.",
            backstory="You've spent 20 years as a hiring manager and lead recruiter at top tech companies (FAANG). You've seen hundreds of thousands of resumes and can spot a good fit—and a bad one—from a mile away. You think in terms of 'signal vs. noise'. Your analysis isn't meant to be just encouraging; it's meant to be the truth that helps a candidate actually get the interview. You are the gatekeeper, and you are sharing your secrets.",
            llm=self.llm, verbose=True, allow_delegation=False
        )
        
    def career_path_agent(self) -> Agent:
        return Agent(
            role="Executive Career & Leadership Mentor",
            goal="Design a realistic, long-term, step-by-step career trajectory for a user, moving them from their current position to their ultimate career aspiration. Your roadmap must be personalized, tactical, and inspiring.",
            backstory="You are a mentor who has guided numerous professionals to the C-suite. You don't deal in vague platitudes; you build architectural plans for careers. You understand corporate ladders, political capital, skill acquisition, and the personal development required at each stage. Your advice is a blend of a master strategist and a wise mentor, providing a clear, actionable path from A to Z.",
            llm=self.llm, verbose=True, allow_delegation=False
        )

    def conversation_summarizer_agent(self) -> Agent:
        return Agent(
            role="AI Scribe & Context Weaver",
            goal="Read a raw conversation log and weave it into a coherent, narrative summary. This summary will be used by other AI agents, so it must be dense with key facts, user goals, and conversational state.",
            backstory="You are a specialized AI designed to be the 'short-term memory' for a team of other AIs. You excel at listening to dialogue and extracting the signal from the noise, creating a perfect briefing document that another agent can read to get up to speed instantly. Accuracy and contextual richness are your prime directives.",
            llm=self.llm, verbose=True, allow_delegation=False
        )
