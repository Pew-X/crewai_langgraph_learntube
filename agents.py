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
        
        if llm_provider == "ollama":

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            
            
            os.environ["OLLAMA_API_BASE"] = base_url
            
            self.llm = f"ollama/{model_name}"
            logger.info(f"Using Ollama model: {model_name} at {base_url}")
        

        self.linkedin_data_tool = get_linkedin_data_tool()
        
        logger.info("LinkedIn Analysis Agents initialized successfully")
    
    def linkedin_data_handler_agent(self) -> Agent:
        """
        Agent responsible for handling LinkedIn profile data (URL or text)
        """
        return Agent(
            role="LinkedIn Data Processor",
            goal="To process LinkedIn profile data from either URLs or direct text input, ensuring comprehensive profile information is available for analysis.",
            backstory="""You are an expert at handling various forms of LinkedIn profile data. 
            You can work with profile URLs and scrape them when possible, or process direct 
            text input when scraping is not available. You ensure that profile data is 
            properly structured and comprehensive for optimization analysis.""",
            tools=[self.linkedin_data_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def profile_analyzer_agent(self) -> Agent:
        """
        Agent responsible for analyzing LinkedIn profiles and identifying areas for improvement
        """
        return Agent(
            role="LinkedIn Career Strategist",
            goal="""Analyze LinkedIn profiles and provide specific, actionable optimization recommendations. 
            Focus on content quality, keyword optimization, professional positioning, and strategic improvements.""",
            backstory="""You are a professional LinkedIn strategist with expertise in career development, 
            personal branding, and recruitment practices. You understand what employers and recruiters look for 
            in profiles. You provide direct, specific feedback with clear rationale and actionable steps. 
            Your analysis is thorough, objective, and focused on measurable improvements.""",
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def job_description_analyzer_agent(self) -> Agent:
        """
        Agent responsible for analyzing job requirements based on role names
        """
        return Agent(
            role="Job Requirements Specialist",
            goal="""To analyze job roles and provide comprehensive information about 
            typical requirements, skills, responsibilities, and qualifications for 
            specific positions based on industry standards and common practices.""",
            backstory="""You are a hiring manager and recruitment specialist with deep 
            knowledge of various industries and job roles. You understand what employers 
            typically look for in candidates across different positions and can provide 
            detailed insights about job requirements without needing to search external sources.""",
            tools=[],  
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def content_enhancement_writer_agent(self) -> Agent:
        """
        Agent responsible for rewriting and improving profile content
        """
        return Agent(
            role="Professional Resume and Profile Writer",
            goal="""To rewrite specific sections of a LinkedIn profile to be more impactful, 
            professional, and aligned with target job roles. The rewritten text should be 
            compelling, concise, and optimized for both human readers and ATS systems.""",
            backstory="""You are a master wordsmith and professional writer who specializes 
            in transforming dull profile descriptions into powerful career narratives. You 
            understand how to use action verbs, quantify achievements, and craft compelling 
            stories that showcase value and impact. You know how to balance keyword optimization 
            with natural, engaging language.""",
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def career_counselor_agent(self) -> Agent:
        """
        Agent responsible for career guidance and skill gap analysis
        """
        return Agent(
            role="Career Development Analyst",
            goal="""Analyze job market requirements and provide strategic career development recommendations. 
            Assess skill gaps, identify growth opportunities, and create actionable development plans.""",
            backstory="""You are a career development specialist with deep knowledge of industry requirements, 
            skill trends, and professional advancement pathways. You analyze job markets, identify skill gaps, 
            and create practical development plans. Your recommendations are data-driven, specific, and 
            focused on measurable career progression.""",
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def conversation_assistant_agent(self) -> Agent:
        """
        Agent responsible for contextual conversations using stored profile data and history
        """
        return Agent(
            role="LinkedIn Optimization Consultant",
            goal="""Provide personalized LinkedIn optimization advice based on user context and conversation history. 
            Deliver specific, actionable recommendations tailored to the user's career goals and current situation.""",
            backstory="""You are a LinkedIn optimization consultant who specializes in personalized career advice. 
            You analyze user profiles, understand their career objectives, and provide targeted recommendations. 
            Your approach is professional, direct, and focused on concrete improvements that advance their career goals.""",
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )


class TaskExecutor:
    """Handles the execution of tasks using CrewAI agents"""
    
    def __init__(self):
        self.agents = LinkedInAnalysisAgents()
        logger.info("Task Executor initialized")
    
    def process_linkedin_data(self, input_data: str) -> dict:
        """
        Execute LinkedIn data processing task
        
        Args:
            input_data (str): LinkedIn URL or profile text
            
        Returns:
            dict: Processed profile data
        """
        try:
            logger.info("Processing LinkedIn data")
            

            result = self.agents.linkedin_data_tool._run(input_data)
            
            if result.get("success") or "data" in result:
                return {
                    "success": True, 
                    "data": result,
                    "source": result.get("source", "unknown")
                }
            elif result.get("fallback") == "text_input_required":
                return {
                    "success": False,
                    "error": result.get("error"),
                    "requires_text_input": True
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error processing data")
                }
                
        except Exception as e:
            logger.error(f"Error processing LinkedIn data: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_profile(self, profile_data: dict, conversation_history: list = None) -> str:
        """
        Execute profile analysis task with conversation context
        
        Args:
            profile_data (dict): The scraped LinkedIn profile data
            conversation_history (list): Previous conversation messages for context
            
        Returns:
            str: Analysis results with improvement suggestions
        """
        agent = self.agents.profile_analyzer_agent()
        
  
        user_info = self._extract_user_info(profile_data)
        
        # Build conversation context 
        context_info = ""
        if conversation_history:
            for msg in conversation_history:
                if hasattr(msg, 'content') and msg.__class__.__name__ == "HumanMessage":
                    context_info += f"User said: {msg.content}\n"
        
       
        profile_text = ""
        if profile_data.get("raw_text"):
            profile_text = profile_data["raw_text"]
        elif profile_data.get("data", {}).get("raw_text"):
            profile_text = profile_data["data"]["raw_text"]
        elif isinstance(profile_data, dict) and "processed_sections" in profile_data:
            sections = profile_data["processed_sections"]
            if isinstance(sections, dict):
                profile_text = f"Name: {sections.get('name', 'Not provided')}\n"
                profile_text += f"Headline: {sections.get('headline', 'Not provided')}\n"
                profile_text += f"About: {sections.get('about', 'Not provided')}\n"
                profile_text += f"Experience: {sections.get('experience', 'Not provided')}\n"
                profile_text += f"Skills: {sections.get('skills', 'Not provided')}"
        
        if not profile_text or profile_text.strip() == "":
            profile_text = str(profile_data)
        
        task = Task(
            description=f"""
            HEY {user_info.get('name', 'FRIEND')}! LET'S OPTIMIZE YOUR LINKEDIN PROFILE TOGETHER:
            
            === YOUR PROFILE DETAILS ===
            Name: {user_info.get('name', 'You')}
            Current Role: {user_info.get('role', 'Your current position')}
            Company: {user_info.get('company', 'Your company')}
            
            === WHAT YOU'VE SHARED ===
            {context_info if context_info else 'You want to optimize your LinkedIn profile - great choice!'}
            
            === PROFILE CONTENT ===
            {profile_text}
            === END OF PROFILE ===
            
            Analyze this LinkedIn profile and provide specific optimization recommendations.
            
            Focus on:
            1. **Profile Completeness**: Identify missing or weak sections
            2. **Professional Headline**: Assess clarity and impact
            3. **About Section**: Evaluate storytelling and value proposition
            4. **Work Experience**: Review achievement descriptions and impact metrics
            5. **Skills & Keywords**: Analyze relevance and optimization
            6. **Overall Strategy**: Assess alignment with career goals
            
            Provide:
            ✓ Specific strengths and areas for improvement
            ✓ Actionable recommendations with examples
            ✓ Keyword suggestions for better visibility
            ✓ Strategic positioning advice
            
            Format your response with clear sections and specific examples.
            """,
            agent=agent,
            expected_output="Professional analysis with specific, actionable recommendations for LinkedIn profile optimization"
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            memory=False  
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            logger.error(f"Error analyzing profile: {str(e)}")
            return f"Error analyzing profile: {str(e)}"
    
    def analyze_job_description(self, job_role: str) -> str:
        """
        Execute job description analysis task using agent's knowledge
        
        Args:
            job_role (str): The target job role to analyze
            
        Returns:
            str: Job description analysis results
        """
        try:
            logger.info(f"Analyzing job requirements for role: {job_role}")
            
            agent = self.agents.job_description_analyzer_agent()
            
            task = Task(
                description=f"""
                Analyze the job role "{job_role}" and provide comprehensive information about:
                
                1. **Core Responsibilities**: What are the main duties and responsibilities?
                2. **Required Skills**: What technical and soft skills are typically required?
                3. **Experience Level**: What experience level is usually expected?
                4. **Education Requirements**: What educational background is preferred?
                5. **Key Technologies**: What tools, languages, or platforms are commonly used?
                6. **Industry Standards**: What are the current market expectations?
                7. **Career Progression**: How does this role fit into career development?
                
                Provide detailed, actionable information based on current industry standards
                and market expectations for the {job_role} position.
                """,
                agent=agent,
                expected_output="Comprehensive analysis of job requirements, skills, and qualifications for the specified role"
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                memory=False  
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            logger.error(f"Error analyzing job description: {str(e)}")
            return f"Error analyzing job description: {str(e)}"
    
    def generate_job_fit_report(self, profile_data: dict, job_description: str) -> str:
        """
        Execute job fit analysis task
        
        Args:
            profile_data (dict): User's LinkedIn profile data
            job_description (str): Target job requirements
            
        Returns:
            str: Job fit analysis report
        """
        agent = self.agents.career_counselor_agent()
        
        task = Task(
            description=f"""
            LET'S SEE HOW WELL YOU FIT THIS ROLE! 🎯
            
            I'm going to compare your awesome background with this target role and give you a clear picture 
            of where you stand and how to get where you want to go.
            
            YOUR PROFILE:
            {profile_data}
            
            TARGET ROLE REQUIREMENTS:
            {job_description}
            
            Here's what I'll give you:
            
            🏆 **Your Job Fit Score** (out of 100) - and why you got this score
            🔍 **Skills Gap Analysis** - what you already rock vs. what you need to develop
            📋 **Top Missing Skills** - the 3-5 most important things to focus on
            📚 **Learning Roadmap** - specific ways YOU can gain these skills
            ⏰ **Your Timeline** - a realistic plan for your career development
            🚀 **Alternative Paths** - other amazing opportunities if this role isn't the perfect fit
            
            I'm speaking directly to YOU about YOUR career journey. This is about your potential 
            and your path forward. Let's make this exciting, not intimidating!
            
            Remember: Every skill gap is just an opportunity to grow. You've got this! 💪
            """,
            agent=agent,
            expected_output="Encouraging job fit analysis that speaks directly to the user with specific, actionable guidance"
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            memory=False  
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            logger.error(f"Error generating job fit report: {str(e)}")
            return f"Error generating job fit report: {str(e)}"
    
    def enhance_profile_content(self, current_content: str, job_requirements: str, section_type: str) -> str:
        """
        Execute content enhancement task
        
        Args:
            current_content (str): Current profile section content
            job_requirements (str): Target job requirements
            section_type (str): Type of section being enhanced (e.g., "headline", "about", "experience")
            
        Returns:
            str: Enhanced content
        """
        agent = self.agents.content_enhancement_writer_agent()
        
        task = Task(
            description=f"""
            Rewrite and enhance the following LinkedIn profile section to better align with 
            the target job requirements.
            
            Section Type: {section_type}
            Current Content:
            {current_content}
            
            Target Job Requirements:
            {job_requirements}
            
            Please:
            1. Maintain the factual accuracy of the original content
            2. Enhance language to be more impactful and professional
            3. Incorporate relevant keywords from the job requirements
            4. Quantify achievements where possible
            5. Use strong action verbs and compelling language
            6. Ensure the content is optimized for both ATS and human readers
            
            Provide the rewritten content along with a brief explanation of the changes made.
            """,
            agent=agent,
            expected_output="Enhanced profile section content with explanation of improvements"
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            memory=False  
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            logger.error(f"Error enhancing content: {str(e)}")
            return f"Error enhancing content: {str(e)}"
    
    def contextual_chat_response(self, user_query: str, profile_data: dict, conversation_history: list, analysis_results: str = None) -> str:
        """
        Execute contextual chat response using conversation history and profile data
        
        Args:
            user_query (str): Current user query
            profile_data (dict): User's LinkedIn profile data
            conversation_history (list): Previous messages in the conversation
            analysis_results (str): Previous analysis results if available
            
        Returns:
            str: Contextual response using stored user information
        """
        agent = self.agents.conversation_assistant_agent()
        
        
        user_info = self._extract_user_info(profile_data)
        
        # Build conversation context 
        chat_history = self._build_efficient_conversation_context(conversation_history)
        
        # Build comprehensive context
        context = f"""
=== USER PROFILE INFORMATION ===
Name: {user_info.get('name', 'Not specified')}
Role: {user_info.get('role', 'Not specified')}
Company: {user_info.get('company', 'Not specified')}
Profile Summary: {user_info.get('summary', 'Not available')}

=== CONVERSATION HISTORY ===
{chat_history}

=== CURRENT USER QUERY ===
{user_query}

=== PREVIOUS ANALYSIS RESULTS ===
{analysis_results if analysis_results else 'No previous analysis available'}
"""

        task = Task(
            description=f"""
            HI {user_info.get('name', 'FRIEND').upper()}! 👋
            
            I'm your personal LinkedIn coach and I remember everything about you and our conversations!
            
            === WHAT I KNOW ABOUT YOU ===
            Your Name: {user_info.get('name', 'You')}
            Your Role: {user_info.get('role', 'Your current position')}
            Your Company: {user_info.get('company', 'Your workplace')}
            Your Profile: {user_info.get('summary', 'Your amazing background')}

            === OUR CONVERSATION SO FAR ===
            {chat_history}

            === WHAT YOU JUST ASKED ===
            {user_query}

            === WHAT WE'VE WORKED ON TOGETHER ===
            {analysis_results if analysis_results else 'We\'re just getting started!'}
            
            Provide a contextual response based on the user's situation and conversation history.
            
            RESPONSE REQUIREMENTS:
            • Address the user directly using available context (name, role, company)
            • Reference relevant conversation history
            • Provide specific, actionable LinkedIn optimization advice
            • Maintain professional tone while being personable
            • Focus on concrete next steps and recommendations
            
            Tailor your response to their specific career situation and goals.
            """,
            agent=agent,
            expected_output="Professional, contextual response with specific LinkedIn optimization recommendations"
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            memory=False  
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            logger.error(f"Error generating contextual response: {str(e)}")
            return f"Error generating contextual response: {str(e)}"
    
    def _extract_user_info(self, profile_data: dict) -> dict:
        """
        Extract key user information from profile data
        
        Args:
            profile_data (dict): Raw profile data
            
        Returns:
            dict: Extracted user information
        """
        user_info = {
            'name': 'Not specified',
            'role': 'Not specified', 
            'company': 'Not specified',
            'summary': 'Not available'
        }
        
        try:
            # Try different ways to extract user information
            if isinstance(profile_data, dict):
                
                if 'data' in profile_data and isinstance(profile_data['data'], dict):
                    data = profile_data['data']
                else:
                    data = profile_data
                
                
                if 'processed_sections' in data:
                    sections = data['processed_sections']
                    if isinstance(sections, dict):
                        user_info['name'] = sections.get('name', user_info['name'])
                        user_info['role'] = sections.get('headline', sections.get('role', user_info['role']))
                        user_info['summary'] = sections.get('about', user_info['summary'])
                
                # Look for raw text and extract information
                raw_text = data.get('raw_text', '')
                if raw_text and isinstance(raw_text, str):
                    # Extract names using pattern matching
                    import re
                    
                    # Look for "I'm [Name]" patterns
                    name_patterns = [
                        r"I'm\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                        r"My name is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", 
                        r"Hello.*I'm\s+([A-Z][a-z]+\s+[A-Z][a-z]+)"
                    ]
                    
                    for pattern in name_patterns:
                        match = re.search(pattern, raw_text)
                        if match:
                            user_info['name'] = match.group(1)
                            break
                    
                    
                    role_patterns = [
                        r"(Marketing Manager)",
                        r"(Product Manager)", 
                        r"(Data Scientist)",
                        r"(Software Engineer)",
                        r"(Project Manager)",
                        r"([A-Z][a-z]+\s+Manager)",
                        r"([A-Z][a-z]+\s+Engineer)",
                        r"([A-Z][a-z]+\s+Analyst)",
                        r"([A-Z][a-z]+\s+Developer)"
                    ]
                    
                    for pattern in role_patterns:
                        match = re.search(pattern, raw_text)
                        if match:
                            user_info['role'] = match.group(1)
                            break
                    
                    
                    company_patterns = [
                        r"at\s+([A-Z][A-Za-z]+)",
                        r"@\s*([A-Z][A-Za-z]+)",
                        r"working for\s+([A-Z][A-Za-z]+)"
                    ]
                    
                    for pattern in company_patterns:
                        match = re.search(pattern, raw_text)
                        if match:
                            user_info['company'] = match.group(1)
                            break
                
                # Also check direct keys
                user_info['name'] = data.get('name', user_info['name'])
                user_info['role'] = data.get('role', data.get('headline', user_info['role']))
                user_info['company'] = data.get('company', user_info['company'])
                user_info['summary'] = data.get('about', data.get('summary', user_info['summary']))
                
                
                logger.info(f"Extracted user info: {user_info}")
        
        except Exception as e:
            logger.error(f"Error extracting user info: {str(e)}")
        
        return user_info
    
    def _build_efficient_conversation_context(self, conversation_history: list, max_recent_messages: int = 8, max_tokens: int = 2000) -> str:
        """
        Build conversation context with sliding window and summarization
        
        Args:
            conversation_history (list): Full conversation history
            max_recent_messages (int): Number of recent messages to keep in full
            max_tokens (int): Maximum tokens for context
            
        Returns:
            str: Optimized conversation context
        """
        if not conversation_history:
            return "No previous conversation"
        
        
        if len(conversation_history) <= max_recent_messages:
            return self._format_messages(conversation_history)
        
        
        older_messages = conversation_history[:-max_recent_messages]
        recent_messages = conversation_history[-max_recent_messages:]
        
        
        older_summary = self._summarize_conversation_chunk(older_messages)
        
       
        recent_formatted = self._format_messages(recent_messages)
        
    
        combined_context = f"""
=== CONVERSATION SUMMARY (Earlier Messages) ===
{older_summary}

=== RECENT CONVERSATION (Last {max_recent_messages} Messages) ===
{recent_formatted}
"""
        
        # Check token estimate and truncate if needed
        estimated_tokens = len(combined_context.split()) * 1.3
        if estimated_tokens > max_tokens:
            # Reduce recent messages if too long
            reduced_recent = recent_messages[-max_recent_messages//2:]
            recent_formatted = self._format_messages(reduced_recent)
            combined_context = f"""
=== CONVERSATION SUMMARY (Earlier Messages) ===
{older_summary}

=== RECENT CONVERSATION (Last {len(reduced_recent)} Messages) ===
{recent_formatted}
"""
        
        return combined_context
    
    def _format_messages(self, messages: list) -> str:
        """
        Format messages for context
        
        Args:
            messages (list): List of messages to format
            
        Returns:
            str: Formatted message string
        """
        formatted = ""
        for msg in messages:
            if hasattr(msg, 'content'):
                role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                content = str(msg.content)[:500]  # Limit individual message length
                if len(str(msg.content)) > 500:
                    content += "... [truncated]"
                formatted += f"\n{role}: {content}\n"
        return formatted
    
    def _summarize_conversation_chunk(self, messages: list) -> str:
        """
        Create a summary of conversation chunk
        
        Args:
            messages (list): Messages to summarize
            
        Returns:
            str: Summary of the conversation chunk
        """
        if not messages:
            return "No previous conversation"
        
        # Extract key information from messages
        user_topics = []
        assistant_responses = []
        
        for msg in messages:
            if hasattr(msg, 'content'):
                content = str(msg.content).lower()
                if msg.__class__.__name__ == "HumanMessage":
                    # Extract user topics/requests
                    if "profile" in content:
                        user_topics.append("profile analysis")
                    if "job" in content or "role" in content:
                        user_topics.append("job fit analysis")
                    if "skill" in content:
                        user_topics.append("skills discussion")
                    if "rewrite" in content or "enhance" in content:
                        user_topics.append("content enhancement")
                    if "network" in content:
                        user_topics.append("networking advice")
                else:
                    # Track assistant capabilities shown
                    if "analysis" in content:
                        assistant_responses.append("provided analysis")
                    if "recommendation" in content:
                        assistant_responses.append("gave recommendations")
        
        # summary
        topics_discussed = list(set(user_topics)) if user_topics else ["general LinkedIn questions"]
        responses_given = list(set(assistant_responses)) if assistant_responses else ["provided assistance"]
        
        summary = f"""Earlier in our conversation ({len(messages)} messages):
- User topics: {', '.join(topics_discussed)}
- Assistant responses: {', '.join(responses_given)}
- Key context: User is seeking LinkedIn profile optimization and career advice"""
        
        return summary


# For testing purposes
if __name__ == "__main__":
    executor = TaskExecutor()
    print("LinkedIn Analysis Agents initialized successfully!")
