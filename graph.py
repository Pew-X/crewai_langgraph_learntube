
import json
import sqlite3
from typing import TypedDict, List, Literal, Optional, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from crewai import Task, Crew  # Import CrewAI for direct node execution
from agents import TaskExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State schema for LinkedIn optimization graph with native LangGraph memory"""
    
    linkedin_url: Optional[str]
    job_role: Optional[str]
    user_query: Optional[str]
    
    
    profile_data: Optional[dict]
    profile_scraped: bool
    
    
    analysis_results: Optional[str]
    job_description: Optional[str]
    job_fit_report: Optional[str]
    rewritten_section: Optional[str]
    
    #  LangGraph memory 
    messages: Annotated[List[BaseMessage], add_messages]
    
    # User profile for personalization
    user_profile: Dict[str, Any]
    
    # Control flow
    next_action: Optional[str]





class LinkedInOptimizationGraph:
    """LangGraph implementation for LinkedIn profile optimization workflow with native memory"""
    
    def __init__(self):
        """Initialize the graph with task executor and SQLite memory"""
        self.task_executor = TaskExecutor()
        
        # Initialize SQLite memory 
        try:
            
            self.conn = sqlite3.connect("linkedin_optimizer_memory.db", check_same_thread=False)
            self.memory = SqliteSaver(self.conn)
            self.memory.setup()  
            logger.info("SQLite memory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite memory: {str(e)}")
    
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
            self.memory = SqliteSaver(self.conn)
            self.memory.setup()
            logger.info("Using in-memory SQLite as fallback")
        
        
        self.graph = self._build_graph()
        logger.info("LinkedIn Optimization Graph initialized with persistent memory")
    
    def _extract_user_info(self, message_content: str, current_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced user info extraction - reading directly from state"""
        content = message_content.lower()
        profile = current_profile.copy()
        
        
        name_patterns = [
            r"i'm ([a-zA-Z\s]+)(?:,|\.|!|\s)",
            r"my name is ([a-zA-Z\s]+)(?:,|\.|!|\s)",
            r"i am ([a-zA-Z\s]+)(?:,|\.|!|\s)",
            r"this is ([a-zA-Z\s]+)(?:,|\.|!|\s)",
            r"hi.*i'm ([a-zA-Z\s]+)(?:,|\.|!|\s)"
        ]
        
        import re
        for pattern in name_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip().title()
               
                if len(potential_name) >= 2 and all(c.isalpha() or c.isspace() for c in potential_name):
                    profile["name"] = potential_name
                    break
        
        
        role_patterns = {
            "data scientist": ["data scientist", "data science"],
            "software engineer": ["software engineer", "developer", "programmer"],
            "product manager": ["product manager", "pm"],
            "analyst": ["analyst", "business analyst"],
            "machine learning engineer": ["ml engineer", "machine learning engineer", "ai engineer"]
        }
        
        for role_key, role_variations in role_patterns.items():
            for variation in role_variations:
                if variation in content:
                    profile["current_role"] = role_key.title()
                    break
        
    
        company_patterns = ["netflix", "google", "microsoft", "apple", "amazon", "meta", "facebook", "tesla", "uber"]
        
        for company in company_patterns:
            if f"at {company}" in content or f"work for {company}" in content or f"working at {company}" in content:
                profile["company"] = company.title()
                break
        
        return profile
    
    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow"""
        
        
        workflow = StateGraph(GraphState)
        
        
        workflow.add_node("router", self.route_user_input_node)
        workflow.add_node("process_data", self.process_data_node)
        workflow.add_node("analyze_profile", self.analyze_profile_node)
        workflow.add_node("analyze_job", self.analyze_job_node)
        workflow.add_node("generate_job_fit", self.generate_job_fit_node)
        workflow.add_node("enhance_content", self.enhance_content_node)
        workflow.add_node("chat_response", self.chat_response_node)
        
        # entry point
        workflow.set_entry_point("router")
        
        # conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "process_data": "process_data",
                "analyze_profile": "analyze_profile", 
                "analyze_job": "analyze_job",
                "generate_job_fit": "generate_job_fit",
                "enhance_content": "enhance_content",
                "chat_response": "chat_response",
                "end": END
            }
        )
        
    
        workflow.add_edge("process_data", "analyze_profile")
        workflow.add_edge("analyze_profile", END)
        workflow.add_edge("analyze_job", "generate_job_fit")
        workflow.add_edge("generate_job_fit", END)
        workflow.add_edge("enhance_content", END)
        workflow.add_edge("chat_response", END)
        
        # Compile with SQLite checkpointer 
        compiled_graph = workflow.compile(checkpointer=self.memory)
        
        return compiled_graph
    
    def route_user_input_node(self, state: GraphState) -> GraphState:
        """
        Router node that determines next action
        
        Args:
            state (GraphState): Current graph state
            
        Returns:
            GraphState: Updated state with next action
        """
        next_action = self.route_decision(state)
        return {**state, "next_action": next_action}
    
    def route_decision(self, state: GraphState) -> Literal["process_data", "analyze_profile", "analyze_job", "generate_job_fit", "enhance_content", "chat_response", "end"]:
        """
        Enhanced router - relies on LangGraph state management with better user info detection
        
        Args:
            state (GraphState): Current graph state
            
        Returns:
            str: Name of the next node to execute
        """
        logger.info("Routing user input...")
        
        # Get conversation history 
        conversation_history = state.get("messages", [])
        
        
        user_profile_from_conversation = {}
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                user_profile_from_conversation = self._extract_user_info(msg.content, user_profile_from_conversation)
        
        
        if user_profile_from_conversation.get("name") or user_profile_from_conversation.get("current_role"):
            logger.info(f"Found user profile info: {user_profile_from_conversation}, routing to chat response")
            return "chat_response"
        
       
        has_profile_data = state.get("profile_scraped", False) or state.get("profile_data") is not None
        
        
        if not has_profile_data and conversation_history:
            last_message = conversation_history[-1]
            if isinstance(last_message, HumanMessage):
                content = last_message.content.lower()
                if any(keyword in content for keyword in ["linkedin.com", "http", "www.", "url"]):
                    logger.info("Found LinkedIn URL, routing to data processing")
                    return "process_data"
        
        
        if has_profile_data and conversation_history:
            last_message = conversation_history[-1]
            if isinstance(last_message, HumanMessage):
                message_content = last_message.content.lower()
                
              
                if any(keyword in message_content for keyword in ["analyze", "review", "feedback", "suggestions"]):
                    if not state.get("analysis_results"):
                        logger.info("Routing to profile analysis")
                        return "analyze_profile"
                
                
                elif any(keyword in message_content for keyword in ["job", "role", "position", "career", "fit", "match"]):
                    if not state.get("job_description"):
                        logger.info("Routing to job analysis")
                        return "analyze_job"
                    else:
                        logger.info("Routing to job fit generation")
                        return "generate_job_fit"
                
               
                elif any(keyword in message_content for keyword in ["rewrite", "improve", "enhance", "optimize", "update", "better"]):
                    logger.info("Routing to content enhancement")
                    return "enhance_content"
        
        
        logger.info("Routing to chat response - conversational")
        return "chat_response"
    
    def process_data_node(self, state: GraphState) -> GraphState:
        """
        Node for processing LinkedIn profile data (URL or text)
        
        Args:
            state (GraphState): Current graph state
            
        Returns:
            GraphState: Updated state with profile data
        """
        logger.info("Executing data processing node")
        
        
        input_data = state.get("linkedin_url") or state.get("user_query", "")
        
        if not input_data:
            return {
                **state,
                "messages": [
                    AIMessage(content="I need either a LinkedIn profile URL or your LinkedIn profile text to get started. Please provide one of these.")
                ]
            }
        
        
        result = self.task_executor.process_linkedin_data(input_data)
        
        if result.get("success"):
            profile_data = result.get("data", {})
            response_message = " Successfully processed your LinkedIn profile! I can now help you with:\n\n" \
                             "• **Profile Analysis** - Get detailed feedback on your profile\n" \
                             "• **Job Fit Analysis** - Compare your profile against specific job roles\n" \
                             "• **Content Enhancement** - Improve specific sections of your profile\n\n" \
                             "What would you like me to help you with?"
            
            return {
                **state,
                "profile_data": profile_data,
                "profile_scraped": True,
                "messages": [AIMessage(content=response_message)]
            }
        elif result.get("requires_text_input"):
            error_message = f" {result.get('error', 'Unable to process URL')}\n\n" \
                          "**Alternative**: Please copy and paste your LinkedIn profile text directly, and I'll analyze it for you.\n\n" \
                          " **Tip**: You can copy your profile text from LinkedIn and paste it here for analysis."
            
            return {
                **state,
                "messages": [AIMessage(content=error_message)]
            }
        else:
            error_message = f" Failed to process the LinkedIn profile: {result.get('error', 'Unknown error')}\n\n" \
                          "Please try:\n" \
                          "• Checking that the URL is a valid LinkedIn profile URL\n" \
                          "• Copying and pasting your LinkedIn profile text directly\n" \
                          "• Making sure your profile is public if using a URL"
            
            return {
                **state,
                "messages": [AIMessage(content=error_message)]
            }
    
    def analyze_profile_node(self, state: GraphState) -> GraphState:
        """
        Simplified profile analysis node - reads directly from state
        
        Args:
            state (GraphState): Current graph state with profile_data
            
        Returns:
            GraphState: Updated state with analysis results
        """
        logger.info("Executing profile analysis node")
        
        # 1. READ the state dictionary
        profile_data = state.get("profile_data")
        conversation_history = state.get("messages", [])  #  history here!
        user_profile = state.get("user_profile", {})

        if not profile_data:
            return {
                "messages": [AIMessage(content="I need your profile data first. Please provide your LinkedIn URL.")]
            }

        # 2. HIRE THE AGENT and build the task prompt 
        agent = self.task_executor.agents.profile_analyzer_agent()
        
        
        task_description = f"""
Profile Analysis for LinkedIn Optimization:

=== USER PROFILE DETAILS ===
Name: {user_profile.get('name', 'Not specified')}
Current Role: {user_profile.get('current_role', 'Not specified')}
Company: {user_profile.get('company', 'Not specified')}

=== CONVERSATION CONTEXT ===
"""
        # Add complete conversation context
        for msg in conversation_history:  # Use FULL history, not truncated
            if hasattr(msg, 'content'):
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                task_description += f"{role} said: {msg.content}\n"
        
        task_description += f"""

=== PROFILE CONTENT ===
{json.dumps(profile_data, indent=2)}
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
"""

        task = Task(description=task_description, agent=agent, expected_output="Professional analysis with specific, actionable recommendations for LinkedIn profile optimization")
        crew = Crew(agents=[agent], tasks=[task], memory=False)
        analysis_results = crew.kickoff()

        # 3. WRITE THE RESULT back 
        return {
            "analysis_results": analysis_results,
            "messages": [AIMessage(content=f"📊 **LinkedIn Profile Analysis Complete**\n\n{analysis_results}")]
        }
    
    def analyze_job_node(self, state: GraphState) -> GraphState:
        """
        Node for analyzing job descriptions
        
        Args:
            state (GraphState): Current graph state
            
        Returns:
            GraphState: Updated state with job analysis
        """
        logger.info("Executing job analysis node")
        
       
        job_role = state.get("job_role")
        if not job_role and state.get("messages"):
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                #  look for job titles in the message
                message_content = last_message.content
                job_role = message_content  # use the entire message as job role for now
        
        if not job_role:
            return {
                **state,
                "messages": state.get("messages", []) + [
                    AIMessage(content="Please specify the job role you're interested in for analysis.")
                ]
            }
        
       
        job_description = self.task_executor.analyze_job_description(job_role)
        
        return {
            **state,
            "job_role": job_role,
            "job_description": job_description
        }
    
    def generate_job_fit_node(self, state: GraphState) -> GraphState:
        """
        Node for generating job fit report
        
        Args:
            state (GraphState): Current graph state
            
        Returns:
            GraphState: Updated state with job fit report
        """
        logger.info("Executing job fit analysis node")
        
        profile_data = state.get("profile_data")
        job_description = state.get("job_description")
        
        if not profile_data or not job_description:
            return {
                **state,
                "messages": state.get("messages", []) + [
                    AIMessage(content="I need both profile data and job description to generate a fit report.")
                ]
            }
        
       
        job_fit_report = self.task_executor.generate_job_fit_report(profile_data, job_description)
        
        response_message = f"🎯 **Job Fit Analysis Report**\n\n{job_fit_report}\n\n" \
                          "Would you like me to help you improve any specific areas or rewrite sections of your profile?"
        
        return {
            **state,
            "job_fit_report": job_fit_report,
            "messages": [AIMessage(content=response_message)]
        }
    
    def enhance_content_node(self, state: GraphState) -> GraphState:
        """
        Node for enhancing profile content
        
        Args:
            state (GraphState): Current graph state
            
        Returns:
            GraphState: Updated state with enhanced content
        """
        logger.info("Executing content enhancement node")
        
        profile_data = state.get("profile_data")
        job_description = state.get("job_description", "")
        
        if not profile_data:
            return {
                **state,
                "messages": state.get("messages", []) + [
                    AIMessage(content="I need your profile data to enhance content. Please provide a LinkedIn URL first.")
                ]
            }
        
       
        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        section_type = "about"  
        current_content = str(profile_data.get("about", ""))
        
        if last_message and isinstance(last_message, HumanMessage):
            message_content = last_message.content.lower()
            if "headline" in message_content:
                section_type = "headline"
                current_content = str(profile_data.get("headline", ""))
            elif "experience" in message_content:
                section_type = "experience"
                current_content = str(profile_data.get("experience", ""))
            elif "summary" in message_content or "about" in message_content:
                section_type = "about"
                current_content = str(profile_data.get("about", ""))
        
        
        enhanced_content = self.task_executor.enhance_profile_content(
            current_content, job_description, section_type
        )
        
        response_message = f"✨ **Enhanced {section_type.title()} Section**\n\n{enhanced_content}\n\n" \
                          "Would you like me to enhance any other sections or provide additional suggestions?"
        
        return {
            **state,
            "rewritten_section": enhanced_content,
            "messages": [AIMessage(content=response_message)]
        }
    
    def chat_response_node(self, state: GraphState) -> GraphState:
        """
        Enhanced chat response node - implements proper state injection pattern
        
        Args:
            state (GraphState): Current graph state
            
        Returns:
            GraphState: Updated state with contextual chat response
        """
        logger.info("Executing enhanced chat response node with memory")
        

        conversation_history = state.get("messages", [])  # Full history here!
        user_profile = state.get("user_profile", {})
        profile_data = state.get("profile_data", {})
        
       
        if conversation_history:
            last_message = conversation_history[-1]
            if isinstance(last_message, HumanMessage):
               
                user_profile = self._extract_user_info(last_message.content, user_profile)
        
      
        if not state.get("profile_scraped"):
            # Build contextual task description 
            agent = self.task_executor.agents.conversation_assistant_agent()
            
            #  STATE INJECTION 
            task_description = f"""
LinkedIn optimization consultation based on user context and conversation history.

=== USER PROFILE INFORMATION ===
User's Name: {user_profile.get('name', 'Not provided')}
User's Current Role: {user_profile.get('current_role', 'Not provided')}
User's Company: {user_profile.get('company', 'Not provided')}
Profile Status: {'Available' if profile_data else 'Not provided'}

=== CONVERSATION HISTORY ===
"""
            # Inject ALL conversation history into the task description - NO TRUNCATION
            for i, msg in enumerate(conversation_history, 1):
                if hasattr(msg, 'content'):
                    role = "User" if isinstance(msg, HumanMessage) else "Assistant" 
                    task_description += f"Turn {i} - {role}: {msg.content}\n"
            
            task_description += f"""

=== CURRENT USER QUERY ===
{conversation_history[-1].content if conversation_history else "Initial consultation"}

=== CONSULTATION REQUIREMENTS ===
Provide a professional LinkedIn optimization response that:

1. Uses available user context (name, role, company) appropriately
2. References relevant conversation history when applicable  
3. Delivers specific, actionable LinkedIn optimization advice
4. Maintains professional tone while being personable
5. Focuses on concrete recommendations and next steps

Provide specific guidance tailored to their career situation and LinkedIn optimization needs.
"""

           
            from crewai import Task, Crew
            task = Task(
                description=task_description, 
                agent=agent, 
                expected_output="Professional, contextual LinkedIn optimization advice with specific recommendations"
            )
            crew = Crew(agents=[agent], tasks=[task], memory=False)  
            response = str(crew.kickoff())
        else:
           
            response = f"""Great to continue our conversation{', ' + user_profile.get('name', '') if user_profile.get('name') else ''}!

I can help you with LinkedIn optimization including:
• Profile analysis and feedback
• Job fit analysis for specific roles  
• Content enhancement for profile sections
• Career strategy and positioning

What specific aspect would you like to focus on today?"""
        
       
        return {
            "user_profile": user_profile,  # Update user profile in state
            "messages": [AIMessage(content=response)]  # add_messages will handle appending
        }
    
    def get_runnable(self):
        """Get the compiled graph for execution"""
        return self.graph
    
    def close(self):
        """Close the SQLite connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("SQLite connection closed")



def create_linkedin_optimization_graph():
    """Create and return the LinkedIn optimization graph"""
    return LinkedInOptimizationGraph()


# For testing
if __name__ == "__main__":
    graph = create_linkedin_optimization_graph()
    print("LinkedIn Optimization Graph created successfully!")
