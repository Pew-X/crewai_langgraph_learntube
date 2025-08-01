

import json
import sqlite3
import re
from typing import TypedDict, List, Optional, Annotated , Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from crewai import Task, Crew
from agents import LinkedInAnalysisAgents
from langchain_core.prompts import ChatPromptTemplate ,PromptTemplate
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser
from tools import get_linkedin_data_tool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    profile_data: Optional[dict]
    profile_scraped: bool = False
    next_action: Optional[str]




class LinkedInOptimizationGraph:
    def __init__(self):
        self.agents = LinkedInAnalysisAgents()
        self.llm = self.agents.llm # Share the same LLM instance
        self.conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        self.graph = self._build_graph()
        logger.info("LinkedIn Optimization Graph initialized correctly.")

    def _create_contextual_prompt_data(self, state: GraphState) -> dict:
        """
        The Memory & Context Engine V2.
        This function processes the raw message history into a clean, structured, and
        artistically formatted string for injection into agent prompts.
        """
        logger.info("--- HELPER: V2 Creating Contextual Prompt Data ---")
        
        #  the latest user message
        latest_user_message = ""
        if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
            latest_user_message = state["messages"][-1].content

        #  TEMPORALLY ENUMERATED recent turn-by-turn history
        recent_turns_text = ""
        last_few_messages = state["messages"][-6:] # last 3 pairs of user/assistant
        if last_few_messages:
            formatted_turns = []
            # We enumerate backwards from T-0 (the latest message)
            for i, msg in enumerate(reversed(last_few_messages)):
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                content = msg.content.replace('\n', ' ').strip()
                content = content[:300] + "..." if len(content) > 300 else content
                formatted_turns.append(f"[T-{i}] {role}: \"{content}\"")
            recent_turns_text = "\n".join(reversed(formatted_turns)) # Reverse back to chronological
            
        
        conversation_summary = "This is the beginning of our conversation."
        if len(state["messages"]) > 2:
            
            full_history_script = ""
            for msg in state['messages']:
                role = "User" if isinstance(msg, HumanMessage) else "AI Career Coach"
                full_history_script += f"{role}: {msg.content}\n---\n"
                
            summarizer_agent = self.agents.conversation_summarizer_agent()
            summary_task = Task(
                description=f"""
                Analyze the following conversation script between a User and an AI Career Coach.
                Your task is to produce a detailed, possibly multi-paragraph narrative summary from the AIs perspective.
                This summary is CRITICAL for other AI agents to understand the conversational context so be thorough.
                Capture the key information:
                - **User Identity:** Who is the user? (e.g., "The user is John Doe, a Software Engineer at Google.")
                - **Core Goal:** What is their primary objective in this conversation? (e.g., "His core goal is to transition from his current role into a Director-level Product Management position.")
                - **Accomplished Steps:** What has been discussed or achieved so far? (e.g., "We have already performed an initial profile analysis and identified a key weakness in his project descriptions.")
                - **Current Focus:** What is the immediate topic of conversation, based on the last few exchanges? (e.g., "He is now asking for a detailed career roadmap.")

                CONVERSATION SCRIPT:
                {full_history_script}
                """,
                agent=summarizer_agent,
                expected_output="A detailed, possibly multi-paragraph narrative summary covering conversational context, the users identity, core goal, accomplished steps, and current focus."
            )
            crew = Crew(agents=[summarizer_agent], tasks=[summary_task]) 
            conversation_summary = str(crew.kickoff()).strip()

        
        return {
            "latest_user_question": latest_user_message,
            "conversation_summary": conversation_summary,
            "recent_conversation_turns": recent_turns_text,
            # Now we pass the intelligently parsed data
            "profile_data_json": json.dumps(state.get("profile_data"), indent=2) if state.get("profile_data") else "Not yet provided by the user."
        }

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        workflow.add_node("router", self.intelligent_router_node)
        workflow.add_node("process_data", self.process_data_node)
        workflow.add_node("analyze_profile", self.analyze_profile_node)
        workflow.add_node("analyze_job_fit", self.analyze_job_fit_node)
        workflow.add_node("provide_career_path", self.provide_career_path_node)
        workflow.add_node("general_chat", self.general_chat_node)
        
        workflow.set_entry_point("router")
        workflow.add_conditional_edges("router", lambda x: x["next_action"], {
            "process_data": "process_data", "analyze_profile": "analyze_profile",
            "analyze_job_fit": "analyze_job_fit", "provide_career_path": "provide_career_path",
            "general_chat": "general_chat", "end": END
        })
        
        # workflow.add_edge("process_data", "analyze_profile")

        workflow.add_edge("process_data", END)
        workflow.add_edge("analyze_profile", END)
        workflow.add_edge("analyze_job_fit", END)
        workflow.add_edge("provide_career_path", END)
        workflow.add_edge("general_chat", END)
        
        return workflow.compile(checkpointer=self.memory)




    def intelligent_router_node(self, state: GraphState) -> dict:
        logger.info("--- NODE: Intelligent Router ---")
        
        # Get the latest message to analyze for this turn
        latest_message = state["messages"][-1]
        
        #  If a profile hasn't been scraped yet AND the user just provided one,
        # force the process_data node. This is the most reliable way.
        if not state.get("profile_scraped"):
            # latest_content = latest_message.content.lower()
            
            # if isinstance(latest_message, HumanMessage) and ("linkedin.com/in/" in latest_content):
            #     logger.info("--- ROUTING (Rule-Based Profile Detection): 'PROCESS_DATA' ---")
                return {"next_action": "process_data"}

        # all other cases, use the intelligent AI-based router.
       
        
        # 1. Get the clean, structured context from our Memory Engine helper.
        prompt_data = self._create_contextual_prompt_data(state)
        
        agent = self.agents.intelligent_router_agent()

    
        task_description = f"""
        You are an expert AI router for a LinkedIn career coach chatbot.
        Your task is to analyze the conversation context and the users latest message to decide which specialist tool to use next.

        === CONVERSATION SUMMARY ===
        {prompt_data['conversation_summary']}
        
        === RECENT CONVERSATION TURNS (The users last message is at the end) ===
        {prompt_data['recent_conversation_turns']}

        === ROUTING GUIDE ===
        Based on the users very latest message and the conversation context and summary, choose the most appropriate tool for the next action.

        Here is the users latest question:
        "{prompt_data['latest_user_question']}"

        Now based on the above context and current question, choose ONE of the following tools appropriately:

        - "analyze_profile": Use this for a general profile review, feedback, or questions like "how can I improve my profile?".
        - "analyze_job_fit": Use this if the user provides a job description OR asks to be compared against a specific job title/role. Their SUITABILITY, FIT, or if their BACKGROUND IS ENOUGH for a specific job title. Use this for direct job comparison questions. (e.g., "am I a good fit for a Senior Engineer?").
        - "provide_career_path": Use this if the user asks for a career roadmap, guidance on reaching a new role (e.g., "how do I become a CTO?"), or skill gap analysis.
        - "general_chat": Use this for anything else: greetings, simple follow-up questions, thank yous, or anything that doesn't cleanly fit the above categories. Basically general chatbot like chat.

        Your final answer MUST be ONLY ONE of these exact tool names:
        
        analyze_profile
        analyze_job_fit
        provide_career_path
        general_chat
        """
        
        task = Task(
            description=task_description,
            agent=agent,
            expected_output="A single string containing only ONE of the allowed tool names."
        )
        
        # a temporary crew to get the routing decision.
        crew = Crew(agents=[agent], tasks=[task], verbose=1)
        decision = str(crew.kickoff()).strip().lower().replace('"', '').replace("'", "")

        # fallback for safety
        allowed_nodes = ["process_data", "analyze_profile", "analyze_job_fit", "provide_career_path", "general_chat"]
        if decision not in allowed_nodes:
            logger.warning(f"Router returned an invalid node '{decision}'. Defaulting to 'general_chat'.")
            decision = "general_chat"

        logger.info(f"--- ROUTING (AI-Based Crew): '{decision.upper()}' ---")
        return {"next_action": decision}




    def process_data_node(self, state: GraphState) -> dict:
        logger.info("--- NODE: Processing Profile Data ---")
        user_input = state["messages"][-1].content
        tool = get_linkedin_data_tool()
        result = tool.run(user_input)
        # if "error" in result:
        #     return {"messages": [AIMessage(content=f"Error processing data: {result['error']}")]}
        # return {"profile_data": result.get("data"), "profile_scraped": True, "messages": [AIMessage(content="Profile data loaded successfully! Now, what would you like to do? For example: 'Analyze my profile' or 'How do I fit for a Senior Engineer role?'")]}
        if result.get("success"):
        # If and ONLY if the tool explicitly signals success...
            logger.info("Tool successfully processed a profile.")
            # ...do we update the state and return the success message.
            return {
                "profile_data": result.get("data"),
                "profile_scraped": True,
                "messages": [AIMessage(content="Profile data loaded successfully! Now, what would you like to do? For example: 'Analyze my profile' or 'How do I fit for a Senior Engineer role?'")]
            }
        else:
            prompt_data = self._create_contextual_prompt_data(state)
        agent = self.agents.general_chat_agent()

        
        task_description = f"""
        **OPERATIONAL BRIEFING: Conversational Interface Management**

        You are an the Empathetic AI Career Coach. Your job is to have a natural, helpful conversation.

        **CONTEXT:** The system just attempted to process the users latest message as a LinkedIn profile, but it failed The users profile has NOT been loaded.

        **USERs LATEST MESSAGE:**
        "{prompt_data['latest_user_question']}"

        **MISSION:**
        Your goal is to respond conversationally to the users latest message. Acknowledge what they said, but since the profile processing failed, you must gently guide them to provide a valid profile if thats whats needed to answer their question. Use your standard response protocol. For example, if they just said "hi", you should just say "hi" back and ask how you can help.
        """
        
        task = Task(description=task_description, agent=agent, expected_output="A natural, helpful, and context-aware conversational response.")
        crew = Crew(agents=[agent], tasks=[task])
        
        
        # do NOT update profile_scraped, which is correct.
        response_content = str(crew.kickoff())
        return {"messages": [AIMessage(content=response_content)]}

    

    def analyze_profile_node(self, state: GraphState) -> dict:
        logger.info("--- NODE: Analyzing Profile ---")
        
        # Ensure profile data exists.
        if not state.get("profile_data"): 
            return {"messages": [AIMessage(content="To analyze your profile, I first need it! Could you please provide your LinkedIn URL or paste the profile text?")]}

        
        prompt_data = self._create_contextual_prompt_data(state)
        
        agent = self.agents.profile_analyzer_agent()
        
       
        task_description = f"""

        (YOU ARE PART OF A MULTI AGENTIC WORKFLOW CULMINATING AS A USER FACING CONVERSATIONAL CHATBOT WHERE YOU ARE ONE OF THE AGENTS)

        **OPERATIONAL BRIEFING: Profile Audit & Strategic Review**
        You are a Senior LinkedIn Strategist & Brand Consultant. Your task is to perform a detailed and actionable analysis of the users LinkedIn profile.


        === CURRENT USER QUERY ===

         THE USERs LATEST QUERY: 
        
        "{prompt_data['latest_user_question']}".

        === CURRENT USER QUERY ===

        === CONTEXT OF OUR CONVERSATION ===
        {prompt_data['conversation_summary']}
        
        === RECENT CONVERSATION TURNS ===
        {prompt_data['recent_conversation_turns']}

        === USERs FULL PROFILE DATA ===
        {prompt_data['profile_data_json']}

        
        
        **MISSION:**

        Your mission is to conduct a professional-grade audit of their LinkedIn profile. Your output MUST be a structured report, written in a direct, authoritative, and consultative tone. **You must address the user directly in second person conversationally as "you" and refer to their profile as "your profile"**.**Do not use platitudes or generic praise. Your analysis must be insightful and actionable.**

        1.  **EXECUTIVE SUMMARY:**
             - A blunt, one-paragraph assessment of the profiles current state.
             - Clearly state its primary strength and its most critical weakness from a strategic branding perspective.

        2.  **STRATEGIC ANALYSIS & ACTION PLAN:**
            - **Headline & Summary Deconstruction:** Does their headline convey a clear value proposition or is it just a job title? Is their summary a compelling narrative or a lazy list of duties? Provide a rewritten example for BOTH the headline and the summary that is significantly more impactful.
            - **Experience Section Deep Dive:** Are they describing accomplishments or just listing tasks? Identify the weakest experience entry and rewrite it using the STAR method (Situation, Task, Action, Result) to demonstrate how it should be done. Focus on quantifiable outcomes.
            - **Skills & Endorsements Audit:** Are their skills aligned with their stated career goals (from the conversation summary)? List 3-5 critical skills they are missing and 3-5 skills they should remove as "noise".
            - **Overall Brand Cohesion:** Does the profile tell a coherent story? Or is it a disjointed collection of facts? Provide a concluding paragraph on their overall personal brand strategy and how to improve it.

        **CRITICAL DIRECTIVE:** Your analysis MUST be tailored based on the users stated goals in the conversation summary. If they want to be a Product Manager, your entire audit must be viewed through that lens. Every recommendation must push them closer to that goal.
        """
        
        task = Task(
            description=task_description, 
            agent=agent, 
            expected_output="A structured, professional-grade profile audit report with an executive summary and a detailed strategic action plan."
        )
        
       
        crew = Crew(agents=[agent], tasks=[task])
        return {"messages": [AIMessage(content=str(crew.kickoff()))]}



    def provide_career_path_node(self, state: GraphState) -> dict:
        logger.info("--- NODE: Providing Career Path ---")
        if not state.get("profile_data"): 
            return {"messages": [AIMessage(content="To give you a personalized career path, I first need your LinkedIn profile. Could you please provide the URL or text?")]}

        
        prompt_data = self._create_contextual_prompt_data(state)
        
        agent = self.agents.career_path_agent()
        
        
        task_description = f"""
        (YOU ARE PART OF A MULTI AGENTIC WORKFLOW CULMINATING AS A USER FACING CONVERSATIONAL CHATBOT WHERE YOU ARE ONE OF THE AGENTS)

        **OPERATIONAL BRIEFING: Career Trajectory Architecture**

        You are an Executive Career Counselor & Leadership Mentor. Your task is to create a realistic, step-by-step career roadmap for the user.


        <CURRENT USER QUERY>
         THE USERs LATEST QUERY: 
        
        "{prompt_data['latest_user_question']}".
        </CURRENT USER QUERY>

        # CONTEXT OF OUR CONVERSATION

        ===CONVERSATION SUMMARY===
        {prompt_data['conversation_summary']}
        
        === RECENT CONVERSATION TURNS & DIALOGUE ===
        {prompt_data['recent_conversation_turns']}

        === USERs PROFILE DATA ===
        {prompt_data['profile_data_json']}


        **MISSION:**
         As per the mentees latest request Your mission is to architect a realistic, multi-stage career roadmap to take them from their current position to their stated long-term goal. **You must address the user directly in second person conversationally as "you" and refer to their profile as "your profile"**. This is not a list of online courses; it is a strategic blueprint for career advancement. **You must address the user directly in second person conversationally as "you" and refer to their profile as "your profile"**. Your tone should be that of a seasoned mentor: wise, strategic, and focused on long-term value.
            1.  **THE STRATEGIC OVERVIEW:**
            - A summary of the journey, acknowledging the starting point and the destination.
            - State the realistic estimated timeframe for this transition (e.g., "This is a 3-5 year journey requiring focused effort.").
            - Identify the single biggest challenge they will face in this transition (e.g., "Your biggest challenge will be shifting perception from a 'technical doer' to a strategic leader.'").

        2.  **PHASE-BASED ACTION PLAN:**
            - **Phase 1: Foundation (Next 6-12 Months):** Detail the immediate actions required. This should focus on leveraging their CURRENT role.
                - *Skill Acquisition:* What 2-3 specific skills must they acquire? Be specific (e.g., "Master User Story mapping," not "Learn Agile").
                - *Internal Visibility:* What kind of projects should they volunteer for *at their current company* to build relevant experience?
            - **Phase 2: Transition (Years 1-3):** Detail the steps needed to make the first major move.
                - *Target Role:* What is the logical "next-step" role? (e.g., "Associate Product Manager" or "Technical Product Manager").
                - *Resume & Branding:* How must their personal brand and resume narrative evolve by this stage?
                - *Networking:* What specific types of people should they be building relationships with? (e.g., "VPs of Product in B2B SaaS companies").
            - **Phase 3: Acceleration (Years 3-5+):** Detail the strategy for growth *after* landing the transition role.
                - *Performance Milestones:* What defines success in the new role to set them up for the next promotion?
                - *Leadership Development:* What specific leadership competencies must be demonstrated to move towards their ultimate goal?

        **CRITICAL DIRECTIVE:** Every piece of advice MUST be grounded in the mentees current reality as described in their profile. Constantly refer back to their existing skills and experience as the foundation for your recommendations.
        """
        
        task = Task(description=task_description, agent=agent, expected_output="A detailed, multi-phase strategic career roadmap, providing a long-term, actionable blueprint for the user")
        crew = Crew(agents=[agent], tasks=[task])
        return {"messages": [AIMessage(content=str(crew.kickoff()))]}
    

    def analyze_job_fit_node(self, state: GraphState) -> dict:
        logger.info("--- NODE: Analyzing Job Fit ---")
        
        
        if not state.get("profile_data"): 
            return {"messages": [AIMessage(content="I can definitely help with a job fit analysis, but I need your LinkedIn profile first. Could you please share it?")]}


        prompt_data = self._create_contextual_prompt_data(state)
        
        agent = self.agents.job_fit_analyzer_agent()

        
        task_description = f"""
        (YOU ARE PART OF A MULTI AGENTIC WORKFLOW CULMINATING AS A USER FACING CONVERSATIONAL CHATBOT WHERE YOU ARE ONE OF THE AGENTS)


        **OPERATIONAL BRIEFING: Candidate-Role Fitment Analysis**

        You are a Veteran Technical Recruiter & Hiring Manager. Your task is to provide a detailed comparison between the users profile and a specific job role or description.


        <CURRENT USER QUERY>
         THE USERs LATEST QUERY: 
        
        "{prompt_data['latest_user_question']}".
        </CURRENT USER QUERY>

        

        === CONTEXT OF OUR CONVERSATION ===
        {prompt_data['conversation_summary']}
        
        === RECENT CONVERSATION TURNS ===
        {prompt_data['recent_conversation_turns']}

        === USERs PROFILE DATA ===
        {prompt_data['profile_data_json']}

       **MISSION:**
        As per the candidates latest request. Your mission is to perform a rigorous, unsentimental analysis of their profile against the target role.**You must address the user directly in second person conversationally as "you" and refer to their profile as "your profile"**. Your perspective is that of a gatekeeper deciding if this candidate is worth a 30-minute screening call. Your tone should be direct, professional, and based on evidence from their profile.
        
        Analyze the users profile against the job role mentioned in their latest request.

        **CRITICAL INSTRUCTION:**
        - If the users request contains a full job description, use that for a detailed, line-by-line comparison.
        - If the users request ONLY contains a job title (e.g., "Senior Product Manager"), you must INFER a standard, industry-accepted job description for that title and perform the analysis against that inferred standard. State that you are doing so (e.g., "Based on a standard job description for a Senior Product Manager...").

        1.  **THE BOTTOM LINE (HIRING DECISION):**
            - Start with a single sentence: "Based on the provided information, my decision would be to [PASS ON / ADVANCE TO SCREENING] this candidate."
            - Follow with a "Fitment Score" from 0% to 100%.

        2.  **EVIDENCE-BASED RATIONALE:**
            - **Signal (Reasons to Advance):** A bulleted list of the strongest points of alignment between their profile and the role. Each point must explicitly connect a specific part of their profile (e.g., "Their experience in 'Quantitative Research at J.P. Morgan'") to a requirement of the job.
            - **Noise & Gaps (Reasons to Pass):** A bulleted list of the most significant gaps or weaknesses. Be specific. Instead of "lacks leadership experience," say "The profile provides no evidence of project leadership, team management, or mentorship, which is a critical requirement for this role."

        3.  **STRATEGIC PREPARATION PLAN (If they were my candidate):**
            - **Immediate Profile Edits:** Provide 2-3 specific, "copy-pasteable" rewrites for their profile summary or experience bullet points that would directly address the identified gaps for THIS job.
            - **Interview Talking Points:** List 3 key talking points they MUST prepare to discuss in an interview to overcome the perceived weaknesses in their profile. For example: "Be prepared to discuss the 'XYZ project' and frame it as a leadership initiative, even if it wasn't official."

        **CRITICAL DIRECTIVE:** If the user only provided a job title, you must first state what standard industry expectations you are using for that title before beginning your analysis.

       """
        
        task = Task(
            description=task_description,
            agent=agent,
            expected_output="A direct, evidence-based fitment report starting with a clear hiring decision and score, followed by a detailed rationale and a strategic preparation plan."
        )
        
        
        crew = Crew(agents=[agent], tasks=[task])
        return {"messages": [AIMessage(content=str(crew.kickoff()))]}


    def general_chat_node(self, state: GraphState) -> dict:
        logger.info("--- NODE: General Chat ---")
        
       
        prompt_data = self._create_contextual_prompt_data(state)
        
        agent = self.agents.general_chat_agent()

        
        task_description = f"""

        (YOU ARE PART OF A MULTI AGENTIC WORKFLOW CULMINATING AS A USER FACING CONVERSATIONAL CHATBOT WHERE YOU ARE THE GLUE THAT HOLDS THE USER EXPERIENCE TOGETHER. YOU ARE THE USER FACING AGENT THAT ENSURES THE USER NEVER FEELS LIKE THEY ARE BEING PASSED BETWEEN DIFFERENT BOTS. YOU ARE THE CENTRAL INTELLIGENCE OF THE CONVERSATION)

        **OPERATIONAL BRIEFING: Conversational Interface Management**
        You are an Empathetic AI Career Coach. Your job is to have a natural, helpful conversation.


        <CURRENT USER QUERY>
         THE USERs LATEST QUERY: 
        
        "{prompt_data['latest_user_question']}".
        </CURRENT USER QUERY>

        === CONTEXT OF OUR PREVIOUS CONVERSATION ===
        {prompt_data['conversation_summary']}
        
        === RECENT CONVERSATION TURNS ===
        {prompt_data['recent_conversation_turns']}

        === USERs PROFILE DATA (If available) ===
        {prompt_data['profile_data_json']}

        **MISSION:**
        As per the users latest message. Your primary goal is to maintain a coherent, helpful, and natural conversation. You are the "face" of the service.

        **RESPONSE PROTOCOL:**

        1.  **Acknowledge and Validate:** Always start by acknowledging the users last statement. Show you've understood it by referencing the context from the summary (e.g., "Thats a great follow-up question about the career path we were just discussing...").
        2.  **Provide a Direct Answer:** If its a simple question you can answer, do so helpfully.
        3.  **Guide if Necessary:**
            - If the users question requires a profile and one hasn't been provided, you must gently guide them: "To give you the best possible answer for that, I'd need to understand your background. Could you please share your LinkedIn profile URL or text?"
            - If the users question seems to hint at a more complex task (like a job fit or profile analysis), set the stage for it. For example: "It sounds like you're wondering how your profile stacks up against that role. Would you like me to perform a detailed Job Fit Analysis for you?"
        4.  **Maintain Persona:** Your tone should always be encouraging, clear, and professional. End with an open-ended question to keep the conversation flowing naturally.

       **CRITICAL DIRECTIVE:** You are the glue that holds the user experience together. Your main job is to ensure the user never feels like they are being passed between different bots. Be the consistent, central intelligence of the conversation.
       **CRITICAL DIRECTIVE 2: DO NOT HALLUCINATE CAPABILITIES.**
        If the user asks for something you cannot do (like search for live jobs, connect to a real person, or access external websites), you MUST state your limitation clearly and gracefully. Do NOT pretend to have specialists or tools that are not part of your core functions (profile analysis, job fit, career roadmaps). A good response would be: "While I can't search for live job postings, my expertise is in getting your profile ready for those applications.
        """
        
        task = Task(description=task_description, agent=agent, expected_output="A natural, helpful, and context-aware conversational response that maintains user engagement and guides them appropriately.")
        crew = Crew(agents=[agent], tasks=[task])
        return {"messages": [AIMessage(content=str(crew.kickoff()))]}




def create_linkedin_optimization_graph():
    return LinkedInOptimizationGraph().graph
