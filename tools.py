
import os
import time
import json
import requests
from typing import Dict, Any
from crewai.tools import BaseTool
import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field 
from typing import List, Optional
from langchain_google_genai.chat_models import  ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ParsedProfile(BaseModel):
    full_name: str = Field(description="The full name of the person.")
    headline: str = Field(description="The main professional headline under their name.")
    summary: str = Field(description="The 'About' section text. If not present, state 'No summary provided'.")
    experience_highlights: List[str] = Field(description="A list of the user's job titles and companies.")
    education_highlights: List[str] = Field(description="A list of the user's educational institutions.")


class ValidationResponse(BaseModel):
    """The output of the initial validation step to classify the user's input."""
    is_profile_data: bool = Field(description="Set to True if the text contains actual profile information (name, experience, etc.), otherwise False.")
    profile_text_only: Optional[str] = Field(description="If is_profile_data is True, this field contains ONLY the cleaned profile text, with all conversational filler like 'here is my profile' or 'check this out' completely removed. If is_profile_data is False, this should be null.")



class LinkedInDataTool(BaseTool):
    name: str = "LinkedIn Profile Processor"
    description: str = """A tool that validates, cleans, and processes LinkedIn profile data from a user's message.
    It intelligently determines if the input is a URL, profile text, or just a conversational message.
    If it's valid profile data, it parses it into a structured format.
    If it's not profile data, it reports that, allowing the system to route to a chat agent."""


    _utility_llm: Any = None

    def _get_utility_llm(self):
        """Initializes the utility LLM on first use."""
        if self._utility_llm is None:
            try:
                
                self._utility_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
                logger.info("Utility LLM (Gemini) for tool initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize utility LLM. Ensure GEMINI_API_KEY is set. Error: {e}")
                raise
        return self._utility_llm

    def _run(self, input_data: str) -> Dict[str, Any]:
        """The main execution method that decides whether to scrape or process text."""
        input_data = input_data.strip()
        if "linkedin.com/in/" in input_data.lower() and input_data.startswith("http"):
            logger.info("Detected LinkedIn URL, attempting to scrape.")
            return self._scrape_from_url(input_data)
        else:
            logger.info("Detected plain text input, processing with validation pipeline.")
            # This is where we call the robust text processing method
            return self._process_text_input_pipeline(input_data)

    def _scrape_from_url(self, linkedin_url: str) -> Dict[str, Any]:
        """Scrapes a LinkedIn profile using the Apify API."""
        api_token = os.getenv("APIFY_API_TOKEN")
        if not api_token:
            return {"error": "APIFY_API_TOKEN is not configured."}

        actor_id = "apify/linkedin-profile-scraper"
        run_url = f"https://api.apify.com/v2/acts/{actor_id}/runs?token={api_token}"
        run_input = {"startUrls": [{"url": linkedin_url}]}
        
        try:
            logger.info(f"Starting scrape for URL: {linkedin_url}")
            response = requests.post(run_url, json=run_input, timeout=30)
            response.raise_for_status()
            run_id = response.json()['data']['id']
            status_url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}"

            for _ in range(12):  
                time.sleep(10)
                status_response = requests.get(status_url, timeout=10)
                status_response.raise_for_status()
                status = status_response.json()['data']['status']
                logger.info(f"Scraping job status: {status}")
                
                if status == 'SUCCEEDED':
                    dataset_url = f"https://api.apify.com/v2/datasets/{status_response.json()['data']['defaultDatasetId']}/items?token={api_token}"
                    results = requests.get(dataset_url, timeout=10).json()
                    if results:
                        logger.info("Scraping successful.")
                        return {"success": True, "data": results[0]}
                    return {"error": "Scraping succeeded but returned no data."}
                elif status in ['FAILED', 'ABORTED']:
                    return {"error": f"Scraping job failed with status: {status}"}
            
            return {"success": False,"error": "Scraping job timed out."}
        except requests.exceptions.RequestException as e:
            logger.warning("Apify scraping is a placeholder. Returning an error.")
        return {"success": False, "error": "Apify scraping is not fully implemented in this example."}

    def _process_text_input_pipeline(self, user_message: str) -> Dict[str, Any]:
        """
        Processes raw text using a robust two-stage LLM pipeline: Validate -> Parse.
        This ensures only genuine, cleaned profile text is ever parsed.
        """
        try:
            utility_llm = self._get_utility_llm()
        except Exception as e:
            # If fails to init, we can't process, so it's not a success.
            return {"success": False, "error": str(e)}

        # --- Stage 1: Validate and Clean the input ---
        validator_parser = JsonOutputParser(pydantic_object=ValidationResponse)
        validator_prompt = ChatPromptTemplate.from_template(
            """You are a data validation bot. Analyze the user's message.
            Does it contain the substantive text of a LinkedIn profile (must include a name, a job title, and some experience)?
            A job description or a simple question is NOT a profile.

            CRITICAL: The user might include conversational filler like "here is my profile:", "sure, check this out...", or "okay, here it is:". You MUST ignore this filler and focus on the actual data.

            If it IS a profile, set is_profile_data to true and extract ONLY the profile text, removing all conversational filler (e.g., "here is my profile:", "sure, check this out...").
            If it is NOT a profile, set is_profile_data to false.

            {format_instructions}
            
            User Message:
            ---
            {user_message}
            ---
            """
        )
        validator_chain = validator_prompt | utility_llm | validator_parser
        
        try:
            validation_result = validator_chain.invoke({
                "user_message": user_message, 
                "format_instructions": validator_parser.get_format_instructions()
            })
        except Exception as e:
            logger.error(f"LLM validation call failed: {e}")
            return {"success": False, "error": "Could not validate input text."}
        
        # the crucial gatekeeper logic
        if not validation_result.get("is_profile_data"):
            logger.info("Tool determined input is not profile data.")
           
            return {"success": False, "data": "Input was determined to be conversational, not profile data."}

        cleaned_profile_text = validation_result.get("profile_text_only")
        if not cleaned_profile_text:
            logger.warning("Tool validated text as profile, but failed to extract it.")
            return {"success": False, "error": "Could not extract clean profile text from the message."}
        
        
        parser = JsonOutputParser(pydantic_object=ParsedProfile)
        parser_prompt = ChatPromptTemplate.from_template(
            """Parse the following CLEANED LinkedIn profile text into the specified JSON format.
            {format_instructions}
            Cleaned LinkedIn Profile Text: --- {cleaned_profile_text} ---
            """
        )
        parser_chain = parser_prompt | utility_llm | parser
        
        try:
            parsed_data = parser_chain.invoke({
                "cleaned_profile_text": cleaned_profile_text,
                "format_instructions": parser.get_format_instructions()
            })
            logger.info("Successfully parsed cleaned profile text into structured data.")
            #Return "success": True so the graph can update the state.
            return {"success": True, "data": parsed_data}
        except Exception as e:
            logger.error(f"LLM parsing call failed: {e}")
            return {"success": False, "error": "Could not parse the cleaned profile text."}

def get_linkedin_data_tool():
    """Factory function to get an instance of the tool."""
    return LinkedInDataTool()
