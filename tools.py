"""
Custom tools for LinkedIn Profile Optimization Assistant
"""
import os
import time
import json
import requests
from typing import Optional, Dict, Any
from crewai.tools import BaseTool
from pydantic import Field
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinkedInDataTool(BaseTool):
    """Tool for handling LinkedIn profile data - either from URL scraping or direct text input"""
    
    name: str = "LinkedIn Data Handler"
    description: str = """
    A tool that processes LinkedIn profile data from either a URL or direct text input.
    For URLs: Attempts to scrape using Apify API
    For text: Processes the provided LinkedIn profile text directly
    Returns structured profile data for analysis.
    """
    
    def _run(self, input_data: str) -> Dict[str, Any]:
        """
        Process LinkedIn profile data from URL or text
        
        Args:
            input_data (str): Either a LinkedIn URL or profile text
            
        Returns:
            Dict[str, Any]: The processed profile data
        """
        try:
            # Check if input is a URL
            if input_data.strip().startswith("http") and "linkedin.com" in input_data.lower():
                logger.info("Processing LinkedIn URL")
                return self._scrape_from_url(input_data.strip())
            else:
                logger.info("Processing LinkedIn profile text")
                return self._process_text_input(input_data)
                
        except Exception as e:
            logger.error(f"Error processing LinkedIn data: {str(e)}")
            return {"error": f"Exception occurred during processing: {str(e)}"}
    
    def _scrape_from_url(self, linkedin_url: str) -> Dict[str, Any]:
        """
        Scrape LinkedIn profile using Apify API
        
        Args:
            linkedin_url (str): The LinkedIn profile URL to scrape
            
        Returns:
            Dict[str, Any]: The scraped profile data
        """
        try:
            # Get API credentials from environment
            api_token = os.getenv("APIFY_API_TOKEN")
            
            if not api_token:
                return {
                    "error": "APIFY_API_TOKEN not found. Please provide profile text instead or configure API token.",
                    "fallback": "text_input_required"
                }
            
            # Apify actor for LinkedIn scraping - Using working actor ID
            actor_id = "2SyF0bVxmgGr8IVCZ"  # dev_fusion/Linkedin-Profile-Scraper
            
            # Prepare the run input
            run_input = {
                "profileUrls": [linkedin_url],
                "sessionCookieValue": "",
                "includePdfs": False,
                "saveToKvStore": False
            }
            
            # Start the actor run
            run_url = f"https://api.apify.com/v2/acts/{actor_id}/runs"
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"Starting LinkedIn profile scrape for: {linkedin_url}")
            response = requests.post(run_url, json=run_input, headers=headers, timeout=30)
            
            if response.status_code != 201:
                return {
                    "error": f"Failed to start scraping job: {response.text}",
                    "fallback": "text_input_required"
                }
            
            run_data = response.json()
            run_id = run_data["data"]["id"]
            
            # Poll for results 
            result_url = f"https://api.apify.com/v2/acts/{actor_id}/runs/{run_id}"
            max_attempts = 12  # 2 minutes max wait time
            attempt = 0
            
            while attempt < max_attempts:
                time.sleep(10)
                attempt += 1
                
                logger.info(f"Checking scraping status... Attempt {attempt}/{max_attempts}")
                status_response = requests.get(result_url, headers=headers, timeout=10)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    run_status = status_data["data"]["status"]
                    
                    if run_status == "SUCCEEDED":
                        # Get the results
                        dataset_id = status_data["data"]["defaultDatasetId"]
                        dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
                        
                        results_response = requests.get(dataset_url, headers=headers, timeout=10)
                        if results_response.status_code == 200:
                            results = results_response.json()
                            if results:
                                logger.info("Successfully scraped LinkedIn profile")
                                return {
                                    "success": True,
                                    "data": results[0],
                                    "source": "scraped"
                                }
                            else:
                                return {
                                    "error": "No profile data found",
                                    "fallback": "text_input_required"
                                }
                        else:
                            return {
                                "error": f"Failed to fetch results: {results_response.text}",
                                "fallback": "text_input_required"
                            }
                    
                    elif run_status == "FAILED":
                        return {
                            "error": f"Scraping job failed: {status_data.get('data', {}).get('statusMessage', 'Unknown error')}",
                            "fallback": "text_input_required"
                        }
                    
                    # Continue polling if status is RUNNING or READY
                    logger.info(f"Job status: {run_status}, continuing to wait...")
                else:
                    return {
                        "error": f"Failed to check status: {status_response.text}",
                        "fallback": "text_input_required"
                    }
            
            return {
                "error": "Scraping job timed out after 2 minutes",
                "fallback": "text_input_required"
            }
            
        except Exception as e:
            logger.error(f"Error scraping LinkedIn profile: {str(e)}")
            return {
                "error": f"Exception occurred during scraping: {str(e)}",
                "fallback": "text_input_required"
            }
    
    def _process_text_input(self, profile_text: str) -> Dict[str, Any]:
        """
        Process LinkedIn profile from text input
        
        Args:
            profile_text (str): The LinkedIn profile text
            
        Returns:
            Dict[str, Any]: The processed profile data
        """
        try:
            # Basic text processing to extract structured data
            profile_data = {
                "success": True,
                "source": "text_input",
                "raw_text": profile_text,
                "processed_sections": self._extract_sections_from_text(profile_text)
            }
            
            logger.info("Successfully processed LinkedIn profile text")
            return profile_data
            
        except Exception as e:
            logger.error(f"Error processing profile text: {str(e)}")
            return {"error": f"Exception occurred during text processing: {str(e)}"}
    
    def _extract_sections_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract common LinkedIn sections from text
        
        Args:
            text (str): The profile text
            
        Returns:
            Dict[str, str]: Extracted sections
        """
        sections = {}
        
        #  keyword-based extraction
        text_lower = text.lower()
        
        # Try to extract name (usually at the beginning)
        lines = text.strip().split('\n')
        if lines:
            sections['name'] = lines[0].strip()
        
        # Extract sections based on common keywords
        if 'about' in text_lower or 'summary' in text_lower:
            sections['has_about'] = True
        
        if 'experience' in text_lower or 'work' in text_lower:
            sections['has_experience'] = True
            
        if 'education' in text_lower or 'university' in text_lower or 'college' in text_lower:
            sections['has_education'] = True
            
        if 'skills' in text_lower or 'technologies' in text_lower:
            sections['has_skills'] = True
            
        if 'certification' in text_lower or 'licensed' in text_lower:
            sections['has_certifications'] = True
        
        sections['word_count'] = len(text.split())
        sections['text_length'] = len(text)
        
        return sections
    
    async def _arun(self, input_data: str) -> Dict[str, Any]:
        """Async version of _run"""
        return self._run(input_data)



def get_linkedin_data_tool():
    """Get an instance of the LinkedIn data handler tool"""
    return LinkedInDataTool()


# For testing purposes
if __name__ == "__main__":
    # Test the tool
    linkedin_tool = get_linkedin_data_tool()
    
    print("LinkedIn Data Tool initialized successfully!")
    print(f"Tool Name: {linkedin_tool.name}")
    print(f"Tool Description: {linkedin_tool.description}")
