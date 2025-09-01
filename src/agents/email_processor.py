"""
Email Processor Agent

This module handles parsing and extracting email metadata from incoming support emails.
It processes raw email data and populates the EmailState with structured information.
"""

import re
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.parser import Parser
from email.policy import default

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from ..state import (
    WorkflowState, 
    EmailState, 
    Priority,
    validate_email_state
)


class EmailProcessor:
    """
    Specialized agent for processing and extracting metadata from emails.
    
    This agent handles:
    - Parsing raw email content
    - Extracting key metadata (sender, subject, body, attachments)
    - Identifying priority indicators
    - Validating email structure
    - Generating unique identifiers
    """
    
    def __init__(self, model_name: str = "anthropic:claude-3-5-sonnet-20241022"):
        """Initialize the email processor with specified LLM model."""
        self.model_name = model_name
        self.llm = self._initialize_model()
        
        # Priority keywords for urgency detection
        self.priority_keywords = {
            Priority.URGENT: [
                "urgent", "emergency", "critical", "asap", "immediately", 
                "down", "outage", "broken", "not working", "help!"
            ],
            Priority.HIGH: [
                "important", "priority", "soon", "quickly", "deadline",
                "issue", "problem", "error", "failed"
            ],
            Priority.MEDIUM: [
                "request", "need", "question", "help", "support"
            ]
        }
    
    def _initialize_model(self):
        """Initialize the appropriate LLM model based on preference hierarchy."""
        try:
            if "anthropic" in self.model_name.lower():
                return ChatAnthropic(model="claude-3-5-sonnet-20241022")
            elif "openai" in self.model_name.lower():
                return ChatOpenAI(model="gpt-4o")
            elif "google" in self.model_name.lower():
                return ChatGoogleGenerativeAI(model="gemini-1.5-pro")
            else:
                # Default to Anthropic
                return ChatAnthropic(model="claude-3-5-sonnet-20241022")
        except Exception as e:
            # Fallback to next available model
            print(f"Warning: Failed to initialize {self.model_name}, falling back to default")
            return ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    def process_email(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Main processing function for extracting email metadata.
        
        Args:
            state: Current workflow state containing raw email data
            
        Returns:
            Updated state with processed email information
        """
        try:
            # Extract raw email data from state or messages
            raw_email_data = self._extract_raw_email_data(state)
            
            # Parse email structure
            parsed_email = self._parse_email_structure(raw_email_data)
            
            # Extract metadata
            email_metadata = self._extract_email_metadata(parsed_email)
            
            # Detect priority indicators
            priority = self._detect_priority(email_metadata)
            
            # Generate unique identifiers
            email_id = self._generate_email_id()
            thread_id = self._extract_thread_id(email_metadata)
            
            # Create EmailState
            email_state = EmailState(
                subject=email_metadata.get("subject", ""),
                body=email_metadata.get("body", ""),
                sender_email=email_metadata.get("sender_email", ""),
                sender_name=email_metadata.get("sender_name"),
                recipient_email=email_metadata.get("recipient_email", ""),
                timestamp=email_metadata.get("timestamp", datetime.now()),
                attachments=email_metadata.get("attachments", []),
                email_id=email_id,
                thread_id=thread_id,
                is_processed=True,
                processing_errors=[]
            )
            
            # Validate the processed email state
            validation_errors = validate_email_state(email_state)
            if validation_errors:
                email_state["processing_errors"] = validation_errors
                email_state["is_processed"] = False
            
            # Update workflow state
            updated_state = {
                **state,
                "email_state": email_state,
                "current_step": "intent_classification",
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Email processed successfully. Subject: '{email_state['subject']}', "
                               f"Sender: {email_state['sender_email']}, Priority: {priority.value}",
                        name="email_processor"
                    )
                ]
            }
            
            return updated_state
            
        except Exception as e:
            # Handle processing errors
            error_message = f"Email processing failed: {str(e)}"
            
            # Create error state
            error_email_state = state.get("email_state", {})
            error_email_state.update({
                "is_processed": False,
                "processing_errors": [error_message]
            })
            
            return {
                **state,
                "email_state": error_email_state,
                "errors": state.get("errors", []) + [{
                    "component": "email_processor",
                    "error": error_message,
                    "timestamp": datetime.now()
                }],
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Email processing failed: {error_message}",
                        name="email_processor"
                    )
                ]
            }
    
    def _extract_raw_email_data(self, state: WorkflowState) -> Dict[str, Any]:
        """Extract raw email data from workflow state or messages."""
        # Check if email data is already in email_state
        if state.get("email_state") and state["email_state"].get("subject"):
            return state["email_state"]
        
        # Extract from messages if available
        if state.get("messages"):
            for message in state["messages"]:
                if isinstance(message, HumanMessage):
                    # Try to parse email from message content
                    return self._parse_message_content(message.content)
        
        # Return empty structure if no email data found
        return {
            "subject": "",
            "body": "",
            "sender_email": "",
            "recipient_email": ""
        }
    
    def _parse_message_content(self, content: str) -> Dict[str, Any]:
        """Parse email information from message content."""
        # Simple parsing - in production, this would be more sophisticated
        lines = content.split('\n')
        email_data = {
            "subject": "",
            "body": content,
            "sender_email": "",
            "recipient_email": ""
        }
        
        # Look for email patterns
        for line in lines:
            if line.lower().startswith("subject:"):
                email_data["subject"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("from:"):
                email_data["sender_email"] = self._extract_email_address(line)
            elif line.lower().startswith("to:"):
                email_data["recipient_email"] = self._extract_email_address(line)
        
        return email_data
    
    def _extract_email_address(self, text: str) -> str:
        """Extract email address from text using regex."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else ""
    
    def _parse_email_structure(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse email structure and extract components."""
        # If raw_data is already structured, return as-is
        if isinstance(raw_data, dict) and "subject" in raw_data:
            return raw_data
        
        # Handle raw email string parsing
        if isinstance(raw_data, str):
            parser = Parser(policy=default)
            try:
                msg = parser.parsestr(raw_data)
                return {
                    "subject": msg.get("Subject", ""),
                    "sender_email": msg.get("From", ""),
                    "recipient_email": msg.get("To", ""),
                    "body": self._extract_body(msg),
                    "timestamp": self._parse_timestamp(msg.get("Date")),
                    "attachments": self._extract_attachments(msg)
                }
            except Exception:
                # Fallback to simple text parsing
                return self._parse_message_content(raw_data)
        
        return raw_data
    
    def _extract_body(self, msg) -> str:
        """Extract email body from parsed message."""
        if msg.is_multipart():
            body_parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body_parts.append(part.get_content())
            return "\n".join(body_parts)
        else:
            return msg.get_content()
    
    def _parse_timestamp(self, date_str: Optional[str]) -> datetime:
        """Parse email timestamp."""
        if not date_str:
            return datetime.now()
        
        try:
            # Simple timestamp parsing - in production, use more robust parsing
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except Exception:
            return datetime.now()
    
    def _extract_attachments(self, msg) -> List[Dict[str, Any]]:
        """Extract attachment information from email."""
        attachments = []
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    filename = part.get_filename()
                    if filename:
                        attachments.append({
                            "filename": filename,
                            "content_type": part.get_content_type(),
                            "size": len(part.get_content()) if part.get_content() else 0
                        })
        
        return attachments
    
    def _extract_email_metadata(self, parsed_email: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and enrich email metadata."""
        metadata = parsed_email.copy()
        
        # Extract sender name from email address
        if metadata.get("sender_email") and not metadata.get("sender_name"):
            sender_email = metadata["sender_email"]
            # Extract name from "Name <email@domain.com>" format
            if "<" in sender_email and ">" in sender_email:
                name_part = sender_email.split("<")[0].strip()
                email_part = sender_email.split("<")[1].split(">")[0].strip()
                metadata["sender_name"] = name_part if name_part else None
                metadata["sender_email"] = email_part
        
        # Ensure timestamp is datetime object
        if not isinstance(metadata.get("timestamp"), datetime):
            metadata["timestamp"] = datetime.now()
        
        return metadata
    
    def _detect_priority(self, email_metadata: Dict[str, Any]) -> Priority:
        """Detect email priority based on content analysis."""
        subject = email_metadata.get("subject", "").lower()
        body = email_metadata.get("body", "").lower()
        combined_text = f"{subject} {body}"
        
        # Check for urgent keywords
        for keyword in self.priority_keywords[Priority.URGENT]:
            if keyword in combined_text:
                return Priority.URGENT
        
        # Check for high priority keywords
        for keyword in self.priority_keywords[Priority.HIGH]:
            if keyword in combined_text:
                return Priority.HIGH
        
        # Check for medium priority keywords
        for keyword in self.priority_keywords[Priority.MEDIUM]:
            if keyword in combined_text:
                return Priority.MEDIUM
        
        # Default to low priority
        return Priority.LOW
    
    def _generate_email_id(self) -> str:
        """Generate unique email identifier."""
        return f"email_{uuid.uuid4().hex[:12]}"
    
    def _extract_thread_id(self, email_metadata: Dict[str, Any]) -> Optional[str]:
        """Extract or generate thread ID for email conversation tracking."""
        # Look for thread indicators in subject
        subject = email_metadata.get("subject", "")
        
        # Check for "Re:" or "Fwd:" patterns
        if subject.lower().startswith(("re:", "fwd:", "fw:")):
            # Generate thread ID based on cleaned subject
            cleaned_subject = re.sub(r'^(re:|fwd?:)\s*', '', subject, flags=re.IGNORECASE)
            return f"thread_{hash(cleaned_subject.lower()) % 1000000}"
        
        return None


# Tool function for LangGraph integration
@tool
def process_email_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool function for processing emails in LangGraph workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated workflow state with processed email information
    """
    processor = EmailProcessor()
    return processor.process_email(state)


# Export the main class and tool
__all__ = ["EmailProcessor", "process_email_tool"]
