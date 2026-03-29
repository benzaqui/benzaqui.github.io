"""
Intent Classifier Agent

This module handles categorizing email intent and urgency for proper routing decisions.
It analyzes email content to determine the appropriate intent category and priority level.
"""

import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ..state import (
    WorkflowState, 
    IntentCategory, 
    Priority,
    IntentClassificationResult,
    validate_intent_classification
)


class IntentClassificationOutput(BaseModel):
    """Structured output for intent classification."""
    intent_category: str = Field(description="The primary intent category of the email")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
    priority: str = Field(description="Priority level: low, medium, high, or urgent")
    urgency_indicators: List[str] = Field(description="List of urgency indicators found in the email")
    keywords_found: List[str] = Field(description="Key terms that influenced the classification")
    reasoning: str = Field(description="Detailed explanation of the classification decision")


class IntentClassifier:
    """
    Specialized agent for classifying email intent and determining priority.
    
    This agent handles:
    - Analyzing email content for intent classification
    - Determining priority levels based on urgency indicators
    - Extracting relevant keywords and patterns
    - Providing confidence scores and reasoning
    - Supporting multiple classification strategies
    """
    
    def __init__(self, model_name: str = "anthropic:claude-3-5-sonnet-20241022"):
        """Initialize the intent classifier with specified LLM model."""
        self.model_name = model_name
        self.llm = self._initialize_model()
        self.output_parser = PydanticOutputParser(pydantic_object=IntentClassificationOutput)
        
        # Intent classification patterns
        self.intent_patterns = {
            IntentCategory.IT_SUPPORT: [
                r'\b(computer|laptop|desktop|server|network|wifi|internet|vpn)\b',
                r'\b(software|application|program|system|database)\b',
                r'\b(login|password|access|authentication|permissions)\b',
                r'\b(email|outlook|teams|office|sharepoint)\b'
            ],
            IntentCategory.PASSWORD_RESET: [
                r'\b(password|login|sign.?in|authentication|access)\b',
                r'\b(reset|forgot|forgotten|locked|unlock)\b',
                r'\b(account|username|credentials)\b'
            ],
            IntentCategory.HARDWARE_REQUEST: [
                r'\b(laptop|computer|monitor|keyboard|mouse|printer)\b',
                r'\b(hardware|equipment|device|machine)\b',
                r'\b(request|need|order|purchase|buy)\b',
                r'\b(new|replacement|additional|extra)\b'
            ],
            IntentCategory.SOFTWARE_ISSUE: [
                r'\b(software|application|program|app)\b',
                r'\b(error|bug|crash|freeze|slow|not.working)\b',
                r'\b(install|update|upgrade|patch)\b',
                r'\b(license|activation|subscription)\b'
            ],
            IntentCategory.TECHNICAL_ISSUE: [
                r'\b(technical|tech|system|server|network)\b',
                r'\b(issue|problem|error|failure|down|outage)\b',
                r'\b(not.working|broken|failed|crashed)\b'
            ],
            IntentCategory.HR_INQUIRY: [
                r'\b(hr|human.resources|personnel|employee)\b',
                r'\b(payroll|salary|benefits|insurance|vacation)\b',
                r'\b(policy|procedure|handbook|training)\b',
                r'\b(leave|time.off|sick|medical|maternity)\b'
            ],
            IntentCategory.FACILITIES: [
                r'\b(facilities|building|office|room|space)\b',
                r'\b(maintenance|repair|cleaning|hvac|temperature)\b',
                r'\b(parking|security|access.card|badge)\b',
                r'\b(furniture|desk|chair|supplies)\b'
            ],
            IntentCategory.BILLING: [
                r'\b(billing|invoice|payment|charge|cost)\b',
                r'\b(account|subscription|license|fee)\b',
                r'\b(refund|credit|dispute|error)\b'
            ],
            IntentCategory.ACCOUNT_ACCESS: [
                r'\b(account|access|permission|authorization)\b',
                r'\b(locked|disabled|suspended|blocked)\b',
                r'\b(activate|enable|restore|unlock)\b'
            ]
        }
        
        # Priority indicators
        self.priority_indicators = {
            Priority.URGENT: [
                r'\b(urgent|emergency|critical|asap|immediately)\b',
                r'\b(down|outage|not.working|broken|failed)\b',
                r'\b(help!|please.help|need.help.now)\b',
                r'\b(production|live|customer.facing)\b'
            ],
            Priority.HIGH: [
                r'\b(important|priority|high.priority|soon)\b',
                r'\b(deadline|due|meeting|presentation)\b',
                r'\b(issue|problem|error|trouble)\b',
                r'\b(quickly|fast|rapid|expedite)\b'
            ],
            Priority.MEDIUM: [
                r'\b(request|need|require|would.like)\b',
                r'\b(question|inquiry|help|support)\b',
                r'\b(when.possible|convenient|time)\b'
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
                return ChatAnthropic(model="claude-3-5-sonnet-20241022")
        except Exception as e:
            print(f"Warning: Failed to initialize {self.model_name}, falling back to default")
            return ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    def classify_intent(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Main classification function for determining email intent and priority.
        
        Args:
            state: Current workflow state containing email data
            
        Returns:
            Updated state with intent classification results
        """
        try:
            # Extract email content
            email_state = state.get("email_state", {})
            if not email_state or not email_state.get("subject"):
                raise ValueError("No email data found in state")
            
            subject = email_state.get("subject", "")
            body = email_state.get("body", "")
            sender_email = email_state.get("sender_email", "")
            
            # Perform pattern-based classification
            pattern_result = self._classify_with_patterns(subject, body)
            
            # Perform LLM-based classification
            llm_result = self._classify_with_llm(subject, body, sender_email)
            
            # Combine results
            final_result = self._combine_classification_results(pattern_result, llm_result)
            
            # Create IntentClassificationResult
            intent_result = IntentClassificationResult(
                intent_category=final_result["intent_category"],
                confidence_score=final_result["confidence_score"],
                priority=final_result["priority"],
                urgency_indicators=final_result["urgency_indicators"],
                keywords_found=final_result["keywords_found"],
                reasoning=final_result["reasoning"]
            )
            
            # Validate the classification result
            validation_errors = validate_intent_classification(intent_result)
            if validation_errors:
                raise ValueError(f"Classification validation failed: {validation_errors}")
            
            # Update routing state
            routing_state = state.get("routing_state", {})
            routing_state.update({
                "intent_result": intent_result,
                "classification_timestamp": datetime.now(),
                "requires_human_review": final_result["confidence_score"] < 0.7
            })
            
            # Update workflow state
            updated_state = {
                **state,
                "routing_state": routing_state,
                "current_step": "knowledge_retrieval",
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Intent classified as {intent_result['intent_category'].value} "
                               f"with {intent_result['priority'].value} priority "
                               f"(confidence: {intent_result['confidence_score']:.2f})",
                        name="intent_classifier"
                    )
                ]
            }
            
            return updated_state
            
        except Exception as e:
            # Handle classification errors
            error_message = f"Intent classification failed: {str(e)}"
            
            return {
                **state,
                "errors": state.get("errors", []) + [{
                    "component": "intent_classifier",
                    "error": error_message,
                    "timestamp": datetime.now()
                }],
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Intent classification failed: {error_message}",
                        name="intent_classifier"
                    )
                ]
            }
    
    def _classify_with_patterns(self, subject: str, body: str) -> Dict[str, Any]:
        """Classify intent using regex patterns."""
        combined_text = f"{subject} {body}".lower()
        
        # Score each intent category
        intent_scores = {}
        keywords_found = []
        
        for intent_category, patterns in self.intent_patterns.items():
            score = 0
            category_keywords = []
            
            for pattern in patterns:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                if matches:
                    score += len(matches)
                    category_keywords.extend(matches)
            
            if score > 0:
                intent_scores[intent_category] = score
                keywords_found.extend(category_keywords)
        
        # Determine best intent category
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            max_score = intent_scores[best_intent]
            confidence = min(max_score / 10.0, 1.0)  # Normalize to 0-1
        else:
            best_intent = IntentCategory.GENERAL_INQUIRY
            confidence = 0.3
        
        # Classify priority
        priority = self._classify_priority_with_patterns(combined_text)
        urgency_indicators = self._extract_urgency_indicators(combined_text)
        
        return {
            "intent_category": best_intent,
            "confidence_score": confidence,
            "priority": priority,
            "urgency_indicators": urgency_indicators,
            "keywords_found": list(set(keywords_found)),
            "reasoning": f"Pattern-based classification found {len(keywords_found)} relevant keywords"
        }
    
    def _classify_priority_with_patterns(self, text: str) -> Priority:
        """Classify priority using regex patterns."""
        for priority, patterns in self.priority_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return priority
        
        return Priority.LOW
    
    def _extract_urgency_indicators(self, text: str) -> List[str]:
        """Extract urgency indicators from text."""
        indicators = []
        
        for priority, patterns in self.priority_indicators.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                indicators.extend(matches)
        
        return list(set(indicators))
    
    def _classify_with_llm(self, subject: str, body: str, sender_email: str) -> Dict[str, Any]:
        """Classify intent using LLM analysis."""
        try:
            # Create classification prompt
            system_prompt = self._create_classification_prompt()
            
            # Prepare email content
            email_content = f"""
Subject: {subject}
From: {sender_email}
Body: {body}
"""
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=email_content)
            ]
            
            # Get structured output from LLM
            llm_with_parser = self.llm.with_structured_output(IntentClassificationOutput)
            result = llm_with_parser.invoke(messages)
            
            # Convert string enums to actual enums
            intent_category = IntentCategory(result.intent_category.lower().replace(" ", "_"))
            priority = Priority(result.priority.lower())
            
            return {
                "intent_category": intent_category,
                "confidence_score": result.confidence_score,
                "priority": priority,
                "urgency_indicators": result.urgency_indicators,
                "keywords_found": result.keywords_found,
                "reasoning": result.reasoning
            }
            
        except Exception as e:
            # Fallback to pattern-based classification
            print(f"LLM classification failed: {e}, falling back to patterns")
            return self._classify_with_patterns(subject, body)
    
    def _create_classification_prompt(self) -> str:
        """Create system prompt for intent classification."""
        intent_categories = [category.value for category in IntentCategory]
        priority_levels = [priority.value for priority in Priority]
        
        return f"""You are an expert email intent classifier for a helpdesk system. 
Your task is to analyze incoming support emails and classify them accurately.

INTENT CATEGORIES:
{', '.join(intent_categories)}

PRIORITY LEVELS:
{', '.join(priority_levels)}

INSTRUCTIONS:
1. Analyze the email subject, sender, and body content
2. Identify the primary intent category that best matches the request
3. Determine the priority level based on urgency indicators
4. Extract relevant keywords that influenced your decision
5. Provide a confidence score between 0.0 and 1.0
6. Give detailed reasoning for your classification

URGENCY INDICATORS:
- Urgent/Critical: System outages, security issues, production problems
- High: Important deadlines, blocking issues, management requests
- Medium: Standard requests, questions, non-blocking issues
- Low: General inquiries, documentation requests, suggestions

Respond with structured output including intent_category, confidence_score, priority, urgency_indicators, keywords_found, and reasoning."""
    
    def _combine_classification_results(
        self, 
        pattern_result: Dict[str, Any], 
        llm_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine pattern-based and LLM-based classification results."""
        
        # Use LLM result as primary if confidence is high
        if llm_result["confidence_score"] >= 0.7:
            primary_result = llm_result
            secondary_result = pattern_result
        else:
            # Use pattern result if LLM confidence is low
            primary_result = pattern_result
            secondary_result = llm_result
        
        # Combine keywords and urgency indicators
        combined_keywords = list(set(
            primary_result["keywords_found"] + secondary_result["keywords_found"]
        ))
        
        combined_urgency = list(set(
            primary_result["urgency_indicators"] + secondary_result["urgency_indicators"]
        ))
        
        # Adjust confidence based on agreement
        if primary_result["intent_category"] == secondary_result["intent_category"]:
            # Boost confidence if both methods agree
            final_confidence = min(primary_result["confidence_score"] + 0.1, 1.0)
        else:
            # Reduce confidence if methods disagree
            final_confidence = max(primary_result["confidence_score"] - 0.1, 0.1)
        
        # Use higher priority if there's disagreement
        priority_order = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.URGENT]
        final_priority = max(
            primary_result["priority"], 
            secondary_result["priority"],
            key=lambda p: priority_order.index(p)
        )
        
        return {
            "intent_category": primary_result["intent_category"],
            "confidence_score": final_confidence,
            "priority": final_priority,
            "urgency_indicators": combined_urgency,
            "keywords_found": combined_keywords,
            "reasoning": f"Combined analysis: {primary_result['reasoning']} "
                        f"Pattern confidence: {pattern_result['confidence_score']:.2f}, "
                        f"LLM confidence: {llm_result['confidence_score']:.2f}"
        }


# Tool function for LangGraph integration
@tool
def classify_intent_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool function for classifying email intent in LangGraph workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated workflow state with intent classification results
    """
    classifier = IntentClassifier()
    return classifier.classify_intent(state)


# Export the main class and tool
__all__ = ["IntentClassifier", "classify_intent_tool"]
