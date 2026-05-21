"""
Response Generator Agent

This module handles creating self-service responses and routing notifications
for support emails based on knowledge base articles and routing decisions.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

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
    RoutingQueue,
    KnowledgeBaseArticle
)


class ResponseType(Enum):
    """Types of responses that can be generated."""
    SELF_SERVICE = "self_service"
    ROUTING_NOTIFICATION = "routing_notification"
    ESCALATION_ALERT = "escalation_alert"
    ACKNOWLEDGMENT = "acknowledgment"
    FOLLOW_UP = "follow_up"


class ResponseOutput(BaseModel):
    """Structured output for generated responses."""
    response_type: str = Field(description="Type of response being generated")
    subject: str = Field(description="Email subject line for the response")
    body: str = Field(description="Main body content of the response")
    tone: str = Field(description="Tone of the response (professional, friendly, urgent)")
    call_to_action: Optional[str] = Field(description="Specific action requested from recipient")
    estimated_resolution: Optional[str] = Field(description="Estimated resolution timeframe")


@dataclass
class ResponseTemplate:
    """Template for generating responses."""
    response_type: ResponseType
    subject_template: str
    body_template: str
    tone: str
    variables: List[str]


class ResponseGenerator:
    """
    Specialized agent for generating appropriate responses and notifications.
    
    This agent handles:
    - Creating self-service response emails with relevant articles
    - Generating routing notifications for assigned teams
    - Composing escalation alerts for urgent issues
    - Crafting acknowledgment messages for users
    - Personalizing responses based on context and user information
    """
    
    def __init__(self, model_name: str = "anthropic:claude-3-5-sonnet-20241022"):
        """Initialize the response generator with specified LLM model."""
        self.model_name = model_name
        self.llm = self._initialize_model()
        self.output_parser = PydanticOutputParser(pydantic_object=ResponseOutput)
        
        # Response templates
        self.response_templates = self._initialize_response_templates()
        
        # Company/organization information
        self.organization_info = {
            "name": "Your Organization",
            "support_email": "support@yourorg.com",
            "support_phone": "1-800-SUPPORT",
            "knowledge_base_url": "https://kb.yourorg.com",
            "ticket_portal_url": "https://tickets.yourorg.com"
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
    
    def _initialize_response_templates(self) -> Dict[ResponseType, ResponseTemplate]:
        """Initialize response templates for different scenarios."""
        templates = {}
        
        # Self-service response template
        templates[ResponseType.SELF_SERVICE] = ResponseTemplate(
            response_type=ResponseType.SELF_SERVICE,
            subject_template="Re: {original_subject} - Self-Service Resources Available",
            body_template="""Dear {sender_name},

Thank you for contacting {organization_name} support regarding "{original_subject}".

We've found some helpful resources that may resolve your issue immediately:

{article_recommendations}

Please try these solutions first. If they don't resolve your issue, your request will be automatically routed to our {target_queue} team for further assistance.

{call_to_action}

Best regards,
{organization_name} Support Team
{support_contact_info}""",
            tone="helpful",
            variables=["sender_name", "organization_name", "original_subject", "article_recommendations", "target_queue", "call_to_action", "support_contact_info"]
        )
        
        # Routing notification template
        templates[ResponseType.ROUTING_NOTIFICATION] = ResponseTemplate(
            response_type=ResponseType.ROUTING_NOTIFICATION,
            subject_template="New Support Request: {original_subject} [{priority}]",
            body_template="""New support request assigned to {target_queue}:

From: {sender_email}
Subject: {original_subject}
Priority: {priority}
Intent: {intent_category}
Estimated Resolution: {estimated_resolution}

{email_summary}

{routing_reasoning}

Ticket ID: {ticket_id}
View in portal: {ticket_url}""",
            tone="professional",
            variables=["target_queue", "sender_email", "original_subject", "priority", "intent_category", "estimated_resolution", "email_summary", "routing_reasoning", "ticket_id", "ticket_url"]
        )
        
        # Escalation alert template
        templates[ResponseType.ESCALATION_ALERT] = ResponseTemplate(
            response_type=ResponseType.ESCALATION_ALERT,
            subject_template="URGENT: Escalation Required - {original_subject}",
            body_template="""URGENT ESCALATION REQUIRED

From: {sender_email}
Subject: {original_subject}
Priority: {priority}
Escalation Reason: {escalation_reason}

{email_summary}

This request requires immediate attention from {target_queue}.
Expected response time: {estimated_resolution}

Ticket ID: {ticket_id}
View in portal: {ticket_url}""",
            tone="urgent",
            variables=["sender_email", "original_subject", "priority", "escalation_reason", "email_summary", "target_queue", "estimated_resolution", "ticket_id", "ticket_url"]
        )
        
        # Acknowledgment template
        templates[ResponseType.ACKNOWLEDGMENT] = ResponseTemplate(
            response_type=ResponseType.ACKNOWLEDGMENT,
            subject_template="Re: {original_subject} - Request Received [Ticket #{ticket_id}]",
            body_template="""Dear {sender_name},

Thank you for contacting {organization_name} support. We have received your request regarding "{original_subject}".

Your ticket details:
- Ticket ID: {ticket_id}
- Priority: {priority}
- Assigned to: {target_queue}
- Estimated resolution: {estimated_resolution}

{next_steps}

You can track the status of your request at: {ticket_url}

If you have any questions, please reply to this email or contact us at {support_contact_info}.

Best regards,
{organization_name} Support Team""",
            tone="professional",
            variables=["sender_name", "organization_name", "original_subject", "ticket_id", "priority", "target_queue", "estimated_resolution", "next_steps", "ticket_url", "support_contact_info"]
        )
        
        return templates
    
    def generate_response(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Main response generation function.
        
        Args:
            state: Current workflow state containing all processing results
            
        Returns:
            Updated state with generated responses
        """
        try:
            # Extract state information
            email_state = state.get("email_state", {})
            routing_state = state.get("routing_state", {})
            kb_state = state.get("knowledge_base_state", {})
            
            # Determine response types needed
            response_types = self._determine_response_types(state)
            
            # Generate each type of response
            generated_responses = {}
            
            for response_type in response_types:
                if response_type == ResponseType.SELF_SERVICE:
                    response = self._generate_self_service_response(email_state, kb_state)
                elif response_type == ResponseType.ROUTING_NOTIFICATION:
                    response = self._generate_routing_notification(email_state, routing_state)
                elif response_type == ResponseType.ESCALATION_ALERT:
                    response = self._generate_escalation_alert(email_state, routing_state)
                elif response_type == ResponseType.ACKNOWLEDGMENT:
                    response = self._generate_acknowledgment_response(email_state, routing_state)
                else:
                    continue
                
                generated_responses[response_type.value] = response
            
            # Update workflow state
            updated_state = {
                **state,
                "generated_responses": generated_responses,
                "current_step": "workflow_complete",
                "completed_at": datetime.now(),
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Generated {len(generated_responses)} responses: {', '.join(generated_responses.keys())}",
                        name="response_generator"
                    )
                ]
            }
            
            return updated_state
            
        except Exception as e:
            # Handle response generation errors
            error_message = f"Response generation failed: {str(e)}"
            
            return {
                **state,
                "errors": state.get("errors", []) + [{
                    "component": "response_generator",
                    "error": error_message,
                    "timestamp": datetime.now()
                }],
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Response generation failed: {error_message}",
                        name="response_generator"
                    )
                ]
            }
    
    def _determine_response_types(self, state: WorkflowState) -> List[ResponseType]:
        """Determine which types of responses need to be generated."""
        response_types = []
        
        kb_state = state.get("knowledge_base_state", {})
        routing_state = state.get("routing_state", {})
        routing_decision = routing_state.get("routing_decision", {})
        
        # Always generate acknowledgment
        response_types.append(ResponseType.ACKNOWLEDGMENT)
        
        # Generate self-service response if articles are available
        if kb_state.get("self_service_possible") and kb_state.get("recommended_articles"):
            response_types.append(ResponseType.SELF_SERVICE)
        
        # Generate routing notification for assigned team
        if routing_decision.get("target_queue"):
            response_types.append(ResponseType.ROUTING_NOTIFICATION)
        
        # Generate escalation alert if escalation is required
        if routing_decision.get("escalation_required"):
            response_types.append(ResponseType.ESCALATION_ALERT)
        
        return response_types
    
    def _generate_self_service_response(
        self, 
        email_state: Dict[str, Any], 
        kb_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate self-service response with recommended articles."""
        try:
            template = self.response_templates[ResponseType.SELF_SERVICE]
            recommended_articles = kb_state.get("recommended_articles", [])
            
            # Format article recommendations
            article_text = self._format_article_recommendations(recommended_articles)
            
            # Generate personalized response using LLM
            llm_response = self._generate_llm_response(
                template, email_state, kb_state, {"article_recommendations": article_text}
            )
            
            return {
                "type": ResponseType.SELF_SERVICE.value,
                "subject": llm_response.get("subject", template.subject_template),
                "body": llm_response.get("body", template.body_template),
                "tone": llm_response.get("tone", template.tone),
                "call_to_action": llm_response.get("call_to_action"),
                "articles_included": [article["article_id"] for article in recommended_articles],
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            return self._generate_fallback_response(ResponseType.SELF_SERVICE, str(e))
    
    def _generate_routing_notification(
        self, 
        email_state: Dict[str, Any], 
        routing_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate routing notification for assigned team."""
        try:
            template = self.response_templates[ResponseType.ROUTING_NOTIFICATION]
            routing_decision = routing_state.get("routing_decision", {})
            intent_result = routing_state.get("intent_result", {})
            
            # Create email summary
            email_summary = self._create_email_summary(email_state)
            
            # Generate ticket ID
            ticket_id = self._generate_ticket_id()
            
            # Prepare template variables
            template_vars = {
                "email_summary": email_summary,
                "routing_reasoning": routing_decision.get("reasoning", ""),
                "ticket_id": ticket_id,
                "ticket_url": f"{self.organization_info['ticket_portal_url']}/ticket/{ticket_id}",
                "intent_category": intent_result.get("intent_category", {}).value if intent_result.get("intent_category") else "unknown",
                "priority": intent_result.get("priority", {}).value if intent_result.get("priority") else "medium",
                "target_queue": routing_decision.get("target_queue", {}).value if routing_decision.get("target_queue") else "general_support",
                "estimated_resolution": routing_decision.get("estimated_resolution_time", "TBD")
            }
            
            # Generate response using LLM
            llm_response = self._generate_llm_response(
                template, email_state, routing_state, template_vars
            )
            
            return {
                "type": ResponseType.ROUTING_NOTIFICATION.value,
                "subject": llm_response.get("subject", template.subject_template),
                "body": llm_response.get("body", template.body_template),
                "tone": llm_response.get("tone", template.tone),
                "ticket_id": ticket_id,
                "target_queue": template_vars["target_queue"],
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            return self._generate_fallback_response(ResponseType.ROUTING_NOTIFICATION, str(e))
    
    def _generate_escalation_alert(
        self, 
        email_state: Dict[str, Any], 
        routing_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate escalation alert for urgent issues."""
        try:
            template = self.response_templates[ResponseType.ESCALATION_ALERT]
            routing_decision = routing_state.get("routing_decision", {})
            
            # Create email summary
            email_summary = self._create_email_summary(email_state)
            
            # Generate ticket ID
            ticket_id = self._generate_ticket_id()
            
            # Determine escalation reason
            escalation_reason = routing_state.get("escalation_reason", "High priority issue requiring immediate attention")
            
            # Prepare template variables
            template_vars = {
                "email_summary": email_summary,
                "escalation_reason": escalation_reason,
                "ticket_id": ticket_id,
                "ticket_url": f"{self.organization_info['ticket_portal_url']}/ticket/{ticket_id}",
                "target_queue": routing_decision.get("target_queue", {}).value if routing_decision.get("target_queue") else "escalation",
                "estimated_resolution": routing_decision.get("estimated_resolution_time", "ASAP")
            }
            
            # Generate response using LLM
            llm_response = self._generate_llm_response(
                template, email_state, routing_state, template_vars
            )
            
            return {
                "type": ResponseType.ESCALATION_ALERT.value,
                "subject": llm_response.get("subject", template.subject_template),
                "body": llm_response.get("body", template.body_template),
                "tone": "urgent",
                "ticket_id": ticket_id,
                "escalation_reason": escalation_reason,
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            return self._generate_fallback_response(ResponseType.ESCALATION_ALERT, str(e))
    
    def _generate_acknowledgment_response(
        self, 
        email_state: Dict[str, Any], 
        routing_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate acknowledgment response for the user."""
        try:
            template = self.response_templates[ResponseType.ACKNOWLEDGMENT]
            routing_decision = routing_state.get("routing_decision", {})
            intent_result = routing_state.get("intent_result", {})
            
            # Generate ticket ID
            ticket_id = self._generate_ticket_id()
            
            # Determine next steps
            next_steps = self._determine_next_steps(routing_state)
            
            # Prepare template variables
            template_vars = {
                "ticket_id": ticket_id,
                "ticket_url": f"{self.organization_info['ticket_portal_url']}/ticket/{ticket_id}",
                "priority": intent_result.get("priority", {}).value if intent_result.get("priority") else "medium",
                "target_queue": routing_decision.get("target_queue", {}).value if routing_decision.get("target_queue") else "general_support",
                "estimated_resolution": routing_decision.get("estimated_resolution_time", "1-2 business days"),
                "next_steps": next_steps,
                "support_contact_info": f"{self.organization_info['support_email']} or {self.organization_info['support_phone']}"
            }
            
            # Generate response using LLM
            llm_response = self._generate_llm_response(
                template, email_state, routing_state, template_vars
            )
            
            return {
                "type": ResponseType.ACKNOWLEDGMENT.value,
                "subject": llm_response.get("subject", template.subject_template),
                "body": llm_response.get("body", template.body_template),
                "tone": llm_response.get("tone", template.tone),
                "ticket_id": ticket_id,
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            return self._generate_fallback_response(ResponseType.ACKNOWLEDGMENT, str(e))
    
    def _format_article_recommendations(self, articles: List[KnowledgeBaseArticle]) -> str:
        """Format knowledge base articles for inclusion in response."""
        if not articles:
            return "No specific articles found, but our support team will assist you."
        
        formatted_articles = []
        for i, article in enumerate(articles[:3], 1):  # Limit to top 3 articles
            formatted_articles.append(f"""
{i}. {article['title']}
   {article['content'][:200]}...
   Link: {article['url']}
   Confidence: {article['confidence_score']:.0%}
""")
        
        return "\n".join(formatted_articles)
    
    def _create_email_summary(self, email_state: Dict[str, Any]) -> str:
        """Create a concise summary of the email for internal notifications."""
        subject = email_state.get("subject", "")
        body = email_state.get("body", "")
        
        # Truncate body for summary
        body_summary = body[:300] + "..." if len(body) > 300 else body
        
        return f"Subject: {subject}\n\nContent:\n{body_summary}"
    
    def _generate_ticket_id(self) -> str:
        """Generate a unique ticket ID."""
        import uuid
        return f"TKT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    
    def _determine_next_steps(self, routing_state: Dict[str, Any]) -> str:
        """Determine next steps message based on routing decision."""
        routing_decision = routing_state.get("routing_decision", {})
        
        if routing_decision.get("escalation_required"):
            return "This request has been marked as high priority and will be reviewed immediately by our senior support team."
        
        target_queue = routing_decision.get("target_queue")
        if target_queue:
            queue_name = target_queue.value.replace("_", " ").title() if hasattr(target_queue, 'value') else str(target_queue)
            return f"Your request has been assigned to our {queue_name} team and will be addressed according to our service level agreement."
        
        return "Your request is being reviewed and will be assigned to the appropriate team shortly."
    
    def _generate_llm_response(
        self, 
        template: ResponseTemplate, 
        email_state: Dict[str, Any],
        context_state: Dict[str, Any],
        template_vars: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate personalized response using LLM."""
        try:
            system_prompt = f"""You are a professional customer service response generator. 
Create a {template.tone} response email based on the provided template and context.

TEMPLATE TYPE: {template.response_type.value}
TONE: {template.tone}

INSTRUCTIONS:
1. Personalize the response based on the email context
2. Maintain a {template.tone} tone throughout
3. Include all necessary information from the template variables
4. Ensure the response is clear, helpful, and professional
5. Keep the subject line concise and informative

Respond with structured output including subject, body, tone, and call_to_action."""
            
            # Prepare context
            context = f"""
Original Email:
Subject: {email_state.get('subject', '')}
From: {email_state.get('sender_email', '')}
Sender Name: {email_state.get('sender_name', 'Valued Customer')}

Template Variables:
{template_vars}

Organization: {self.organization_info['name']}
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            # Get structured output from LLM
            llm_with_parser = self.llm.with_structured_output(ResponseOutput)
            result = llm_with_parser.invoke(messages)
            
            return {
                "subject": result.subject,
                "body": result.body,
                "tone": result.tone,
                "call_to_action": result.call_to_action,
                "estimated_resolution": result.estimated_resolution
            }
            
        except Exception as e:
            print(f"LLM response generation failed: {e}")
            # Fallback to template-based response
            return self._generate_template_response(template, email_state, template_vars)
    
    def _generate_template_response(
        self, 
        template: ResponseTemplate, 
        email_state: Dict[str, Any],
        template_vars: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response using template substitution as fallback."""
        # Prepare all template variables
        all_vars = {
            "sender_name": email_state.get("sender_name", "Valued Customer"),
            "sender_email": email_state.get("sender_email", ""),
            "original_subject": email_state.get("subject", ""),
            "organization_name": self.organization_info["name"],
            "support_contact_info": f"{self.organization_info['support_email']} or {self.organization_info['support_phone']}",
            **template_vars
        }
        
        # Substitute variables in templates
        try:
            subject = template.subject_template.format(**all_vars)
            body = template.body_template.format(**all_vars)
        except KeyError as e:
            # Handle missing variables
            subject = template.subject_template
            body = template.body_template
            print(f"Template variable missing: {e}")
        
        return {
            "subject": subject,
            "body": body,
            "tone": template.tone,
            "call_to_action": None
        }
    
    def _generate_fallback_response(self, response_type: ResponseType, error: str) -> Dict[str, Any]:
        """Generate a basic fallback response when generation fails."""
        return {
            "type": response_type.value,
            "subject": f"Support Request - {response_type.value.replace('_', ' ').title()}",
            "body": f"Thank you for your support request. We are processing your inquiry and will respond shortly.\n\nError: {error}",
            "tone": "professional",
            "error": error,
            "generated_at": datetime.now()
        }


# Tool function for LangGraph integration
@tool
def generate_response_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool function for generating responses in LangGraph workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated workflow state with generated responses
    """
    generator = ResponseGenerator()
    return generator.generate_response(state)


# Export the main class and tool
__all__ = ["ResponseGenerator", "ResponseType", "generate_response_tool"]
