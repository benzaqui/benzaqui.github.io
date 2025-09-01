"""
Router Agent

This module handles determining the appropriate team/queue routing for support emails
based on intent classification, priority, and business rules.
"""

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
    RoutingQueue,
    RoutingDecision,
    validate_routing_decision
)


class RoutingDecisionOutput(BaseModel):
    """Structured output for routing decisions."""
    target_queue: str = Field(description="The target routing queue for the email")
    assigned_agent: Optional[str] = Field(description="Specific agent if available")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Detailed explanation of the routing decision")
    escalation_required: bool = Field(description="Whether escalation is required")
    estimated_resolution_time: Optional[str] = Field(description="Estimated time to resolve")


@dataclass
class RoutingRule:
    """Routing rule configuration."""
    intent_category: IntentCategory
    priority: Priority
    target_queue: RoutingQueue
    escalation_required: bool = False
    conditions: Optional[Dict[str, Any]] = None
    estimated_resolution_time: Optional[str] = None


class Router:
    """
    Specialized agent for determining appropriate routing decisions.
    
    This agent handles:
    - Applying business rules for routing based on intent and priority
    - Considering workload balancing and agent availability
    - Determining escalation requirements
    - Providing confidence scores and reasoning for decisions
    - Supporting custom routing rules and overrides
    """
    
    def __init__(self, model_name: str = "anthropic:claude-3-5-sonnet-20241022"):
        """Initialize the router with specified LLM model."""
        self.model_name = model_name
        self.llm = self._initialize_model()
        self.output_parser = PydanticOutputParser(pydantic_object=RoutingDecisionOutput)
        
        # Define routing rules matrix
        self.routing_rules = self._initialize_routing_rules()
        
        # Queue capacity and SLA information
        self.queue_info = {
            RoutingQueue.IT_TIER1: {
                "capacity": "high",
                "sla_hours": 4,
                "specialties": ["password_reset", "account_access", "basic_software"]
            },
            RoutingQueue.IT_TIER2: {
                "capacity": "medium", 
                "sla_hours": 8,
                "specialties": ["complex_software", "system_issues", "network_problems"]
            },
            RoutingQueue.HR_GENERAL: {
                "capacity": "medium",
                "sla_hours": 24,
                "specialties": ["policies", "benefits", "leave_requests"]
            },
            RoutingQueue.FACILITIES_MAINTENANCE: {
                "capacity": "low",
                "sla_hours": 48,
                "specialties": ["building_issues", "maintenance", "security"]
            },
            RoutingQueue.BILLING_SUPPORT: {
                "capacity": "medium",
                "sla_hours": 12,
                "specialties": ["invoices", "payments", "account_billing"]
            },
            RoutingQueue.GENERAL_SUPPORT: {
                "capacity": "high",
                "sla_hours": 24,
                "specialties": ["general_inquiries", "information_requests"]
            },
            RoutingQueue.ESCALATION: {
                "capacity": "low",
                "sla_hours": 2,
                "specialties": ["urgent_issues", "management_requests", "critical_problems"]
            }
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
    
    def _initialize_routing_rules(self) -> List[RoutingRule]:
        """Initialize the routing rules matrix."""
        rules = []
        
        # Password Reset Rules
        rules.extend([
            RoutingRule(IntentCategory.PASSWORD_RESET, Priority.URGENT, RoutingQueue.IT_TIER1, 
                       estimated_resolution_time="15 minutes"),
            RoutingRule(IntentCategory.PASSWORD_RESET, Priority.HIGH, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="30 minutes"),
            RoutingRule(IntentCategory.PASSWORD_RESET, Priority.MEDIUM, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="2 hours"),
            RoutingRule(IntentCategory.PASSWORD_RESET, Priority.LOW, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="4 hours")
        ])
        
        # Software Issue Rules
        rules.extend([
            RoutingRule(IntentCategory.SOFTWARE_ISSUE, Priority.URGENT, RoutingQueue.IT_TIER2,
                       escalation_required=True, estimated_resolution_time="1 hour"),
            RoutingRule(IntentCategory.SOFTWARE_ISSUE, Priority.HIGH, RoutingQueue.IT_TIER2,
                       estimated_resolution_time="4 hours"),
            RoutingRule(IntentCategory.SOFTWARE_ISSUE, Priority.MEDIUM, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="8 hours"),
            RoutingRule(IntentCategory.SOFTWARE_ISSUE, Priority.LOW, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="24 hours")
        ])
        
        # Hardware Request Rules
        rules.extend([
            RoutingRule(IntentCategory.HARDWARE_REQUEST, Priority.URGENT, RoutingQueue.IT_TIER2,
                       escalation_required=True, estimated_resolution_time="2 hours"),
            RoutingRule(IntentCategory.HARDWARE_REQUEST, Priority.HIGH, RoutingQueue.IT_TIER2,
                       estimated_resolution_time="1 day"),
            RoutingRule(IntentCategory.HARDWARE_REQUEST, Priority.MEDIUM, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="3 days"),
            RoutingRule(IntentCategory.HARDWARE_REQUEST, Priority.LOW, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="5 days")
        ])
        
        # Technical Issue Rules
        rules.extend([
            RoutingRule(IntentCategory.TECHNICAL_ISSUE, Priority.URGENT, RoutingQueue.ESCALATION,
                       escalation_required=True, estimated_resolution_time="30 minutes"),
            RoutingRule(IntentCategory.TECHNICAL_ISSUE, Priority.HIGH, RoutingQueue.IT_TIER2,
                       estimated_resolution_time="2 hours"),
            RoutingRule(IntentCategory.TECHNICAL_ISSUE, Priority.MEDIUM, RoutingQueue.IT_TIER2,
                       estimated_resolution_time="8 hours"),
            RoutingRule(IntentCategory.TECHNICAL_ISSUE, Priority.LOW, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="24 hours")
        ])
        
        # IT Support Rules
        rules.extend([
            RoutingRule(IntentCategory.IT_SUPPORT, Priority.URGENT, RoutingQueue.IT_TIER2,
                       escalation_required=True, estimated_resolution_time="1 hour"),
            RoutingRule(IntentCategory.IT_SUPPORT, Priority.HIGH, RoutingQueue.IT_TIER2,
                       estimated_resolution_time="4 hours"),
            RoutingRule(IntentCategory.IT_SUPPORT, Priority.MEDIUM, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="8 hours"),
            RoutingRule(IntentCategory.IT_SUPPORT, Priority.LOW, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="24 hours")
        ])
        
        # Account Access Rules
        rules.extend([
            RoutingRule(IntentCategory.ACCOUNT_ACCESS, Priority.URGENT, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="30 minutes"),
            RoutingRule(IntentCategory.ACCOUNT_ACCESS, Priority.HIGH, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="1 hour"),
            RoutingRule(IntentCategory.ACCOUNT_ACCESS, Priority.MEDIUM, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="4 hours"),
            RoutingRule(IntentCategory.ACCOUNT_ACCESS, Priority.LOW, RoutingQueue.IT_TIER1,
                       estimated_resolution_time="8 hours")
        ])
        
        # HR Inquiry Rules
        rules.extend([
            RoutingRule(IntentCategory.HR_INQUIRY, Priority.URGENT, RoutingQueue.HR_GENERAL,
                       escalation_required=True, estimated_resolution_time="2 hours"),
            RoutingRule(IntentCategory.HR_INQUIRY, Priority.HIGH, RoutingQueue.HR_GENERAL,
                       estimated_resolution_time="4 hours"),
            RoutingRule(IntentCategory.HR_INQUIRY, Priority.MEDIUM, RoutingQueue.HR_GENERAL,
                       estimated_resolution_time="1 day"),
            RoutingRule(IntentCategory.HR_INQUIRY, Priority.LOW, RoutingQueue.HR_GENERAL,
                       estimated_resolution_time="2 days")
        ])
        
        # Facilities Rules
        rules.extend([
            RoutingRule(IntentCategory.FACILITIES, Priority.URGENT, RoutingQueue.FACILITIES_MAINTENANCE,
                       escalation_required=True, estimated_resolution_time="1 hour"),
            RoutingRule(IntentCategory.FACILITIES, Priority.HIGH, RoutingQueue.FACILITIES_MAINTENANCE,
                       estimated_resolution_time="4 hours"),
            RoutingRule(IntentCategory.FACILITIES, Priority.MEDIUM, RoutingQueue.FACILITIES_MAINTENANCE,
                       estimated_resolution_time="1 day"),
            RoutingRule(IntentCategory.FACILITIES, Priority.LOW, RoutingQueue.FACILITIES_MAINTENANCE,
                       estimated_resolution_time="3 days")
        ])
        
        # Billing Rules
        rules.extend([
            RoutingRule(IntentCategory.BILLING, Priority.URGENT, RoutingQueue.BILLING_SUPPORT,
                       escalation_required=True, estimated_resolution_time="2 hours"),
            RoutingRule(IntentCategory.BILLING, Priority.HIGH, RoutingQueue.BILLING_SUPPORT,
                       estimated_resolution_time="4 hours"),
            RoutingRule(IntentCategory.BILLING, Priority.MEDIUM, RoutingQueue.BILLING_SUPPORT,
                       estimated_resolution_time="1 day"),
            RoutingRule(IntentCategory.BILLING, Priority.LOW, RoutingQueue.BILLING_SUPPORT,
                       estimated_resolution_time="2 days")
        ])
        
        # General Inquiry Rules
        rules.extend([
            RoutingRule(IntentCategory.GENERAL_INQUIRY, Priority.URGENT, RoutingQueue.GENERAL_SUPPORT,
                       estimated_resolution_time="1 hour"),
            RoutingRule(IntentCategory.GENERAL_INQUIRY, Priority.HIGH, RoutingQueue.GENERAL_SUPPORT,
                       estimated_resolution_time="4 hours"),
            RoutingRule(IntentCategory.GENERAL_INQUIRY, Priority.MEDIUM, RoutingQueue.GENERAL_SUPPORT,
                       estimated_resolution_time="1 day"),
            RoutingRule(IntentCategory.GENERAL_INQUIRY, Priority.LOW, RoutingQueue.GENERAL_SUPPORT,
                       estimated_resolution_time="2 days")
        ])
        
        # Other/Unknown Rules
        rules.extend([
            RoutingRule(IntentCategory.OTHER, Priority.URGENT, RoutingQueue.ESCALATION,
                       escalation_required=True, estimated_resolution_time="1 hour"),
            RoutingRule(IntentCategory.OTHER, Priority.HIGH, RoutingQueue.GENERAL_SUPPORT,
                       estimated_resolution_time="4 hours"),
            RoutingRule(IntentCategory.OTHER, Priority.MEDIUM, RoutingQueue.GENERAL_SUPPORT,
                       estimated_resolution_time="1 day"),
            RoutingRule(IntentCategory.OTHER, Priority.LOW, RoutingQueue.GENERAL_SUPPORT,
                       estimated_resolution_time="2 days")
        ])
        
        return rules
    
    def determine_routing(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Main routing function for determining appropriate queue and agent assignment.
        
        Args:
            state: Current workflow state containing email and intent data
            
        Returns:
            Updated state with routing decision
        """
        try:
            # Extract required data
            email_state = state.get("email_state", {})
            routing_state = state.get("routing_state", {})
            kb_state = state.get("knowledge_base_state", {})
            intent_result = routing_state.get("intent_result")
            
            if not intent_result:
                raise ValueError("Missing intent classification results")
            
            # Apply rule-based routing
            rule_based_decision = self._apply_routing_rules(intent_result, email_state)
            
            # Consider knowledge base results
            kb_adjusted_decision = self._adjust_for_knowledge_base(rule_based_decision, kb_state)
            
            # Apply LLM-based routing validation
            llm_validated_decision = self._validate_with_llm(
                kb_adjusted_decision, email_state, intent_result, kb_state
            )
            
            # Consider workload and capacity
            final_decision = self._adjust_for_capacity(llm_validated_decision)
            
            # Create RoutingDecision
            routing_decision = RoutingDecision(
                target_queue=final_decision["target_queue"],
                assigned_agent=final_decision.get("assigned_agent"),
                confidence_score=final_decision["confidence_score"],
                reasoning=final_decision["reasoning"],
                escalation_required=final_decision["escalation_required"],
                estimated_resolution_time=final_decision.get("estimated_resolution_time")
            )
            
            # Validate the routing decision
            validation_errors = validate_routing_decision(routing_decision)
            if validation_errors:
                raise ValueError(f"Routing decision validation failed: {validation_errors}")
            
            # Update routing state
            updated_routing_state = routing_state.copy()
            updated_routing_state.update({
                "routing_decision": routing_decision,
                "routing_timestamp": datetime.now(),
                "should_escalate": routing_decision["escalation_required"]
            })
            
            # Determine next step
            next_step = "response_generation"
            if kb_state.get("self_service_possible") and not routing_decision["escalation_required"]:
                next_step = "human_review"  # Show self-service options first
            
            # Update workflow state
            updated_state = {
                **state,
                "routing_state": updated_routing_state,
                "current_step": next_step,
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Routing decision: {routing_decision['target_queue'].value} "
                               f"(confidence: {routing_decision['confidence_score']:.2f}). "
                               f"Escalation required: {routing_decision['escalation_required']}. "
                               f"ETA: {routing_decision['estimated_resolution_time']}",
                        name="router"
                    )
                ]
            }
            
            return updated_state
            
        except Exception as e:
            # Handle routing errors
            error_message = f"Routing decision failed: {str(e)}"
            
            # Create fallback routing decision
            fallback_decision = RoutingDecision(
                target_queue=RoutingQueue.GENERAL_SUPPORT,
                assigned_agent=None,
                confidence_score=0.3,
                reasoning=f"Fallback routing due to error: {error_message}",
                escalation_required=True,
                estimated_resolution_time="4 hours"
            )
            
            updated_routing_state = routing_state.copy()
            updated_routing_state.update({
                "routing_decision": fallback_decision,
                "routing_timestamp": datetime.now(),
                "should_escalate": True,
                "escalation_reason": error_message
            })
            
            return {
                **state,
                "routing_state": updated_routing_state,
                "current_step": "response_generation",
                "errors": state.get("errors", []) + [{
                    "component": "router",
                    "error": error_message,
                    "timestamp": datetime.now()
                }],
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Routing failed: {error_message}. Using fallback routing to General Support.",
                        name="router"
                    )
                ]
            }
    
    def _apply_routing_rules(
        self, 
        intent_result: Dict[str, Any], 
        email_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply rule-based routing logic."""
        intent_category = intent_result.get("intent_category")
        priority = intent_result.get("priority")
        
        # Find matching rule
        matching_rule = None
        for rule in self.routing_rules:
            if rule.intent_category == intent_category and rule.priority == priority:
                matching_rule = rule
                break
        
        if not matching_rule:
            # Fallback rule
            matching_rule = RoutingRule(
                intent_category=IntentCategory.OTHER,
                priority=priority,
                target_queue=RoutingQueue.GENERAL_SUPPORT,
                estimated_resolution_time="1 day"
            )
        
        return {
            "target_queue": matching_rule.target_queue,
            "escalation_required": matching_rule.escalation_required,
            "estimated_resolution_time": matching_rule.estimated_resolution_time,
            "confidence_score": 0.8,  # High confidence for rule-based decisions
            "reasoning": f"Rule-based routing: {intent_category.value if intent_category else 'unknown'} "
                        f"with {priority.value if priority else 'unknown'} priority"
        }
    
    def _adjust_for_knowledge_base(
        self, 
        routing_decision: Dict[str, Any], 
        kb_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust routing decision based on knowledge base results."""
        if not kb_state:
            return routing_decision
        
        # If self-service is possible, reduce urgency
        if kb_state.get("self_service_possible"):
            # Don't escalate if self-service is available
            routing_decision["escalation_required"] = False
            
            # Adjust confidence based on KB confidence
            kb_confidence = kb_state.get("search_confidence", 0.0)
            if kb_confidence > 0.8:
                routing_decision["confidence_score"] = min(
                    routing_decision["confidence_score"] + 0.1, 1.0
                )
            
            routing_decision["reasoning"] += f". Self-service articles available (confidence: {kb_confidence:.2f})"
        
        return routing_decision
    
    def _validate_with_llm(
        self, 
        routing_decision: Dict[str, Any], 
        email_state: Dict[str, Any],
        intent_result: Dict[str, Any],
        kb_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and potentially adjust routing decision using LLM."""
        try:
            system_prompt = self._create_routing_validation_prompt()
            
            # Prepare context
            context = f"""
Email Subject: {email_state.get('subject', '')}
Email Body: {email_state.get('body', '')[:500]}...
Intent: {intent_result.get('intent_category', {}).value if intent_result.get('intent_category') else 'unknown'}
Priority: {intent_result.get('priority', {}).value if intent_result.get('priority') else 'unknown'}
Current Routing: {routing_decision['target_queue'].value}
Escalation Required: {routing_decision['escalation_required']}
Self-Service Available: {kb_state.get('self_service_possible', False)}
KB Confidence: {kb_state.get('search_confidence', 0.0)}
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            # Get structured output from LLM
            llm_with_parser = self.llm.with_structured_output(RoutingDecisionOutput)
            result = llm_with_parser.invoke(messages)
            
            # Convert string to enum
            target_queue = RoutingQueue(result.target_queue.lower().replace(" ", "_"))
            
            # Combine LLM validation with rule-based decision
            if result.confidence_score > 0.7:
                # Use LLM decision if confident
                return {
                    "target_queue": target_queue,
                    "assigned_agent": result.assigned_agent,
                    "confidence_score": result.confidence_score,
                    "reasoning": f"LLM-validated routing: {result.reasoning}",
                    "escalation_required": result.escalation_required,
                    "estimated_resolution_time": result.estimated_resolution_time or routing_decision.get("estimated_resolution_time")
                }
            else:
                # Keep rule-based decision but add LLM insights
                routing_decision["reasoning"] += f". LLM validation: {result.reasoning}"
                return routing_decision
                
        except Exception as e:
            print(f"LLM routing validation failed: {e}")
            return routing_decision
    
    def _create_routing_validation_prompt(self) -> str:
        """Create system prompt for routing validation."""
        queue_options = [queue.value for queue in RoutingQueue]
        
        return f"""You are an expert helpdesk routing specialist. Your task is to validate and potentially adjust routing decisions for support emails.

AVAILABLE QUEUES:
{', '.join(queue_options)}

ROUTING PRINCIPLES:
1. Urgent issues should be escalated or routed to appropriate tier-2 support
2. Self-service opportunities should reduce escalation needs
3. Match technical complexity to appropriate support tier
4. Consider business impact and user urgency
5. Balance workload across teams

INSTRUCTIONS:
1. Review the email content, intent, and current routing decision
2. Validate if the routing is appropriate
3. Suggest adjustments if needed
4. Provide confidence score (0.0-1.0)
5. Explain your reasoning
6. Determine if escalation is truly required

Respond with structured output including target_queue, assigned_agent (if specific), confidence_score, reasoning, escalation_required, and estimated_resolution_time."""
    
    def _adjust_for_capacity(self, routing_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust routing decision based on queue capacity and workload."""
        target_queue = routing_decision["target_queue"]
        queue_capacity = self.queue_info.get(target_queue, {}).get("capacity", "medium")
        
        # If target queue is at low capacity and not urgent, consider alternatives
        if queue_capacity == "low" and not routing_decision["escalation_required"]:
            # Look for alternative queues with higher capacity
            alternative_queues = [
                queue for queue, info in self.queue_info.items()
                if info.get("capacity") in ["high", "medium"] and queue != target_queue
            ]
            
            if alternative_queues:
                # For now, keep original routing but note capacity concern
                routing_decision["reasoning"] += f". Note: {target_queue.value} has low capacity"
        
        return routing_decision


# Tool function for LangGraph integration
@tool
def determine_routing_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool function for determining routing decisions in LangGraph workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated workflow state with routing decision
    """
    router = Router()
    return router.determine_routing(state)


# Export the main class and tool
__all__ = ["Router", "RoutingRule", "determine_routing_tool"]
