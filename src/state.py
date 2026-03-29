"""
Core state management for the helpdesk email routing agent.

This module defines TypedDict classes for managing email data, routing decisions,
intent classification results, and knowledge base query results throughout the workflow.
"""

from typing import Annotated, List, Dict, Any, Optional, Union
from typing_extensions import TypedDict
from datetime import datetime
from enum import Enum

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Priority(Enum):
    """Email priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class IntentCategory(Enum):
    """Email intent categories for routing."""
    IT_SUPPORT = "it_support"
    HR_INQUIRY = "hr_inquiry"
    FACILITIES = "facilities"
    GENERAL_INQUIRY = "general_inquiry"
    BILLING = "billing"
    TECHNICAL_ISSUE = "technical_issue"
    PASSWORD_RESET = "password_reset"
    HARDWARE_REQUEST = "hardware_request"
    SOFTWARE_ISSUE = "software_issue"
    ACCOUNT_ACCESS = "account_access"
    OTHER = "other"


class RoutingQueue(Enum):
    """Available routing queues."""
    IT_TIER1 = "it_tier1"
    IT_TIER2 = "it_tier2"
    HR_GENERAL = "hr_general"
    FACILITIES_MAINTENANCE = "facilities_maintenance"
    BILLING_SUPPORT = "billing_support"
    GENERAL_SUPPORT = "general_support"
    ESCALATION = "escalation"


class EmailState(TypedDict):
    """State for managing raw email data and metadata."""
    # Core email data
    subject: str
    body: str
    sender_email: str
    sender_name: Optional[str]
    recipient_email: str
    timestamp: datetime
    
    # Extracted metadata
    attachments: List[Dict[str, Any]]
    email_id: str
    thread_id: Optional[str]
    
    # Processing flags
    is_processed: bool
    processing_errors: List[str]


class IntentClassificationResult(TypedDict):
    """Result from intent classification analysis."""
    intent_category: IntentCategory
    confidence_score: float
    priority: Priority
    urgency_indicators: List[str]
    keywords_found: List[str]
    reasoning: str


class KnowledgeBaseArticle(TypedDict):
    """Knowledge base article from Kendra search."""
    article_id: str
    title: str
    content: str
    url: str
    confidence_score: float
    source: str  # e.g., "ServiceNow", "Confluence", etc.
    tags: List[str]
    last_updated: Optional[datetime]


class RoutingDecision(TypedDict):
    """Final routing decision with justification."""
    target_queue: RoutingQueue
    assigned_agent: Optional[str]
    confidence_score: float
    reasoning: str
    escalation_required: bool
    estimated_resolution_time: Optional[str]


class HumanFeedback(TypedDict):
    """Human feedback on self-service articles or routing decisions."""
    feedback_type: str  # "article_helpful", "article_not_helpful", "routing_override"
    is_helpful: bool
    comments: Optional[str]
    preferred_queue: Optional[RoutingQueue]
    timestamp: datetime


class RoutingState(TypedDict):
    """State for managing routing decisions and intent classification."""
    # Intent classification results
    intent_result: Optional[IntentClassificationResult]
    
    # Routing decision
    routing_decision: Optional[RoutingDecision]
    
    # Human feedback
    human_feedback: Optional[HumanFeedback]
    
    # Processing metadata
    classification_timestamp: Optional[datetime]
    routing_timestamp: Optional[datetime]
    requires_human_review: bool
    
    # Workflow control
    should_escalate: bool
    escalation_reason: Optional[str]


class KnowledgeBaseState(TypedDict):
    """State for managing knowledge base query results."""
    # Query information
    query_text: str
    query_timestamp: Optional[datetime]
    
    # Search results
    articles_found: List[KnowledgeBaseArticle]
    total_results: int
    search_confidence: float
    
    # Article recommendations
    recommended_articles: List[KnowledgeBaseArticle]
    self_service_possible: bool
    
    # User interaction
    articles_presented: List[str]  # Article IDs shown to user
    user_selected_article: Optional[str]
    
    # Processing metadata
    kendra_query_id: Optional[str]
    search_errors: List[str]


class WorkflowState(TypedDict):
    """Main workflow state combining all sub-states."""
    # LangGraph messages for agent communication
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Sub-states for different workflow stages
    email_state: EmailState
    routing_state: RoutingState
    knowledge_base_state: KnowledgeBaseState
    
    # Workflow control
    current_step: str
    workflow_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    
    # Error handling
    errors: List[Dict[str, Any]]
    retry_count: int
    max_retries: int
    
    # Human-in-the-loop control
    awaiting_human_input: bool
    human_input_type: Optional[str]  # "article_feedback", "routing_confirmation", etc.
    interrupt_data: Optional[Dict[str, Any]]


# State update helper functions
def create_empty_email_state() -> EmailState:
    """Create an empty EmailState with default values."""
    return EmailState(
        subject="",
        body="",
        sender_email="",
        sender_name=None,
        recipient_email="",
        timestamp=datetime.now(),
        attachments=[],
        email_id="",
        thread_id=None,
        is_processed=False,
        processing_errors=[]
    )


def create_empty_routing_state() -> RoutingState:
    """Create an empty RoutingState with default values."""
    return RoutingState(
        intent_result=None,
        routing_decision=None,
        human_feedback=None,
        classification_timestamp=None,
        routing_timestamp=None,
        requires_human_review=False,
        should_escalate=False,
        escalation_reason=None
    )


def create_empty_knowledge_base_state() -> KnowledgeBaseState:
    """Create an empty KnowledgeBaseState with default values."""
    return KnowledgeBaseState(
        query_text="",
        query_timestamp=None,
        articles_found=[],
        total_results=0,
        search_confidence=0.0,
        recommended_articles=[],
        self_service_possible=False,
        articles_presented=[],
        user_selected_article=None,
        kendra_query_id=None,
        search_errors=[]
    )


def create_initial_workflow_state(
    email_data: Dict[str, Any],
    workflow_id: str
) -> WorkflowState:
    """Create initial WorkflowState from email data."""
    email_state = create_empty_email_state()
    email_state.update(email_data)
    
    return WorkflowState(
        messages=[],
        email_state=email_state,
        routing_state=create_empty_routing_state(),
        knowledge_base_state=create_empty_knowledge_base_state(),
        current_step="email_processing",
        workflow_id=workflow_id,
        started_at=datetime.now(),
        completed_at=None,
        errors=[],
        retry_count=0,
        max_retries=3,
        awaiting_human_input=False,
        human_input_type=None,
        interrupt_data=None
    )


# State validation functions
def validate_email_state(state: EmailState) -> List[str]:
    """Validate EmailState and return list of validation errors."""
    errors = []
    
    if not state.get("subject"):
        errors.append("Email subject is required")
    
    if not state.get("body"):
        errors.append("Email body is required")
    
    if not state.get("sender_email"):
        errors.append("Sender email is required")
    
    if not state.get("email_id"):
        errors.append("Email ID is required")
    
    return errors


def validate_intent_classification(result: IntentClassificationResult) -> List[str]:
    """Validate IntentClassificationResult and return list of validation errors."""
    errors = []
    
    if result["confidence_score"] < 0.0 or result["confidence_score"] > 1.0:
        errors.append("Confidence score must be between 0.0 and 1.0")
    
    if not result.get("reasoning"):
        errors.append("Classification reasoning is required")
    
    return errors


def validate_routing_decision(decision: RoutingDecision) -> List[str]:
    """Validate RoutingDecision and return list of validation errors."""
    errors = []
    
    if decision["confidence_score"] < 0.0 or decision["confidence_score"] > 1.0:
        errors.append("Confidence score must be between 0.0 and 1.0")
    
    if not decision.get("reasoning"):
        errors.append("Routing reasoning is required")
    
    return errors


# State reducer functions for concurrent updates
def merge_errors(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge error lists from concurrent updates."""
    return left + right


def merge_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Merge message lists from concurrent updates."""
    return left + right


# Export all state classes and utilities
__all__ = [
    # Enums
    "Priority",
    "IntentCategory", 
    "RoutingQueue",
    
    # State classes
    "EmailState",
    "IntentClassificationResult",
    "KnowledgeBaseArticle",
    "RoutingDecision",
    "HumanFeedback",
    "RoutingState",
    "KnowledgeBaseState",
    "WorkflowState",
    
    # Helper functions
    "create_empty_email_state",
    "create_empty_routing_state",
    "create_empty_knowledge_base_state",
    "create_initial_workflow_state",
    
    # Validation functions
    "validate_email_state",
    "validate_intent_classification",
    "validate_routing_decision",
    
    # Reducer functions
    "merge_errors",
    "merge_messages",
]
