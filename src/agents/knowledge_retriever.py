"""
Knowledge Retriever Agent

This module handles querying the Kendra knowledge base with ServiceNow (SNOW) articles
to find relevant self-service solutions for incoming support emails.
"""

import boto3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from botocore.exceptions import ClientError, BotoCoreError

from ..state import (
    WorkflowState, 
    KnowledgeBaseState,
    KnowledgeBaseArticle,
    IntentCategory,
    Priority
)


class KnowledgeRetriever:
    """
    Specialized agent for querying Kendra knowledge base and retrieving relevant articles.
    
    This agent handles:
    - Constructing optimized search queries based on email content and intent
    - Querying AWS Kendra knowledge base with SNOW articles
    - Filtering and ranking search results by relevance
    - Determining self-service potential based on article quality
    - Extracting and structuring article metadata
    """
    
    def __init__(
        self, 
        kendra_index_id: Optional[str] = None,
        aws_region: str = "us-east-1",
        model_name: str = "anthropic:claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.7
    ):
        """Initialize the knowledge retriever with AWS Kendra configuration."""
        self.kendra_index_id = kendra_index_id
        self.aws_region = aws_region
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Initialize LLM for query optimization
        self.llm = self._initialize_model()
        
        # Initialize Kendra client
        self.kendra_client = self._initialize_kendra_client()
        
        # Query templates for different intent categories
        self.query_templates = {
            IntentCategory.PASSWORD_RESET: [
                "password reset instructions",
                "forgot password recovery",
                "account unlock procedure",
                "login authentication help"
            ],
            IntentCategory.SOFTWARE_ISSUE: [
                "software troubleshooting guide",
                "application error resolution",
                "program installation help",
                "software update instructions"
            ],
            IntentCategory.HARDWARE_REQUEST: [
                "hardware request process",
                "equipment ordering procedure",
                "device replacement guide",
                "hardware procurement policy"
            ],
            IntentCategory.IT_SUPPORT: [
                "IT support procedures",
                "technical assistance guide",
                "system configuration help",
                "network troubleshooting"
            ],
            IntentCategory.HR_INQUIRY: [
                "HR policies and procedures",
                "employee handbook",
                "benefits information",
                "leave request process"
            ],
            IntentCategory.FACILITIES: [
                "facilities management",
                "building maintenance request",
                "office space procedures",
                "security access guide"
            ],
            IntentCategory.BILLING: [
                "billing procedures",
                "invoice processing",
                "payment methods",
                "account management"
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
    
    def _initialize_kendra_client(self):
        """Initialize AWS Kendra client."""
        try:
            return boto3.client('kendra', region_name=self.aws_region)
        except Exception as e:
            print(f"Warning: Failed to initialize Kendra client: {e}")
            return None
    
    def retrieve_knowledge(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Main retrieval function for finding relevant knowledge base articles.
        
        Args:
            state: Current workflow state containing email and intent data
            
        Returns:
            Updated state with knowledge base search results
        """
        try:
            # Extract email and intent data
            email_state = state.get("email_state", {})
            routing_state = state.get("routing_state", {})
            intent_result = routing_state.get("intent_result")
            
            if not email_state or not intent_result:
                raise ValueError("Missing email data or intent classification")
            
            # Generate optimized search queries
            search_queries = self._generate_search_queries(email_state, intent_result)
            
            # Search knowledge base
            search_results = self._search_knowledge_base(search_queries)
            
            # Process and rank results
            processed_articles = self._process_search_results(search_results, intent_result)
            
            # Determine self-service potential
            self_service_analysis = self._analyze_self_service_potential(
                processed_articles, intent_result
            )
            
            # Create KnowledgeBaseState
            kb_state = KnowledgeBaseState(
                query_text=" | ".join(search_queries),
                query_timestamp=datetime.now(),
                articles_found=processed_articles,
                total_results=len(processed_articles),
                search_confidence=self_service_analysis["confidence"],
                recommended_articles=self_service_analysis["recommended_articles"],
                self_service_possible=self_service_analysis["self_service_possible"],
                articles_presented=[],
                user_selected_article=None,
                kendra_query_id=search_results.get("QueryId"),
                search_errors=[]
            )
            
            # Update workflow state
            updated_state = {
                **state,
                "knowledge_base_state": kb_state,
                "current_step": "routing_decision" if not kb_state["self_service_possible"] else "human_review",
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Found {len(processed_articles)} relevant articles. "
                               f"Self-service possible: {kb_state['self_service_possible']}. "
                               f"Top recommendation: {processed_articles[0]['title'] if processed_articles else 'None'}",
                        name="knowledge_retriever"
                    )
                ]
            }
            
            return updated_state
            
        except Exception as e:
            # Handle retrieval errors
            error_message = f"Knowledge retrieval failed: {str(e)}"
            
            # Create error state
            error_kb_state = KnowledgeBaseState(
                query_text="",
                query_timestamp=datetime.now(),
                articles_found=[],
                total_results=0,
                search_confidence=0.0,
                recommended_articles=[],
                self_service_possible=False,
                articles_presented=[],
                user_selected_article=None,
                kendra_query_id=None,
                search_errors=[error_message]
            )
            
            return {
                **state,
                "knowledge_base_state": error_kb_state,
                "current_step": "routing_decision",  # Skip to routing if KB fails
                "errors": state.get("errors", []) + [{
                    "component": "knowledge_retriever",
                    "error": error_message,
                    "timestamp": datetime.now()
                }],
                "messages": state["messages"] + [
                    AIMessage(
                        content=f"Knowledge retrieval failed: {error_message}. Proceeding to direct routing.",
                        name="knowledge_retriever"
                    )
                ]
            }
    
    def _generate_search_queries(
        self, 
        email_state: Dict[str, Any], 
        intent_result: Dict[str, Any]
    ) -> List[str]:
        """Generate optimized search queries based on email content and intent."""
        subject = email_state.get("subject", "")
        body = email_state.get("body", "")
        intent_category = intent_result.get("intent_category")
        keywords = intent_result.get("keywords_found", [])
        
        queries = []
        
        # Add template-based queries for the intent category
        if intent_category in self.query_templates:
            queries.extend(self.query_templates[intent_category])
        
        # Generate keyword-based queries
        if keywords:
            # Combine top keywords
            top_keywords = keywords[:3]  # Use top 3 keywords
            keyword_query = " ".join(top_keywords)
            queries.append(keyword_query)
        
        # Generate LLM-optimized query
        try:
            llm_query = self._generate_llm_optimized_query(subject, body, intent_category)
            if llm_query:
                queries.append(llm_query)
        except Exception as e:
            print(f"Failed to generate LLM query: {e}")
        
        # Add subject-based query (cleaned)
        cleaned_subject = self._clean_subject_for_query(subject)
        if cleaned_subject:
            queries.append(cleaned_subject)
        
        # Remove duplicates and empty queries
        unique_queries = list(set(q.strip() for q in queries if q.strip()))
        
        return unique_queries[:5]  # Limit to top 5 queries
    
    def _generate_llm_optimized_query(
        self, 
        subject: str, 
        body: str, 
        intent_category: IntentCategory
    ) -> str:
        """Generate an optimized search query using LLM analysis."""
        try:
            system_prompt = """You are an expert at creating search queries for a knowledge base.
Given an email subject, body, and intent category, create a concise search query 
that will find the most relevant help articles.

Focus on:
- Key technical terms and concepts
- Specific problems or requests mentioned
- Actionable keywords that would appear in solution articles

Return only the search query, nothing else."""
            
            user_prompt = f"""
Email Subject: {subject}
Email Body: {body[:500]}...  
Intent Category: {intent_category.value if intent_category else 'general'}

Generate an optimized search query:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            print(f"LLM query generation failed: {e}")
            return ""
    
    def _clean_subject_for_query(self, subject: str) -> str:
        """Clean email subject for use as search query."""
        # Remove common email prefixes
        cleaned = subject
        prefixes = ["re:", "fwd:", "fw:", "urgent:", "important:"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove special characters but keep spaces
        import re
        cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def _search_knowledge_base(self, queries: List[str]) -> Dict[str, Any]:
        """Search AWS Kendra knowledge base with multiple queries."""
        if not self.kendra_client or not self.kendra_index_id:
            # Return mock results for development/testing
            return self._get_mock_search_results(queries)
        
        all_results = []
        query_id = None
        
        for query in queries:
            try:
                response = self.kendra_client.query(
                    IndexId=self.kendra_index_id,
                    QueryText=query,
                    PageSize=10,
                    AttributeFilter={
                        'EqualsTo': {
                            'Key': 'source',
                            'Value': {
                                'StringValue': 'ServiceNow'
                            }
                        }
                    }
                )
                
                if not query_id:
                    query_id = response.get('QueryId')
                
                # Extract results
                for result in response.get('ResultItems', []):
                    all_results.append({
                        'query': query,
                        'result': result
                    })
                    
            except (ClientError, BotoCoreError) as e:
                print(f"Kendra query failed for '{query}': {e}")
                continue
        
        return {
            'QueryId': query_id,
            'Results': all_results
        }
    
    def _get_mock_search_results(self, queries: List[str]) -> Dict[str, Any]:
        """Generate mock search results for development/testing."""
        mock_results = []
        
        for i, query in enumerate(queries[:3]):  # Limit to 3 results
            mock_results.append({
                'query': query,
                'result': {
                    'Id': f'mock_result_{i}',
                    'Type': 'DOCUMENT',
                    'DocumentTitle': {
                        'Text': f'How to resolve: {query}',
                        'Highlights': []
                    },
                    'DocumentExcerpt': {
                        'Text': f'This article provides step-by-step instructions for {query}. '
                               f'Follow these procedures to resolve the issue quickly.',
                        'Highlights': []
                    },
                    'DocumentURI': f'https://servicenow.example.com/kb/article/{i}',
                    'ScoreAttributes': {
                        'ScoreConfidence': 'HIGH' if i == 0 else 'MEDIUM'
                    },
                    'AdditionalAttributes': [
                        {
                            'Key': 'source',
                            'Value': {'StringValue': 'ServiceNow'}
                        },
                        {
                            'Key': 'category',
                            'Value': {'StringValue': 'IT Support'}
                        }
                    ]
                }
            })
        
        return {
            'QueryId': 'mock_query_id',
            'Results': mock_results
        }
    
    def _process_search_results(
        self, 
        search_results: Dict[str, Any], 
        intent_result: Dict[str, Any]
    ) -> List[KnowledgeBaseArticle]:
        """Process and structure search results into KnowledgeBaseArticle objects."""
        articles = []
        
        for item in search_results.get('Results', []):
            result = item['result']
            
            try:
                # Extract article information
                title = result.get('DocumentTitle', {}).get('Text', 'Untitled Article')
                excerpt = result.get('DocumentExcerpt', {}).get('Text', '')
                uri = result.get('DocumentURI', '')
                
                # Calculate confidence score
                confidence = self._calculate_confidence_score(result, intent_result)
                
                # Extract metadata
                source = 'ServiceNow'
                tags = []
                last_updated = None
                
                for attr in result.get('AdditionalAttributes', []):
                    key = attr.get('Key', '')
                    value = attr.get('Value', {})
                    
                    if key == 'source':
                        source = value.get('StringValue', 'ServiceNow')
                    elif key == 'category':
                        tags.append(value.get('StringValue', ''))
                    elif key == 'last_updated':
                        try:
                            last_updated = datetime.fromisoformat(value.get('StringValue', ''))
                        except:
                            pass
                
                # Create article object
                article = KnowledgeBaseArticle(
                    article_id=result.get('Id', f'article_{len(articles)}'),
                    title=title,
                    content=excerpt,
                    url=uri,
                    confidence_score=confidence,
                    source=source,
                    tags=tags,
                    last_updated=last_updated
                )
                
                articles.append(article)
                
            except Exception as e:
                print(f"Failed to process search result: {e}")
                continue
        
        # Sort by confidence score
        articles.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return articles
    
    def _calculate_confidence_score(
        self, 
        result: Dict[str, Any], 
        intent_result: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for search result relevance."""
        base_score = 0.5
        
        # Kendra confidence mapping
        kendra_confidence = result.get('ScoreAttributes', {}).get('ScoreConfidence', 'LOW')
        confidence_mapping = {
            'VERY_HIGH': 0.95,
            'HIGH': 0.85,
            'MEDIUM': 0.65,
            'LOW': 0.45
        }
        base_score = confidence_mapping.get(kendra_confidence, 0.5)
        
        # Boost score based on keyword matches
        title = result.get('DocumentTitle', {}).get('Text', '').lower()
        excerpt = result.get('DocumentExcerpt', {}).get('Text', '').lower()
        content = f"{title} {excerpt}"
        
        keywords = intent_result.get('keywords_found', [])
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content)
        
        if keywords:
            keyword_boost = (keyword_matches / len(keywords)) * 0.2
            base_score = min(base_score + keyword_boost, 1.0)
        
        return round(base_score, 2)
    
    def _analyze_self_service_potential(
        self, 
        articles: List[KnowledgeBaseArticle], 
        intent_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze whether self-service is possible based on article quality."""
        if not articles:
            return {
                "self_service_possible": False,
                "confidence": 0.0,
                "recommended_articles": [],
                "reasoning": "No relevant articles found"
            }
        
        # Get top articles above confidence threshold
        high_confidence_articles = [
            article for article in articles 
            if article['confidence_score'] >= self.confidence_threshold
        ]
        
        if not high_confidence_articles:
            return {
                "self_service_possible": False,
                "confidence": max(article['confidence_score'] for article in articles),
                "recommended_articles": articles[:2],  # Show top 2 anyway
                "reasoning": f"No articles above confidence threshold ({self.confidence_threshold})"
            }
        
        # Check intent category suitability for self-service
        self_service_friendly_intents = [
            IntentCategory.PASSWORD_RESET,
            IntentCategory.SOFTWARE_ISSUE,
            IntentCategory.ACCOUNT_ACCESS,
            IntentCategory.GENERAL_INQUIRY
        ]
        
        intent_category = intent_result.get('intent_category')
        priority = intent_result.get('priority', Priority.LOW)
        
        # Reduce self-service potential for urgent/high priority items
        if priority in [Priority.URGENT, Priority.HIGH]:
            confidence_penalty = 0.2
        else:
            confidence_penalty = 0.0
        
        # Calculate overall confidence
        avg_confidence = sum(
            article['confidence_score'] for article in high_confidence_articles[:3]
        ) / min(len(high_confidence_articles), 3)
        
        final_confidence = max(avg_confidence - confidence_penalty, 0.0)
        
        # Determine if self-service is recommended
        self_service_possible = (
            final_confidence >= self.confidence_threshold and
            intent_category in self_service_friendly_intents and
            priority != Priority.URGENT
        )
        
        return {
            "self_service_possible": self_service_possible,
            "confidence": final_confidence,
            "recommended_articles": high_confidence_articles[:3],
            "reasoning": f"Confidence: {final_confidence:.2f}, Intent: {intent_category.value if intent_category else 'unknown'}, Priority: {priority.value if priority else 'unknown'}"
        }


# Tool function for LangGraph integration
@tool
def retrieve_knowledge_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool function for retrieving knowledge base articles in LangGraph workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated workflow state with knowledge base search results
    """
    # Get configuration from environment or state
    import os
    kendra_index_id = os.getenv('KENDRA_INDEX_ID')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    confidence_threshold = float(os.getenv('KNOWLEDGE_BASE_CONFIDENCE_THRESHOLD', '0.7'))
    
    retriever = KnowledgeRetriever(
        kendra_index_id=kendra_index_id,
        aws_region=aws_region,
        confidence_threshold=confidence_threshold
    )
    return retriever.retrieve_knowledge(state)


# Export the main class and tool
__all__ = ["KnowledgeRetriever", "retrieve_knowledge_tool"]
