"""
Bedrock-compatible agent that works with MCP tools.
This is a wrapper that uses BedrockAgent instead of Claude Agent SDK.
"""

import json
import logging
from typing import Any, AsyncIterator

from .agent import (
    AVAILABLE_TOOLS,
    CONTEXT_GRAPH_SYSTEM_PROMPT,
    detect_fraud_patterns,
    execute_cypher,
    find_accounts_with_high_shared_transaction_volume,
    find_decision_community,
    find_precedents,
    find_similar_decisions,
    get_causal_chain,
    get_customer_decisions,
    get_policy,
    get_schema,
    record_decision,
    search_customer,
)
from .bedrock_agent import BedrockAgent
from .config import config

logger = logging.getLogger(__name__)


def convert_mcp_tools_to_bedrock_format(mcp_tools: list) -> list[dict[str, Any]]:
    """
    Convert MCP tool definitions to Bedrock/Anthropic tool format.
    
    MCP tools are decorated with @tool, we need to extract their schemas.
    """
    bedrock_tools = []
    
    tool_schemas = {
        "search_customer": {
            "name": "search_customer",
            "description": "Search for customers by name, email, or account number. Returns customer profiles with risk scores and related account counts.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 10},
                },
                "required": ["query"],
            },
        },
        "get_customer_decisions": {
            "name": "get_customer_decisions",
            "description": "Get all decisions made about a specific customer, including approvals, rejections, escalations, and exceptions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID"},
                    "decision_type": {"type": "string", "description": "Filter by decision type"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 20},
                },
                "required": ["customer_id"],
            },
        },
        "find_similar_decisions": {
            "name": "find_similar_decisions",
            "description": "Find structurally similar past decisions using FastRP graph embeddings. Returns decisions with similar influences, causes, and precedents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string", "description": "The internal decision ID"},
                    "limit": {"type": "integer", "description": "Number of similar decisions", "default": 5},
                },
                "required": ["decision_id"],
            },
        },
        "find_precedents": {
            "name": "find_precedents",
            "description": "Find precedent decisions that could inform the current decision. Uses both semantic similarity (meaning) and structural similarity (graph patterns).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string", "description": "Scenario description"},
                    "category": {"type": "string", "description": "Decision category"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 5},
                },
                "required": ["scenario"],
            },
        },
        "get_causal_chain": {
            "name": "get_causal_chain",
            "description": "Trace the causal chain of a decision - what caused it and what it led to.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string", "description": "Decision ID"},
                    "direction": {"type": "string", "description": "Direction: 'upstream', 'downstream', or 'both'", "default": "both"},
                    "depth": {"type": "integer", "description": "Depth to traverse", "default": 3},
                },
                "required": ["decision_id"],
            },
        },
        "record_decision": {
            "name": "record_decision",
            "description": "Record a new decision with full reasoning context. Creates a decision trace in the context graph.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision_type": {"type": "string", "description": "Type of decision"},
                    "category": {"type": "string", "description": "Decision category"},
                    "reasoning": {"type": "string", "description": "Full reasoning"},
                    "customer_id": {"type": "string", "description": "Customer ID"},
                    "account_id": {"type": "string", "description": "Account ID"},
                    "risk_factors": {"type": "array", "items": {"type": "string"}, "description": "Risk factors"},
                    "precedent_ids": {"type": "array", "items": {"type": "string"}, "description": "Precedent decision IDs"},
                    "confidence_score": {"type": "number", "description": "Confidence score 0-1", "default": 0.8},
                },
                "required": ["decision_type", "category", "reasoning"],
            },
        },
        "detect_fraud_patterns": {
            "name": "detect_fraud_patterns",
            "description": "Analyze accounts or transactions for potential fraud patterns using graph structure analysis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "The internal account ID"},
                    "neighbor_count": {"type": "integer", "description": "Number of examples to return", "default": 5},
                },
                "required": ["account_id"],
            },
        },
        "find_decision_community": {
            "name": "find_decision_community",
            "description": "Find decisions in the same community using Leiden community detection.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string", "description": "Decision ID"},
                    "example_count": {"type": "integer", "description": "Number of examples", "default": 5},
                },
                "required": ["decision_id"],
            },
        },
        "find_accounts_with_high_shared_transaction_volume": {
            "name": "find_accounts_with_high_shared_transaction_volume",
            "description": "Find accounts that share high transaction volumes with a given account.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "The internal account ID"},
                },
                "required": ["account_id"],
            },
        },
        "get_policy": {
            "name": "get_policy",
            "description": "Get the current policy rules for a specific category.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Policy category"},
                    "policy_name": {"type": "string", "description": "Policy name to search for"},
                },
                "required": [],
            },
        },
        "execute_cypher": {
            "name": "execute_cypher",
            "description": "Execute a read-only Cypher query against the context graph for custom analysis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "cypher": {"type": "string", "description": "Cypher query"},
                },
                "required": ["cypher"],
            },
        },
        "get_schema": {
            "name": "get_schema",
            "description": "Get the graph database schema including node labels, relationship types, and property keys.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
    
    return list(tool_schemas.values())


class ContextGraphAgentBedrock:
    """Bedrock-based agent for context graph operations."""

    def __init__(self):
        self.tools = convert_mcp_tools_to_bedrock_format(AVAILABLE_TOOLS)
        
        # Map tool names to their SdkMcpTool objects
        self.tool_objects = {
            "search_customer": search_customer,
            "get_customer_decisions": get_customer_decisions,
            "find_similar_decisions": find_similar_decisions,
            "find_precedents": find_precedents,
            "get_causal_chain": get_causal_chain,
            "record_decision": record_decision,
            "detect_fraud_patterns": detect_fraud_patterns,
            "find_decision_community": find_decision_community,
            "find_accounts_with_high_shared_transaction_volume": find_accounts_with_high_shared_transaction_volume,
            "get_policy": get_policy,
            "execute_cypher": execute_cypher,
            "get_schema": get_schema,
        }
        
        # Create wrapper functions that properly invoke the SdkMcpTool objects
        self.tool_handlers = {}
        for name, tool_obj in self.tool_objects.items():
            # SdkMcpTool objects have a 'handler' attribute that is the actual async function
            if hasattr(tool_obj, 'handler'):
                self.tool_handlers[name] = tool_obj.handler
            elif callable(tool_obj):
                self.tool_handlers[name] = tool_obj
            else:
                # Fallback: try to get the underlying function
                self.tool_handlers[name] = getattr(tool_obj, 'func', tool_obj)
        
        self.agent = BedrockAgent(
            system_prompt=CONTEXT_GRAPH_SYSTEM_PROMPT,
            tools=self.tools,
            model_id=config.bedrock.claude_model_id,
        )

    async def __aenter__(self):
        await self.agent.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.agent.disconnect()

    async def query(
        self, message: str, conversation_history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Send a query to the agent and get the response."""
        if conversation_history:
            for msg in conversation_history[-6:]:
                self.agent.conversation_history.append(msg)
        
        result = await self.agent.run_agentic_loop(
            message=message,
            tool_handlers=self.tool_handlers,
            max_iterations=10,
        )
        
        return {
            "response": result.get("response", ""),
            "tool_calls": result.get("tool_calls", []),
            "decisions_made": [],
        }

    async def query_stream(
        self, message: str, conversation_history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a query to the agent with streaming response."""
        if conversation_history:
            for msg in conversation_history[-6:]:
                self.agent.conversation_history.append(msg)
        
        yield {"type": "agent_context", "context": {
            "system_prompt": CONTEXT_GRAPH_SYSTEM_PROMPT,
            "model": config.bedrock.claude_model_id,
            "available_tools": AVAILABLE_TOOLS,
        }}
        
        async for event in self.agent.run_agentic_loop_stream(
            message=message,
            tool_handlers=self.tool_handlers,
            max_iterations=10,
        ):
            yield event
        
        yield {"type": "done", "tool_calls": [], "decisions_made": []}
"""
Bedrock-compatible agent that works with MCP tools.
This is a wrapper that uses BedrockAgent instead of Claude Agent SDK.
"""

import json
import logging
from typing import Any, AsyncIterator

from .agent import (
    AVAILABLE_TOOLS,
    CONTEXT_GRAPH_SYSTEM_PROMPT,
    detect_fraud_patterns,
    execute_cypher,
    find_accounts_with_high_shared_transaction_volume,
    find_decision_community,
    find_precedents,
    find_similar_decisions,
    get_causal_chain,
    get_customer_decisions,
    get_policy,
    get_schema,
    record_decision,
    search_customer,
)
from .bedrock_agent import BedrockAgent
from .config import config

logger = logging.getLogger(__name__)


def convert_mcp_tools_to_bedrock_format(mcp_tools: list) -> list[dict[str, Any]]:
    """
    Convert MCP tool definitions to Bedrock/Anthropic tool format.
    
    MCP tools are decorated with @tool, we need to extract their schemas.
    """
    bedrock_tools = []
    
    tool_schemas = {
        "search_customer": {
            "name": "search_customer",
            "description": "Search for customers by name, email, or account number. Returns customer profiles with risk scores and related account counts.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 10},
                },
                "required": ["query"],
            },
        },
        "get_customer_decisions": {
            "name": "get_customer_decisions",
            "description": "Get all decisions made about a specific customer, including approvals, rejections, escalations, and exceptions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID"},
                    "decision_type": {"type": "string", "description": "Filter by decision type"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 20},
                },
                "required": ["customer_id"],
            },
        },
        "find_similar_decisions": {
            "name": "find_similar_decisions",
            "description": "Find structurally similar past decisions using FastRP graph embeddings. Returns decisions with similar influences, causes, and precedents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string", "description": "The internal decision ID"},
                    "limit": {"type": "integer", "description": "Number of similar decisions", "default": 5},
                },
                "required": ["decision_id"],
            },
        },
        "find_precedents": {
            "name": "find_precedents",
            "description": "Find precedent decisions that could inform the current decision. Uses both semantic similarity (meaning) and structural similarity (graph patterns).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string", "description": "Scenario description"},
                    "category": {"type": "string", "description": "Decision category"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 5},
                },
                "required": ["scenario"],
            },
        },
        "get_causal_chain": {
            "name": "get_causal_chain",
            "description": "Trace the causal chain of a decision - what caused it and what it led to.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string", "description": "Decision ID"},
                    "direction": {"type": "string", "description": "Direction: 'upstream', 'downstream', or 'both'", "default": "both"},
                    "depth": {"type": "integer", "description": "Depth to traverse", "default": 3},
                },
                "required": ["decision_id"],
            },
        },
        "record_decision": {
            "name": "record_decision",
            "description": "Record a new decision with full reasoning context. Creates a decision trace in the context graph.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision_type": {"type": "string", "description": "Type of decision"},
                    "category": {"type": "string", "description": "Decision category"},
                    "reasoning": {"type": "string", "description": "Full reasoning"},
                    "customer_id": {"type": "string", "description": "Customer ID"},
                    "account_id": {"type": "string", "description": "Account ID"},
                    "risk_factors": {"type": "array", "items": {"type": "string"}, "description": "Risk factors"},
                    "precedent_ids": {"type": "array", "items": {"type": "string"}, "description": "Precedent decision IDs"},
                    "confidence_score": {"type": "number", "description": "Confidence score 0-1", "default": 0.8},
                },
                "required": ["decision_type", "category", "reasoning"],
            },
        },
        "detect_fraud_patterns": {
            "name": "detect_fraud_patterns",
            "description": "Analyze accounts or transactions for potential fraud patterns using graph structure analysis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "The internal account ID"},
                    "neighbor_count": {"type": "integer", "description": "Number of examples to return", "default": 5},
                },
                "required": ["account_id"],
            },
        },
        "find_decision_community": {
            "name": "find_decision_community",
            "description": "Find decisions in the same community using Leiden community detection.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string", "description": "Decision ID"},
                    "example_count": {"type": "integer", "description": "Number of examples", "default": 5},
                },
                "required": ["decision_id"],
            },
        },
        "find_accounts_with_high_shared_transaction_volume": {
            "name": "find_accounts_with_high_shared_transaction_volume",
            "description": "Find accounts that share high transaction volumes with a given account.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "The internal account ID"},
                },
                "required": ["account_id"],
            },
        },
        "get_policy": {
            "name": "get_policy",
            "description": "Get the current policy rules for a specific category.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Policy category"},
                    "policy_name": {"type": "string", "description": "Policy name to search for"},
                },
                "required": [],
            },
        },
        "execute_cypher": {
            "name": "execute_cypher",
            "description": "Execute a read-only Cypher query against the context graph for custom analysis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "cypher": {"type": "string", "description": "Cypher query"},
                },
                "required": ["cypher"],
            },
        },
        "get_schema": {
            "name": "get_schema",
            "description": "Get the graph database schema including node labels, relationship types, and property keys.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
    
    return list(tool_schemas.values())


class ContextGraphAgentBedrock:
    """Bedrock-based agent for context graph operations."""

    def __init__(self):
        self.tools = convert_mcp_tools_to_bedrock_format(AVAILABLE_TOOLS)
        
        # Map tool names to their SdkMcpTool objects
        self.tool_objects = {
            "search_customer": search_customer,
            "get_customer_decisions": get_customer_decisions,
            "find_similar_decisions": find_similar_decisions,
            "find_precedents": find_precedents,
            "get_causal_chain": get_causal_chain,
            "record_decision": record_decision,
            "detect_fraud_patterns": detect_fraud_patterns,
            "find_decision_community": find_decision_community,
            "find_accounts_with_high_shared_transaction_volume": find_accounts_with_high_shared_transaction_volume,
            "get_policy": get_policy,
            "execute_cypher": execute_cypher,
            "get_schema": get_schema,
        }
        
        # Create wrapper functions that properly invoke the SdkMcpTool objects
        self.tool_handlers = {}
        for name, tool_obj in self.tool_objects.items():
            # SdkMcpTool objects have a 'handler' attribute that is the actual async function
            if hasattr(tool_obj, 'handler'):
                self.tool_handlers[name] = tool_obj.handler
            elif callable(tool_obj):
                self.tool_handlers[name] = tool_obj
            else:
                # Fallback: try to get the underlying function
                self.tool_handlers[name] = getattr(tool_obj, 'func', tool_obj)
        
        self.agent = BedrockAgent(
            system_prompt=CONTEXT_GRAPH_SYSTEM_PROMPT,
            tools=self.tools,
            model_id=config.bedrock.claude_model_id,
        )

    async def __aenter__(self):
        await self.agent.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.agent.disconnect()

    async def query(
        self, message: str, conversation_history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Send a query to the agent and get the response."""
        if conversation_history:
            for msg in conversation_history[-6:]:
                self.agent.conversation_history.append(msg)
        
        result = await self.agent.run_agentic_loop(
            message=message,
            tool_handlers=self.tool_handlers,
            max_iterations=10,
        )
        
        return {
            "response": result.get("response", ""),
            "tool_calls": result.get("tool_calls", []),
            "decisions_made": [],
        }

    async def query_stream(
        self, message: str, conversation_history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a query to the agent with streaming response."""
        if conversation_history:
            for msg in conversation_history[-6:]:
                self.agent.conversation_history.append(msg)
        
        yield {"type": "agent_context", "context": {
            "system_prompt": CONTEXT_GRAPH_SYSTEM_PROMPT,
            "model": config.bedrock.claude_model_id,
            "available_tools": AVAILABLE_TOOLS,
        }}
        
        async for event in self.agent.run_agentic_loop_stream(
            message=message,
            tool_handlers=self.tool_handlers,
            max_iterations=10,
        ):
            yield event
        
        yield {"type": "done", "tool_calls": [], "decisions_made": []}
