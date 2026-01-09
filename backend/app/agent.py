"""
Claude Agent SDK integration with Context Graph tools.
Provides MCP tools for querying and updating the context graph.
"""

import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, tool

from .context_graph_client import context_graph_client
from .gds_client import gds_client
from .vector_client import vector_client


def get_graph_data_for_entity(entity_id: str, depth: int = 2) -> dict:
    """Get graph visualization data centered on an entity."""
    try:
        graph_data = context_graph_client.get_graph_data(
            center_node_id=entity_id, depth=depth, limit=50
        )
        return {
            "nodes": [
                {
                    "id": node.id,
                    "labels": node.labels,
                    "properties": node.properties,
                }
                for node in graph_data.nodes
            ],
            "relationships": [
                {
                    "id": rel.id,
                    "type": rel.type,
                    "startNodeId": rel.startNodeId,
                    "endNodeId": rel.endNodeId,
                    "properties": rel.properties,
                }
                for rel in graph_data.relationships
            ],
        }
    except Exception:
        return {"nodes": [], "relationships": []}


# ============================================
# SYSTEM PROMPT
# ============================================

CONTEXT_GRAPH_SYSTEM_PROMPT = """You are an AI assistant for a financial institution with access to a Context Graph.

The Context Graph stores decision traces - the reasoning, context, and causal relationships behind every significant decision made in the organization. This enables you to:

1. **Find Precedents**: Search for similar past decisions to inform current recommendations
2. **Trace Causality**: Understand how past decisions influenced subsequent outcomes
3. **Record Decisions**: Create new decision traces with full reasoning context
4. **Detect Patterns**: Identify fraud patterns and entity duplicates using graph structure

## Key Concepts

**Event Clock vs State Clock**:
- Traditional systems store the "state clock" - what is true right now
- The Context Graph stores the "event clock" - what happened, when, and with what reasoning

**Decision Traces**:
- Every significant decision is recorded with full reasoning
- Risk factors, confidence scores, and applied policies are captured
- Causal chains show how decisions influenced each other

## Guidelines

When helping users:
1. **Always search for precedents** before making recommendations
2. **Explain your reasoning thoroughly** - this becomes part of the decision trace
3. **Cite specific past decisions** when they inform your recommendation
4. **Flag exceptions or escalations** that may be needed
5. **Consider both structural and semantic similarity** when finding related cases

You have access to tools that leverage both:
- **Semantic similarity** (text embeddings) - for matching by meaning
- **Structural similarity** (FastRP graph embeddings) - for matching by relationship patterns

This combination provides insights that are impossible with traditional databases."""


# ============================================
# MCP TOOLS
# ============================================


@tool(
    "search_customer",
    "Search for customers by name, email, or account number. Returns customer profiles with risk scores and related account counts.",
    {"query": str, "limit": int},
)
async def search_customer(args: dict[str, Any]) -> dict[str, Any]:
    """Search for customers in the context graph."""
    try:
        results = context_graph_client.search_customers(
            query=args["query"], limit=args.get("limit", 10)
        )
        # Include graph data for the first result
        graph_data = None
        if results and len(results) > 0:
            graph_data = get_graph_data_for_entity(results[0].get("id"), depth=2)

        response = {
            "customers": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error searching customers: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "get_customer_decisions",
    "Get all decisions made about a specific customer, including approvals, rejections, escalations, and exceptions.",
    {"customer_id": str, "decision_type": str, "limit": int},
)
async def get_customer_decisions(args: dict[str, Any]) -> dict[str, Any]:
    """Get decisions about a customer."""
    try:
        results = context_graph_client.get_customer_decisions(
            customer_id=args["customer_id"],
            decision_type=args.get("decision_type"),
            limit=args.get("limit", 20),
        )
        # Include graph data centered on the customer
        graph_data = get_graph_data_for_entity(args["customer_id"], depth=2)

        response = {
            "decisions": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error getting decisions: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "find_similar_decisions",
    "Find structurally similar past decisions using FastRP graph embeddings. Returns decisions with similar patterns of entities, relationships, and outcomes.",
    {"decision_id": str, "limit": int},
)
async def find_similar_decisions(args: dict[str, Any]) -> dict[str, Any]:
    """Find similar decisions using FastRP embeddings."""
    try:
        results = gds_client.find_similar_decisions_knn(
            decision_id=args["decision_id"], limit=args.get("limit", 5)
        )
        # Include graph data centered on the original decision
        graph_data = get_graph_data_for_entity(args["decision_id"], depth=2)

        response = {
            "similar_decisions": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error finding similar decisions: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "find_precedents",
    "Find precedent decisions that could inform the current decision. Uses both semantic similarity (meaning) and structural similarity (graph patterns).",
    {"scenario": str, "category": str, "limit": int},
)
async def find_precedents(args: dict[str, Any]) -> dict[str, Any]:
    """Find precedent decisions using hybrid search."""
    try:
        results = vector_client.find_precedents_hybrid(
            scenario=args["scenario"], category=args.get("category"), limit=args.get("limit", 5)
        )
        # Include graph data for the first precedent found
        graph_data = None
        if results and len(results) > 0:
            first_id = results[0].get("id") if isinstance(results[0], dict) else None
            if first_id:
                graph_data = get_graph_data_for_entity(first_id, depth=2)

        response = {
            "precedents": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error finding precedents: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "get_causal_chain",
    "Trace the causal chain of a decision - what caused it and what it led to. Useful for understanding decision impact and history.",
    {"decision_id": str, "direction": str, "depth": int},
)
async def get_causal_chain(args: dict[str, Any]) -> dict[str, Any]:
    """Get the causal chain for a decision."""
    try:
        results = context_graph_client.get_causal_chain(
            decision_id=args["decision_id"],
            direction=args.get("direction", "both"),
            depth=args.get("depth", 3),
        )
        # Include graph data centered on the decision
        graph_data = get_graph_data_for_entity(args["decision_id"], depth=3)

        response = {
            "causal_chain": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error getting causal chain: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "record_decision",
    "Record a new decision with full reasoning context. Creates a decision trace in the context graph that can be referenced by future decisions.",
    {
        "decision_type": str,
        "category": str,
        "reasoning": str,
        "customer_id": str,
        "account_id": str,
        "risk_factors": list,
        "precedent_ids": list,
        "confidence_score": float,
    },
)
async def record_decision(args: dict[str, Any]) -> dict[str, Any]:
    """Record a new decision in the context graph."""
    try:
        # Generate embedding for the reasoning
        reasoning_embedding = None
        try:
            reasoning_embedding = vector_client.generate_embedding(args["reasoning"])
        except Exception:
            pass  # Continue without embedding if it fails

        decision_id = context_graph_client.record_decision(
            decision_type=args["decision_type"],
            category=args["category"],
            reasoning=args["reasoning"],
            customer_id=args.get("customer_id"),
            account_id=args.get("account_id"),
            risk_factors=args.get("risk_factors", []),
            precedent_ids=args.get("precedent_ids", []),
            confidence_score=args.get("confidence_score", 0.8),
            reasoning_embedding=reasoning_embedding,
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "success": True,
                            "decision_id": decision_id,
                            "message": f"Decision recorded successfully with ID {decision_id}",
                        },
                        indent=2,
                    ),
                }
            ]
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error recording decision: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "detect_fraud_patterns",
    "Analyze accounts or transactions for potential fraud patterns using graph structure analysis. Uses Node Similarity to compare against known fraud cases.",
    {"account_id": str, "similarity_threshold": float},
)
async def detect_fraud_patterns(args: dict[str, Any]) -> dict[str, Any]:
    """Detect fraud patterns using graph analysis."""
    try:
        results = gds_client.detect_fraud_patterns(
            account_id=args.get("account_id"),
            similarity_threshold=args.get("similarity_threshold", 0.7),
        )
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error detecting fraud patterns: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "get_policy",
    "Get the current policy rules for a specific category. Returns policy details including thresholds and requirements.",
    {"category": str, "policy_name": str},
)
async def get_policy(args: dict[str, Any]) -> dict[str, Any]:
    """Get policy information."""
    try:
        if args.get("policy_name"):
            # Search by name
            policies = context_graph_client.get_policies(category=args.get("category"))
            matching = [
                p for p in policies if args["policy_name"].lower() in p.get("name", "").lower()
            ]
            results = matching[0] if matching else None
        else:
            results = context_graph_client.get_policies(category=args.get("category"))

        return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error getting policy: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "execute_cypher",
    "Execute a read-only Cypher query against the context graph for custom analysis. Only SELECT/MATCH queries are allowed.",
    {"cypher": str},
)
async def execute_cypher(args: dict[str, Any]) -> dict[str, Any]:
    """Execute a read-only Cypher query."""
    try:
        results = context_graph_client.execute_cypher(cypher=args["cypher"])
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}
    except ValueError as e:
        return {
            "content": [{"type": "text", "text": f"Query not allowed: {str(e)}"}],
            "is_error": True,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error executing query: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "get_schema",
    "Get the graph database schema including node labels, relationship types, property keys, indexes, and constraints. Also returns counts for each node label and relationship type.",
    {},
)
async def get_schema(args: dict[str, Any]) -> dict[str, Any]:
    """Get the graph database schema."""
    try:
        schema = context_graph_client.get_schema()
        return {"content": [{"type": "text", "text": json.dumps(schema, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error getting schema: {str(e)}"}],
            "is_error": True,
        }


# ============================================
# MCP SERVER CREATION
# ============================================


def create_context_graph_server():
    """Create the MCP server with all context graph tools."""
    return create_sdk_mcp_server(
        name="context-graph",
        version="1.0.0",
        tools=[
            search_customer,
            get_customer_decisions,
            find_similar_decisions,
            find_precedents,
            get_causal_chain,
            record_decision,
            detect_fraud_patterns,
            get_policy,
            execute_cypher,
            get_schema,
        ],
    )


def get_agent_options() -> ClaudeAgentOptions:
    """Get the agent options with context graph server configured."""
    context_graph_server = create_context_graph_server()

    return ClaudeAgentOptions(
        system_prompt=CONTEXT_GRAPH_SYSTEM_PROMPT,
        mcp_servers={"graph": context_graph_server},
        allowed_tools=[
            "mcp__graph__search_customer",
            "mcp__graph__get_customer_decisions",
            "mcp__graph__find_similar_decisions",
            "mcp__graph__find_precedents",
            "mcp__graph__get_causal_chain",
            "mcp__graph__record_decision",
            "mcp__graph__detect_fraud_patterns",
            "mcp__graph__get_policy",
            "mcp__graph__execute_cypher",
            "mcp__graph__get_schema",
        ],
    )


# ============================================
# AGENT CONTEXT
# ============================================

AVAILABLE_TOOLS = [
    "search_customer",
    "get_customer_decisions",
    "find_similar_decisions",
    "find_precedents",
    "get_causal_chain",
    "record_decision",
    "detect_fraud_patterns",
    "get_policy",
    "execute_cypher",
    "get_schema",
]


def get_agent_context() -> dict[str, Any]:
    """Get agent context information for transparency/debugging."""
    return {
        "system_prompt": CONTEXT_GRAPH_SYSTEM_PROMPT,
        "model": "claude-sonnet-4-20250514",
        "available_tools": AVAILABLE_TOOLS,
        "mcp_server": "context-graph",
    }


# ============================================
# AGENT SESSION MANAGEMENT
# ============================================


class ContextGraphAgent:
    """Wrapper for managing Claude Agent SDK sessions."""

    def __init__(self):
        self.options = get_agent_options()
        self.client: ClaudeSDKClient | None = None

    async def __aenter__(self):
        self.client = ClaudeSDKClient(options=self.options)
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.disconnect()

    async def query(
        self, message: str, conversation_history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Send a query to the agent and get the response."""
        if not self.client:
            raise RuntimeError("Agent not connected. Use 'async with' context manager.")

        # Build message with conversation context
        if conversation_history and len(conversation_history) > 0:
            # Format history as context in the message
            history_text = "\n".join(
                [
                    f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history[-6:]
                ]  # Last 6 messages
            )
            full_message = f"""Previous conversation:
{history_text}

Current message from USER: {message}

Please respond to the current message, taking the conversation history into account."""
        else:
            full_message = message

        # Send the message
        await self.client.query(full_message)

        response_text = ""
        tool_calls = []
        decisions_made = []

        async for msg in self.client.receive_response():
            # Process different message types
            if hasattr(msg, "content"):
                for block in msg.content:
                    if hasattr(block, "text"):
                        response_text += block.text
                    elif hasattr(block, "name"):
                        # Tool use block
                        tool_calls.append(
                            {
                                "name": block.name,
                                "input": block.input if hasattr(block, "input") else {},
                            }
                        )
                        # Track decisions made
                        if block.name == "mcp__graph__record_decision":
                            # Will be populated when we get the result
                            pass

        return {
            "response": response_text,
            "tool_calls": tool_calls,
            "decisions_made": decisions_made,
        }

    async def query_stream(
        self, message: str, conversation_history: list[dict[str, str]] | None = None
    ):
        """Send a query to the agent and stream the response."""
        if not self.client:
            raise RuntimeError("Agent not connected. Use 'async with' context manager.")

        # Build message with conversation context
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history[-6:]]
            )
            full_message = f"""Previous conversation:
{history_text}

Current message from USER: {message}

Please respond to the current message, taking the conversation history into account."""
        else:
            full_message = message

        # Emit agent context first
        yield {"type": "agent_context", "context": get_agent_context()}

        # Send the message
        await self.client.query(full_message)

        tool_calls = []
        tool_id_to_name = {}  # Map tool_use_id to tool name
        decisions_made = []

        async for msg in self.client.receive_response():
            msg_type = type(msg).__name__

            # Handle UserMessage containing ToolResultBlock objects
            if msg_type == "UserMessage" and hasattr(msg, "content"):
                for block in msg.content:
                    block_type = type(block).__name__
                    # ToolResultBlock has tool_use_id and content attributes
                    if block_type == "ToolResultBlock":
                        tool_use_id = getattr(block, "tool_use_id", None)
                        block_content = getattr(block, "content", None)

                        print(f"[DEBUG] ToolResultBlock - tool_use_id: {tool_use_id}")

                        if tool_use_id:
                            parsed_output = None

                            # Parse the block content (list of content items)
                            if isinstance(block_content, list):
                                for item in block_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        try:
                                            parsed_output = json.loads(item.get("text", "{}"))
                                        except json.JSONDecodeError:
                                            parsed_output = item.get("text")
                                        break
                                    elif hasattr(item, "text"):
                                        try:
                                            parsed_output = json.loads(item.text)
                                        except json.JSONDecodeError:
                                            parsed_output = item.text
                                        break
                            elif isinstance(block_content, str):
                                try:
                                    parsed_output = json.loads(block_content)
                                except json.JSONDecodeError:
                                    parsed_output = block_content

                            # Look up the tool name from the tool_use_id
                            tool_name = tool_id_to_name.get(tool_use_id, "unknown")
                            print(
                                f"[DEBUG] Yielding tool_result: name={tool_name}, output_type={type(parsed_output)}"
                            )

                            yield {
                                "type": "tool_result",
                                "name": tool_name,
                                "output": parsed_output,
                            }
                continue

            # Handle AssistantMessage content blocks
            if hasattr(msg, "content"):
                for block in msg.content:
                    if hasattr(block, "text"):
                        # Stream text content
                        yield {"type": "text", "content": block.text}
                    elif hasattr(block, "name"):
                        # Tool use block
                        tool_id = getattr(block, "id", None)
                        tool_call = {
                            "name": block.name,
                            "input": block.input if hasattr(block, "input") else {},
                        }
                        tool_calls.append(tool_call)

                        # Track tool_use_id to name mapping
                        if tool_id:
                            tool_id_to_name[tool_id] = block.name

                        yield {"type": "tool_use", **tool_call}

                        # Track decisions made
                        if block.name == "mcp__graph__record_decision":
                            pass

        # Final event with summary
        yield {
            "type": "done",
            "tool_calls": tool_calls,
            "decisions_made": decisions_made,
        }
