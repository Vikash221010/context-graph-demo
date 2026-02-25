"""
Bedrock-based agent implementation that mimics Claude Agent SDK interface.
This allows using Amazon Bedrock instead of direct Anthropic API.
"""

import json
import logging
from typing import Any, AsyncIterator, Optional

from .bedrock_client import BedrockClaudeClient
from .config import config

logger = logging.getLogger(__name__)


class BedrockAgent:
    """
    Agent implementation using Amazon Bedrock for Claude.
    Provides similar interface to Claude Agent SDK but uses Bedrock.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ):
        """
        Initialize Bedrock agent.
        
        Args:
            system_prompt: System prompt for the agent
            tools: List of tool definitions in Anthropic format
            model_id: Bedrock model ID
        """
        self.system_prompt = system_prompt
        self.tools = tools
        self.model_id = model_id
        
        self.client = BedrockClaudeClient(
            region_name=config.bedrock.region_name,
            model_id=model_id,
            aws_access_key_id=config.bedrock.aws_access_key_id,
            aws_secret_access_key=config.bedrock.aws_secret_access_key,
            aws_session_token=config.bedrock.aws_session_token,
        )
        
        self.conversation_history: list[dict[str, Any]] = []

    async def connect(self):
        """Connect to Bedrock (no-op, connection is per-request)."""
        pass

    async def disconnect(self):
        """Disconnect from Bedrock (no-op)."""
        pass

    async def query(
        self,
        message: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """
        Send a query to Claude via Bedrock.
        
        Args:
            message: User message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response dict with text and tool calls
        """
        messages = self.conversation_history + [
            {"role": "user", "content": message}
        ]

        response = self.client.invoke(
            messages=messages,
            system=self.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=self.tools if self.tools else None,
        )

        self.conversation_history.append({"role": "user", "content": message})
        
        if response.get("content"):
            self.conversation_history.append({
                "role": "assistant",
                "content": response["content"]
            })

        return response

    async def query_stream(
        self,
        message: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Send a query to Claude via Bedrock with streaming.
        
        Yields response chunks as they arrive.
        """
        messages = self.conversation_history + [
            {"role": "user", "content": message}
        ]

        self.conversation_history.append({"role": "user", "content": message})
        
        accumulated_content = []

        async for chunk in self.client.invoke_stream(
            messages=messages,
            system=self.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=self.tools if self.tools else None,
        ):
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    accumulated_content.append({
                        "type": "text",
                        "text": delta.get("text", "")
                    })
                    yield chunk
            elif chunk.get("type") == "content_block_start":
                content_block = chunk.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    accumulated_content.append(content_block)
                yield chunk
            else:
                yield chunk

        if accumulated_content:
            self.conversation_history.append({
                "role": "assistant",
                "content": accumulated_content
            })

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_handlers: dict[str, callable],
    ) -> dict[str, Any]:
        """
        Execute a tool and return the result.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            tool_handlers: Dict mapping tool names to handler functions
            
        Returns:
            Tool execution result
        """
        if tool_name not in tool_handlers:
            return {
                "content": [{"type": "text", "text": f"Tool {tool_name} not found"}],
                "is_error": True,
            }

        try:
            handler = tool_handlers[tool_name]
            result = await handler(tool_input)
            return result
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return {
                "content": [{"type": "text", "text": f"Error executing {tool_name}: {str(e)}"}],
                "is_error": True,
            }

    async def run_agentic_loop(
        self,
        message: str,
        tool_handlers: dict[str, callable],
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Run the agentic loop: query -> tool use -> tool result -> repeat.
        
        Args:
            message: Initial user message
            tool_handlers: Dict mapping tool names to handler functions
            max_iterations: Maximum number of iterations
            
        Returns:
            Final response with text and tool calls
        """
        response_text = ""
        tool_calls = []
        
        for iteration in range(max_iterations):
            response = await self.query(message if iteration == 0 else "")
            
            stop_reason = response.get("stop_reason")
            content = response.get("content", [])
            
            has_tool_use = False
            tool_results = []
            
            for block in content:
                if block.get("type") == "text":
                    response_text += block.get("text", "")
                elif block.get("type") == "tool_use":
                    has_tool_use = True
                    tool_name = block.get("name")
                    tool_input = block.get("input", {})
                    tool_use_id = block.get("id")
                    
                    tool_calls.append({
                        "name": tool_name,
                        "input": tool_input,
                    })
                    
                    result = await self.execute_tool(
                        tool_name,
                        tool_input,
                        tool_handlers,
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result.get("content", []),
                        "is_error": result.get("is_error", False),
                    })
            
            if has_tool_use and tool_results:
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results,
                })
            elif stop_reason == "end_turn":
                break
        
        return {
            "response": response_text,
            "tool_calls": tool_calls,
        }

    async def run_agentic_loop_stream(
        self,
        message: str,
        tool_handlers: dict[str, callable],
        max_iterations: int = 10,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Run the agentic loop with streaming responses.
        
        Yields events as they occur.
        """
        for iteration in range(max_iterations):
            current_tool_use = None
            tool_results = []
            has_tool_use = False
            
            async for chunk in self.query_stream(message if iteration == 0 else ""):
                chunk_type = chunk.get("type")
                
                if chunk_type == "content_block_start":
                    content_block = chunk.get("content_block", {})
                    if content_block.get("type") == "tool_use":
                        current_tool_use = {
                            "id": content_block.get("id"),
                            "name": content_block.get("name"),
                            "input": {},
                        }
                        has_tool_use = True
                        yield {
                            "type": "tool_use",
                            "name": current_tool_use["name"],
                            "input": {},
                        }
                
                elif chunk_type == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield {
                            "type": "text",
                            "content": delta.get("text", ""),
                        }
                    elif delta.get("type") == "input_json_delta" and current_tool_use:
                        partial_json = delta.get("partial_json", "")
                        if partial_json:
                            try:
                                current_tool_use["input"] = json.loads(
                                    json.dumps(current_tool_use.get("input", {})) + partial_json
                                )
                            except:
                                pass
                
                elif chunk_type == "content_block_stop":
                    if current_tool_use:
                        result = await self.execute_tool(
                            current_tool_use["name"],
                            current_tool_use["input"],
                            tool_handlers,
                        )
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": current_tool_use["id"],
                            "content": result.get("content", []),
                            "is_error": result.get("is_error", False),
                        })
                        
                        yield {
                            "type": "tool_result",
                            "name": current_tool_use["name"],
                            "output": result,
                        }
                        
                        current_tool_use = None
                
                elif chunk_type == "message_stop":
                    if has_tool_use and tool_results:
                        self.conversation_history.append({
                            "role": "user",
                            "content": tool_results,
                        })
                    else:
                        return
"""
Bedrock-based agent implementation that mimics Claude Agent SDK interface.
This allows using Amazon Bedrock instead of direct Anthropic API.
"""

import json
import logging
from typing import Any, AsyncIterator, Optional

from .bedrock_client import BedrockClaudeClient
from .config import config

logger = logging.getLogger(__name__)


class BedrockAgent:
    """
    Agent implementation using Amazon Bedrock for Claude.
    Provides similar interface to Claude Agent SDK but uses Bedrock.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ):
        """
        Initialize Bedrock agent.
        
        Args:
            system_prompt: System prompt for the agent
            tools: List of tool definitions in Anthropic format
            model_id: Bedrock model ID
        """
        self.system_prompt = system_prompt
        self.tools = tools
        self.model_id = model_id
        
        self.client = BedrockClaudeClient(
            region_name=config.bedrock.region_name,
            model_id=model_id,
            aws_access_key_id=config.bedrock.aws_access_key_id,
            aws_secret_access_key=config.bedrock.aws_secret_access_key,
            aws_session_token=config.bedrock.aws_session_token,
        )
        
        self.conversation_history: list[dict[str, Any]] = []

    async def connect(self):
        """Connect to Bedrock (no-op, connection is per-request)."""
        pass

    async def disconnect(self):
        """Disconnect from Bedrock (no-op)."""
        pass

    async def query(
        self,
        message: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """
        Send a query to Claude via Bedrock.
        
        Args:
            message: User message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response dict with text and tool calls
        """
        messages = self.conversation_history + [
            {"role": "user", "content": message}
        ]

        response = self.client.invoke(
            messages=messages,
            system=self.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=self.tools if self.tools else None,
        )

        self.conversation_history.append({"role": "user", "content": message})
        
        if response.get("content"):
            self.conversation_history.append({
                "role": "assistant",
                "content": response["content"]
            })

        return response

    async def query_stream(
        self,
        message: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Send a query to Claude via Bedrock with streaming.
        
        Yields response chunks as they arrive.
        """
        messages = self.conversation_history + [
            {"role": "user", "content": message}
        ]

        self.conversation_history.append({"role": "user", "content": message})
        
        accumulated_content = []

        async for chunk in self.client.invoke_stream(
            messages=messages,
            system=self.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=self.tools if self.tools else None,
        ):
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    accumulated_content.append({
                        "type": "text",
                        "text": delta.get("text", "")
                    })
                    yield chunk
            elif chunk.get("type") == "content_block_start":
                content_block = chunk.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    accumulated_content.append(content_block)
                yield chunk
            else:
                yield chunk

        if accumulated_content:
            self.conversation_history.append({
                "role": "assistant",
                "content": accumulated_content
            })

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_handlers: dict[str, callable],
    ) -> dict[str, Any]:
        """
        Execute a tool and return the result.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            tool_handlers: Dict mapping tool names to handler functions
            
        Returns:
            Tool execution result
        """
        if tool_name not in tool_handlers:
            return {
                "content": [{"type": "text", "text": f"Tool {tool_name} not found"}],
                "is_error": True,
            }

        try:
            handler = tool_handlers[tool_name]
            result = await handler(tool_input)
            return result
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return {
                "content": [{"type": "text", "text": f"Error executing {tool_name}: {str(e)}"}],
                "is_error": True,
            }

    async def run_agentic_loop(
        self,
        message: str,
        tool_handlers: dict[str, callable],
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Run the agentic loop: query -> tool use -> tool result -> repeat.
        
        Args:
            message: Initial user message
            tool_handlers: Dict mapping tool names to handler functions
            max_iterations: Maximum number of iterations
            
        Returns:
            Final response with text and tool calls
        """
        response_text = ""
        tool_calls = []
        
        for iteration in range(max_iterations):
            response = await self.query(message if iteration == 0 else "")
            
            stop_reason = response.get("stop_reason")
            content = response.get("content", [])
            
            has_tool_use = False
            tool_results = []
            
            for block in content:
                if block.get("type") == "text":
                    response_text += block.get("text", "")
                elif block.get("type") == "tool_use":
                    has_tool_use = True
                    tool_name = block.get("name")
                    tool_input = block.get("input", {})
                    tool_use_id = block.get("id")
                    
                    tool_calls.append({
                        "name": tool_name,
                        "input": tool_input,
                    })
                    
                    result = await self.execute_tool(
                        tool_name,
                        tool_input,
                        tool_handlers,
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result.get("content", []),
                        "is_error": result.get("is_error", False),
                    })
            
            if has_tool_use and tool_results:
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results,
                })
            elif stop_reason == "end_turn":
                break
        
        return {
            "response": response_text,
            "tool_calls": tool_calls,
        }

    async def run_agentic_loop_stream(
        self,
        message: str,
        tool_handlers: dict[str, callable],
        max_iterations: int = 10,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Run the agentic loop with streaming responses.
        
        Yields events as they occur.
        """
        for iteration in range(max_iterations):
            current_tool_use = None
            tool_results = []
            has_tool_use = False
            
            async for chunk in self.query_stream(message if iteration == 0 else ""):
                chunk_type = chunk.get("type")
                
                if chunk_type == "content_block_start":
                    content_block = chunk.get("content_block", {})
                    if content_block.get("type") == "tool_use":
                        current_tool_use = {
                            "id": content_block.get("id"),
                            "name": content_block.get("name"),
                            "input": {},
                        }
                        has_tool_use = True
                        yield {
                            "type": "tool_use",
                            "name": current_tool_use["name"],
                            "input": {},
                        }
                
                elif chunk_type == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield {
                            "type": "text",
                            "content": delta.get("text", ""),
                        }
                    elif delta.get("type") == "input_json_delta" and current_tool_use:
                        partial_json = delta.get("partial_json", "")
                        if partial_json:
                            try:
                                current_tool_use["input"] = json.loads(
                                    json.dumps(current_tool_use.get("input", {})) + partial_json
                                )
                            except:
                                pass
                
                elif chunk_type == "content_block_stop":
                    if current_tool_use:
                        result = await self.execute_tool(
                            current_tool_use["name"],
                            current_tool_use["input"],
                            tool_handlers,
                        )
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": current_tool_use["id"],
                            "content": result.get("content", []),
                            "is_error": result.get("is_error", False),
                        })
                        
                        yield {
                            "type": "tool_result",
                            "name": current_tool_use["name"],
                            "output": result,
                        }
                        
                        current_tool_use = None
                
                elif chunk_type == "message_stop":
                    if has_tool_use and tool_results:
                        self.conversation_history.append({
                            "role": "user",
                            "content": tool_results,
                        })
                    else:
                        return
