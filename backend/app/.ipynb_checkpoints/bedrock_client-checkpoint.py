"""
Amazon Bedrock client for Claude and embeddings.
Replaces direct Anthropic and OpenAI API calls with Bedrock.
"""

import json
import logging
from typing import Any, AsyncIterator, Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)


class BedrockClaudeClient:
    """Client for invoking Claude via Amazon Bedrock."""

    def __init__(
        self,
        region_name: str = "us-east-1",
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize Bedrock Claude client.
        
        Args:
            region_name: AWS region where Bedrock is available
            model_id: Bedrock model ID for Claude
            aws_access_key_id: AWS access key (optional, uses default credentials if not provided)
            aws_secret_access_key: AWS secret key
            aws_session_token: AWS session token (for temporary credentials)
        """
        self.model_id = model_id
        self.region_name = region_name

        config = Config(
            region_name=region_name,
            read_timeout=300,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        if aws_access_key_id and aws_secret_access_key:
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                config=config,
            )
        else:
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name,
                config=config,
            )

    def invoke(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """
        Invoke Claude via Bedrock (non-streaming).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: List of tool definitions
            
        Returns:
            Response dict with Claude's output
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            body["system"] = system

        if tools:
            body["tools"] = tools

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
            )

            response_body = json.loads(response["body"].read())
            return response_body

        except Exception as e:
            logger.error(f"Bedrock invocation error: {e}")
            raise

    async def invoke_stream(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Invoke Claude via Bedrock with streaming.
        
        Yields response chunks as they arrive.
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            body["system"] = system

        if tools:
            body["tools"] = tools

        try:
            response = self.bedrock_runtime.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body),
            )

            stream = response.get("body")
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_data = json.loads(chunk.get("bytes").decode())
                        yield chunk_data

        except Exception as e:
            logger.error(f"Bedrock streaming error: {e}")
            raise


class BedrockEmbeddingsClient:
    """Client for generating embeddings via Amazon Bedrock."""

    def __init__(
        self,
        region_name: str = "us-east-1",
        model_id: str = "amazon.titan-embed-text-v2:0",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize Bedrock embeddings client.
        
        Args:
            region_name: AWS region where Bedrock is available
            model_id: Bedrock model ID for embeddings
                     Options: 
                     - amazon.titan-embed-text-v2:0 (1024 dims, recommended)
                     - cohere.embed-english-v3 (1024 dims)
                     - cohere.embed-multilingual-v3 (1024 dims)
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
            aws_session_token: AWS session token
        """
        self.model_id = model_id
        self.region_name = region_name

        config = Config(
            region_name=region_name,
            read_timeout=60,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        if aws_access_key_id and aws_secret_access_key:
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                config=config,
            )
        else:
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name,
                config=config,
            )

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if "titan" in self.model_id.lower():
            body = json.dumps({"inputText": text})
        elif "cohere" in self.model_id.lower():
            body = json.dumps({
                "texts": [text],
                "input_type": "search_document",
            })
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_id}")

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
            )

            response_body = json.loads(response["body"].read())

            if "titan" in self.model_id.lower():
                return response_body["embedding"]
            elif "cohere" in self.model_id.lower():
                return response_body["embeddings"][0]

        except Exception as e:
            logger.error(f"Bedrock embedding error: {e}")
            raise

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if "cohere" in self.model_id.lower():
            body = json.dumps({
                "texts": texts,
                "input_type": "search_document",
            })

            try:
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=body,
                )
                response_body = json.loads(response["body"].read())
                return response_body["embeddings"]

            except Exception as e:
                logger.error(f"Bedrock batch embedding error: {e}")
                raise
        else:
            return [self.generate_embedding(text) for text in texts]
