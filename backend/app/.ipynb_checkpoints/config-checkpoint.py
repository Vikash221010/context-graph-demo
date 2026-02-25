"""
Configuration management for the Context Graph application.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""

    uri: str
    username: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


@dataclass
class OpenAIConfig:
    """OpenAI configuration for text embeddings."""

    api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")),
        )


@dataclass
class AnthropicConfig:
    """Anthropic configuration for Claude Agent SDK."""

    api_key: str

    @classmethod
    def from_env(cls) -> "AnthropicConfig":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )


@dataclass
class BedrockConfig:
    """Amazon Bedrock configuration."""

    region_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str
    claude_model_id: str
    embedding_model_id: str
    embedding_dimensions: int

    @classmethod
    def from_env(cls) -> "BedrockConfig":
        return cls(
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN", ""),
            claude_model_id=os.getenv(
                "BEDROCK_CLAUDE_MODEL_ID",
                "anthropic.claude-3-5-sonnet-20241022-v2:0"
            ),
            embedding_model_id=os.getenv(
                "BEDROCK_EMBEDDING_MODEL_ID",
                "amazon.titan-embed-text-v2:0"
            ),
            embedding_dimensions=int(os.getenv("BEDROCK_EMBEDDING_DIMENSIONS", "1024")),
        )


@dataclass
class AppConfig:
    """Main application configuration."""

    neo4j: Neo4jConfig
    openai: OpenAIConfig
    anthropic: AnthropicConfig
    bedrock: BedrockConfig
    
    # Toggle between direct API and Bedrock
    use_bedrock: bool = False

    # FastRP embedding dimensions (structural)
    fastrp_dimensions: int = 128

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            neo4j=Neo4jConfig.from_env(),
            openai=OpenAIConfig.from_env(),
            anthropic=AnthropicConfig.from_env(),
            bedrock=BedrockConfig.from_env(),
            use_bedrock=os.getenv("USE_BEDROCK", "false").lower() == "true",
            fastrp_dimensions=int(os.getenv("FASTRP_DIMENSIONS", "128")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


# Global config instance
config = AppConfig.from_env()
