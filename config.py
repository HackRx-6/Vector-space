from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import ClassVar

# Load environment variables from a .env file
load_dotenv()

class AppConfig(BaseModel):
    token_limit: int = Field(650, description="Maximum number of tokens per text chunk.")
    overlap: int = Field(50, description="Number of tokens to overlap between chunks.")
    embedding_model: str = Field("text-embedding-3-small", description="The OpenAI model for creating text embeddings.")
    qa_model: str = Field("gpt-4.1", description="The OpenAI model for generating answers from context.")
    top_k: int = Field(5, description="The number of relevant chunks to retrieve for context.")
    MAX_RETRIES: ClassVar[int] = 2
    RETRY_DELAY_SECONDS: ClassVar[float] = 0

# Global config object that will be used by other modules.
config = AppConfig()
