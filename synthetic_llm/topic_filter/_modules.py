from langchain_core.pydantic_v1 import BaseModel, Field
from enum import Enum

class Binary(str, Enum):
    yes = "Yes"
    no = "No"

class NodeRelevancy(BaseModel):
    """Context Relevancy to a query"""
    relevance: Binary = Field(description="Whether the context is relevant to the query")