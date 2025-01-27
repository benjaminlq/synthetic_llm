from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Dict
from llama_index.core.llama_dataset import LabelledRagDataExample

class GeneratedQuestion(BaseModel):
    """Generated Query based on a certain topic"""
    question: str = Field(
        default=None,
        description="question generated based on the given topic"
    )

class QuestionList(BaseModel):
    """List of Generated Queries on a certain topic"""
    question_list: Optional[List[GeneratedQuestion]] = Field(
        default_factory=list,
        description="List of generated questions on a certain topic",
    )
    
class RagDataExampleWithMetadata(LabelledRagDataExample):
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional Metadata to store with Data Examples"
    )