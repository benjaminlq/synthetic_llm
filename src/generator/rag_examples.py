from llama_index.core.llama_dataset import LabelledRagDataExample
from llama_index.core.bridge.pydantic import Field
from typing import Dict

class RagDataExampleWithMetadata(LabelledRagDataExample):
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional Metadata to store with Data Examples"
    )