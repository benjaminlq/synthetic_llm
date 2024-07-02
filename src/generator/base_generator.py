import re
import warnings
import random
from typing import List, Optional

from llama_index.core import Document, ServiceContext, SummaryIndex
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs, asyncio_run
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.ingestion import run_transformations
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataExample,
    LabelledRagDataset,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.node import KeywordNodePostprocessor
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT

from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    TransformComponent,
)
from llama_index.core.settings import (
    Settings,
    llm_from_settings_or_context,
    transformations_from_settings_or_context,
)

DEFAULT_QUESTION_GENERATION_PROMPT = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
generate only questions based on the below query.
{query_str}
"""

class CustomRAGDatasetGenerator(RagDatasetGenerator):
    def __init__(
        self,
        nodes: List[BaseNode],
        llm: Optional[LLM] = None,
        generation_llm: Optional[LLM] = None,
        qa_llm: Optional[LLM] = None,
        num_questions_per_chunk: int = 3,
        maximum_source_nodes: int = 1,
        n_shots: int = 0,
        examples: Optional[List[str]] = None,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        metadata_mode: MetadataMode = MetadataMode.NONE,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
        # deprecated
        service_context: Optional[ServiceContext] = None,
    ):
        """Init params."""
        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._gen_llm = generation_llm or self._llm
        self._qa_llm = qa_llm or self._llm
        
        self.num_questions_per_chunk = num_questions_per_chunk
        self._maximum_source_nodes = maximum_source_nodes
        
        self.text_question_template = text_question_template or PromptTemplate(
            DEFAULT_QUESTION_GENERATION_PROMPT
        )
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.question_gen_query = (
            question_gen_query
            or f"You are a Teacher/Professor. Your task is to setup {num_questions_per_chunk} questions for an upcoming quiz/examination. The questions should be diverse in nature across the document. Restrict the questions to the context information provided."
        )
        self.nodes = nodes
        self.examples = examples or []
        self._n_shots = n_shots
        
        self._metadata_mode = metadata_mode
        self._show_progress = show_progress
        self._workers = workers

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        llm: Optional[LLM] = None,
        generation_llm: Optional[LLM] = None,
        qa_llm: Optional[LLM] = None,
        transformations: Optional[List[TransformComponent]] = None,
        num_questions_per_chunk: int = 3,
        maximum_source_nodes: int = 1,
        n_shots: int = 0,
        examples: Optional[List[str]] = None,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
        # deprecated
        service_context: Optional[ServiceContext] = None,
    ):
        """Generate dataset from documents."""
        llm = llm or llm_from_settings_or_context(Settings, service_context)
            
        transformations = transformations or transformations_from_settings_or_context(
            Settings, service_context
        )

        nodes = run_transformations(
            documents, transformations, show_progress=show_progress
        )

        # use node postprocessor to filter nodes
        required_keywords = required_keywords or []
        exclude_keywords = exclude_keywords or []
        node_postprocessor = KeywordNodePostprocessor(
            llm=llm,
            service_context=service_context,
            required_keywords=required_keywords,
            exclude_keywords=exclude_keywords,
        )
        node_with_scores = [NodeWithScore(node=node) for node in nodes]
        node_with_scores = node_postprocessor.postprocess_nodes(node_with_scores)
        nodes = [node_with_score.node for node_with_score in node_with_scores]

        return cls(
            nodes=nodes,
            llm=llm,
            generation_llm=generation_llm,
            qa_llm=qa_llm,
            service_context=service_context,
            num_questions_per_chunk=num_questions_per_chunk,
            maximum_source_nodes=maximum_source_nodes,
            n_shots=n_shots,
            examples=examples,
            text_question_template=text_question_template,
            text_qa_template=text_qa_template,
            question_gen_query=question_gen_query,
            show_progress=show_progress,
            workers=workers,
        )

    def 

    async def _agenerate_dataset(
        self,
        nodes: List[BaseNode],
        labelled: bool = False,
        use_generated_data_as_examples: bool = False,
        size: int = 30
    ):
        query_tasks = []
        examples: List[LabelledRagDataExample] = []
        summary_indices: List[SummaryIndex] = []
        for idx in range(size):
            nodes_no = random.choice(range(1, self._maximum_source_nodes + 1))
            nodes = random.sample(self.nodes, nodes_no)
            index = SummaryIndex.from_documents(
                [
                    Document(
                        text=node.get_content(metadata_mode=self._metadata_mode),
                        metadata=node.metadata,
                        excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                        excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        relationships=node.relationships,
                    ) for node in nodes
                ]
            )
            
            query_engine = index.as_query_engine(
                llm = self._gen_llm,
                text_qa_template=self.text_question_template,
                use_async=True,
            )
            
            
            
            task = query_engine.aquery(
                self.question_gen_query
            )