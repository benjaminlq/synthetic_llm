import re
import warnings
import random
import numpy as np
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
    BaseNode, MetadataMode, NodeWithScore, TransformComponent,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.settings import (
    Settings,
    llm_from_settings_or_context,
    transformations_from_settings_or_context,
)
from custom_pydantic import QuestionList
from prompts import (
    DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS,
    QUESTION_GEN_PROMPT
)

from copy import deepcopy

from generator import RagDataExampleWithMetadata
from utils import convert_examples_to_string

class CustomRAGDatasetGenerator(RagDatasetGenerator):
    def __init__(
        self,
        nodes: List[BaseNode],
        llm: Optional[LLM] = None,
        num_questions_per_chunk: int = 3,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        metadata_mode: MetadataMode = MetadataMode.NONE,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
        
        # deprecated
        service_context: Optional[ServiceContext] = None,
        
        # Added
        generation_llm: Optional[LLM] = None, 
        qa_llm: Optional[LLM] = None, 
        maximum_source_nodes: int = 1, 
        n_shots: int = 0, 
        few_shot_examples: Optional[RagDataExampleWithMetadata] = None,
    ):
        """Init params."""
        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._gen_llm = generation_llm or self._llm
        self._qa_llm = qa_llm or self._llm
        
        self.num_questions_per_chunk = num_questions_per_chunk
        self._maximum_source_nodes = maximum_source_nodes
        
        self.text_question_template = text_question_template or PromptTemplate(
            DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS
        )
        
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.question_gen_query = (
            question_gen_query
            or PromptTemplate(QUESTION_GEN_PROMPT).format(num_questions_per_chunk=num_questions_per_chunk)
        )
        self.nodes = nodes
        self._examples_bank = deepcopy(few_shot_examples) or []
        for example in self._examples_bank:
            example.metadata["occurence"] = 0
            example.metadata["example_type"] = "original"
            
        self._n_shots = n_shots
        
        self._metadata_mode = metadata_mode
        self._show_progress = show_progress
        self._workers = workers

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        llm: Optional[LLM] = None,
        transformations: Optional[List[TransformComponent]] = None,
        num_questions_per_chunk: int = 3,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
        
        # deprecated
        service_context: Optional[ServiceContext] = None,
        
        # added
        generation_llm: Optional[LLM] = None,
        qa_llm: Optional[LLM] = None,
        maximum_source_nodes: int = 1,
        n_shots: int = 0,
        few_shot_examples: Optional[List[RagDataExampleWithMetadata]] = None,
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
            few_shot_examples=few_shot_examples,
            text_question_template=text_question_template,
            text_qa_template=text_qa_template,
            question_gen_query=question_gen_query,
            show_progress=show_progress,
            workers=workers,
        )
        
    def reset_example_counter(self):
        for example in self._examples_bank:
            example.metadata["occurence"] = 0
        
    async def _agenerate_dataset(
        self,
        nodes: List[BaseNode],
        labelled: bool = False,
        use_examples: bool = False,
        reset_examples: bool = True,
        add_generated_data_as_examples: bool = False,
        iterations: int = 50,
        example_wrappers: str = "Context:\n<START OF CONTEXT>\n{context}\n</END OF CONTEXT>\n\nGenerated Questions:\n{answer}\n\n",
    ):
        
        def adjustment_factor(occurences: int, alpha: float=0.1):
            return np.exp(-alpha * occurences)
        
        query_tasks = []
        examples: List[LabelledRagDataExample] = []
        summary_indices: List[SummaryIndex] = []
        
        occurence_list = [0] * len(nodes)
        
        # Generate idx for iterations
        node_indices_all_runs = []
        
        for _ in range(iterations):
            nodes_no = random.choice(range(1, self._maximum_source_nodes + 1))
            scores = [adjustment_factor(occurence) for occurence in occurence_list]
            probs = np.array(scores) / np.sum(scores)
            node_indices = np.random.choice(range(len(nodes)), size=nodes_no, replace=False, p=probs)
            
            for node_idx in node_indices:
                occurence_list[node_idx] += 1
                
            node_indices_all_runs.append(node_indices)
        
        if reset_examples:
            self.reset_example_counter()
        
        for node_indices in node_indices_all_runs:
            retrieved_nodes = [nodes[node_idx] for node_idx in node_indices]
            index = SummaryIndex.from_documents(
                [
                    Document(
                        text=node.get_content(metadata_mode=self._metadata_mode),
                        metadata=node.metadata,
                        excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                        excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        relationships=node.relationships,
                    ) for node in retrieved_nodes
                ]
            )
            
            if use_examples:
                example_list = []
                examples_no = min(self._n_shots, len(self._examples_bank))
                
                # Example probs
                if examples_no > 0:
                    example_types = [1 if example.metadata["example_type"] == "original" else 0.5 for example in self._examples_bank]
                    print(example_types)
                    example_probs = np.array(example_types) / np.sum(example_types)
                    print(example_probs)
                    example_indices = np.random.choice(list(range(len(self._examples_bank))), size=examples_no, replace=False, p=example_probs)
                    
                    example_indices = random.sample(list(range(len(self._examples_bank))), examples_no)
                    for example_idx in example_indices:
                        example_list.append(self._examples_bank[example_idx])
                        self._examples_bank[example_idx].metadata["occurence"] += 1
                    
                few_shot_example_str = convert_examples_to_string(
                    example_list, example_wrappers)
            
            query_engine = index.as_query_engine(
                llm = self._gen_llm,
                text_qa_template=self.text_question_template.partial_format(few_shot_examples=few_shot_example_str),
                output_cls=QuestionList,
                use_async=True
            )
            task = query_engine.aquery(
                self.question_gen_query
            )
            query_tasks.append(task)
            summary_indices.append(index)
            
        responses = await run_jobs(query_tasks, self._show_progress, self._workers)
        for run_idx, response in enumerate(responses):
            question_list_str = [gen_question.question for gen_question in response.response.question_list]
            cleaned_questions = question_list_str[: self.num_questions_per_chunk]
            num_questions_generated = len(cleaned_questions)
            if num_questions_generated < self.num_questions_per_chunk:
                warnings.warn(
                    f"Fewer questions generated ({num_questions_generated}) "
                    f"than requested ({self.num_questions_per_chunk})."
                )

            if num_questions_generated > 0:
                
                reference_contexts = [nodes[node_idx].text for node_idx in node_indices_all_runs[run_idx]]
                model_name = self._gen_llm.metadata.model_name
                created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)
                if labelled:
                    index = summary_indices[run_idx]
                    qr_tasks = []
                    for query in cleaned_questions:
                        qa_query_engine = index.as_query_engine(
                            llm=self._qa_llm,
                            text_qa_template=self.text_qa_template,
                        )
                        qr_task = qa_query_engine.aquery(query)
                        qr_tasks.append(qr_task)
                    
                    answer_responses: List[RESPONSE_TYPE] = await run_jobs(
                        qr_tasks, self._show_progress, self._workers
                    )
                    for question, answer_response in zip(cleaned_questions, answer_responses):
                        example = RagDataExampleWithMetadata(
                            query=question,
                            reference_answer=str(answer_response),
                            reference_contexts=reference_contexts,
                            reference_answer_by=created_by,
                            query_by=created_by,
                        )
                        examples.append(example)
                        
                else:
                    for question in cleaned_questions:
                        example = RagDataExampleWithMetadata(
                            query=question,
                            reference_answer="",
                            reference_contexts=reference_contexts,
                            reference_answer_by=created_by,
                            query_by=created_by,
                        )
                        examples.append(example)

                if add_generated_data_as_examples:
                    added_example = deepcopy(example)
                    added_example.metadata["occurence"] = 0
                    added_example.metadata["example_type"] = "generated"
                    self._examples_bank.append(added_example)
        
        return LabelledRagDataset(examples=examples)
    
    async def agenerate_questions_from_nodes(self, **kwargs) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return await self._agenerate_dataset(self.nodes, labelled=False, **kwargs)

    async def agenerate_dataset_from_nodes(self,  **kwargs) -> LabelledRagDataset:
        """Generates questions for each document."""
        return await self._agenerate_dataset(self.nodes, labelled=True, **kwargs)

    def generate_questions_from_nodes(self, **kwargs) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return asyncio_run(self.agenerate_questions_from_nodes(**kwargs))

    def generate_dataset_from_nodes(self, **kwargs) -> LabelledRagDataset:
        """Generates questions for each document."""
        return asyncio_run(self.agenerate_dataset_from_nodes(**kwargs))