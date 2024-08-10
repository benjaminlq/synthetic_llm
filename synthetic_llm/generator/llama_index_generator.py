import warnings
import random
import numpy as np
from typing import List, Optional, Literal

from llama_index.core import Document, ServiceContext, SummaryIndex
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs, asyncio_run
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.ingestion import run_transformations
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataset,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.node import KeywordNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate, ChatPromptTemplate, ChatMessage, MessageRole
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.schema import (
    BaseNode, MetadataMode, NodeWithScore, TransformComponent,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.settings import (
    Settings,
    llm_from_settings_or_context,
    transformations_from_settings_or_context,
    embed_model_from_settings_or_context
)
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.embeddings.utils import resolve_embed_model, EmbedType
from llama_index.program.openai import OpenAIPydanticProgram

from synthetic_llm.generator._types import QuestionList, RagDataExampleWithMetadata
from copy import deepcopy

from .rag_few_shot import convert_examples_to_string, convert_examples_to_chat_messages

DEFAULT_QUESTION_GENERATION_PROMPT_SYSTEM_PROMPT = """\
Given the context information and not prior knowledge generate only questions based on the below instructions.
If there are multiple contexts, try your best to generate questions which require information across all the contexts.

{query_str}
-----------
"""

DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS = DEFAULT_QUESTION_GENERATION_PROMPT_SYSTEM_PROMPT + """\
{few_shot_examples}
-----------
Context:
<START OF CONTEXT>
{context_str}
</END OF CONTEXT>

Generated Questions:
"""

QUESTION_GEN_PROMPT = (
    "You are a question generation engine. Your task is to setup up to {num_questions_per_chunk} "
    "questions based on the facts given inside Context. The questions should be diverse in nature "
    "across the document. Generated questions should be answerable only with reference to information given "
    "within Context. Return empty list if questions cannot be generated to fulfill above requirements."
    )

class CustomRAGDatasetGenerator(RagDatasetGenerator):
    def __init__(
        self,
        nodes: List[BaseNode],
        llm: Optional[LLM] = None,
        embed_model: Optional[EmbedType] = None,
        num_questions_per_chunk: int = 3,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        metadata_mode: MetadataMode = MetadataMode.NONE,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
        service_context: Optional[ServiceContext] = None,
        use_function_calling: bool = True,
        generation_llm: Optional[LLM] = None, 
        qa_llm: Optional[LLM] = None, 
        maximum_source_nodes: int = 1,
        neighbor_method: Literal["random", "nearest"] = "nearest",
        n_shots: int = 0, 
        few_shot_examples: Optional[RagDataExampleWithMetadata] = None,
        llm_relevance_filter: Optional[LLMRerank] = None
    ):  
        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._embed_model = (
            resolve_embed_model(embed_model)
            if embed_model
            else embed_model_from_settings_or_context(Settings, service_context)
        )
        self._gen_llm = generation_llm or self._llm
        self._qa_llm = qa_llm or self._llm
        
        self.num_questions_per_chunk = num_questions_per_chunk
        self._maximum_source_nodes = maximum_source_nodes
        self._neighbor_nodes = neighbor_method
        self._use_function_calling = use_function_calling
        
        if use_function_calling:
            self.text_question_template = text_question_template or ChatPromptTemplate(
                [ChatMessage(content=DEFAULT_QUESTION_GENERATION_PROMPT_SYSTEM_PROMPT, role=MessageRole.SYSTEM)]
            )
            assert isinstance(self.text_question_template, ChatPromptTemplate), "Must use Chat Sequence for few shot tool calling"

        else:
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
        self._llm_reranker = llm_relevance_filter or LLMRerank(top_n=100, llm=self._qa_llm)

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
        service_context: Optional[ServiceContext] = None,
        use_function_calling: bool = True,
        generation_llm: Optional[LLM] = None,
        qa_llm: Optional[LLM] = None,
        maximum_source_nodes: int = 1,
        n_shots: int = 0,
        few_shot_examples: Optional[List[RagDataExampleWithMetadata]] = None,
    ):
        llm = llm or llm_from_settings_or_context(Settings, service_context)
            
        transformations = transformations or transformations_from_settings_or_context(
            Settings, service_context
        )

        nodes = run_transformations(
            documents, transformations, show_progress=show_progress
        )

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
            use_function_calling=use_function_calling,
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
        llm_relevance_filter: bool = False,
        example_wrappers: str = "Context:\n<START OF CONTEXT>\n{context_str}\n</END OF CONTEXT>\n\nGenerated Questions:\n{answer}\n\n",
        chat_user_template: str = "Context:\n<START OF CONTEXT>\n{context_str}\n</END OF CONTEXT>"
    ):
        
        def adjustment_factor(occurences: int, alpha: float=0.1):
            return np.exp(-alpha * occurences)
        
        query_tasks = []
        examples: List[RagDataExampleWithMetadata] = []
        # summary_indices: List[SummaryIndex] = []
        
        occurence_list = [0] * len(nodes)
        
        # Generate node_idx for iterations
        node_indices_all_runs = []
        node_ids_all_runs = []
        id_to_idx_dict = {node.id_: idx for idx, node in enumerate(nodes)}
        nodes_retriever = VectorStoreIndex(nodes=nodes, embed_model=self._embed_model).as_retriever(similarity_top_k=self._maximum_source_nodes)
        
        for _ in range(iterations):
            nodes_no = random.choice(range(1, self._maximum_source_nodes + 1))
            scores = [adjustment_factor(occurence) for occurence in occurence_list]
            probs = np.array(scores) / np.sum(scores)
            
            if self._neighbor_nodes == "random":        
                node_indices = np.random.choice(range(len(nodes)), size=nodes_no, replace=False, p=probs)
                
            elif self._neighbor_nodes == "nearest":
                node_indices = np.random.choice(range(len(nodes)), size=1, replace=False, p=probs).tolist()
                seed_node_content = nodes[node_indices[0]].text
                nearest_node_indices = [
                    id_to_idx_dict[node.node.id_]
                    for node in nodes_retriever.retrieve(seed_node_content)[1:nodes_no]
                ]
                node_indices.extend(nearest_node_indices)
                
            else:
                raise ValueError("Invalid neighbor method")
            
            for node_idx in node_indices:
                occurence_list[node_idx] += 1
                
            node_indices_all_runs.append(node_indices)
        
        if reset_examples:
            self.reset_example_counter()
        
        for node_indices in node_indices_all_runs:
            retrieved_nodes = [nodes[node_idx] for node_idx in node_indices]
            retrieved_nodes_ids_ = [node.id_ for node in retrieved_nodes]
            node_ids_all_runs.append(retrieved_nodes_ids_)
            
            if use_examples:
                example_list = []
                examples_no = min(self._n_shots, len(self._examples_bank))
                
                if examples_no > 0:
                    example_types = [1 if example.metadata["example_type"] == "original" else 0.5 for example in self._examples_bank]
                    example_probs = np.array(example_types) / np.sum(example_types)
                    example_indices = np.random.choice(list(range(len(self._examples_bank))), size=examples_no, replace=False, p=example_probs)

                    for example_idx in example_indices:
                        example_list.append(self._examples_bank[example_idx])
                        self._examples_bank[example_idx].metadata["occurence"] += 1
                
                if self._use_function_calling:
                    few_shot_example_messages = convert_examples_to_chat_messages(example_list, user_template=chat_user_template, context_key="context_str")
                    updated_chat_messages = deepcopy(self.text_question_template.message_templates)
                    updated_chat_messages.extend(few_shot_example_messages)
                    updated_chat_messages.append(ChatMessage(content=chat_user_template, role=MessageRole.USER)) # Final Query
                    updated_text_question_template = ChatPromptTemplate(updated_chat_messages)
                    
                else:
                    few_shot_example_str = convert_examples_to_string(example_list, example_wrappers, context_key="context_str")
                    updated_text_question_template = deepcopy(self.text_question_template)
                    updated_text_question_template = updated_text_question_template.partial_format(few_shot_examples=few_shot_example_str)

            else:
                if self._use_function_calling:
                    updated_chat_messages = deepcopy(self.text_question_template.message_templates)
                    updated_chat_messages.append(ChatMessage(content=chat_user_template, role=MessageRole.USER))
                    updated_text_question_template = ChatPromptTemplate(updated_chat_messages)
                else:
                    updated_text_question_template = deepcopy(self.text_question_template)
                    updated_text_question_template = updated_text_question_template.partial_format(few_shot_examples="")
            
            program = OpenAIPydanticProgram.from_defaults(
                output_cls=QuestionList,
                prompt=updated_text_question_template,
                llm=self._gen_llm, 
                tool_choice="required",
            )

            context_str = ""
            for idx, node in enumerate(retrieved_nodes):
                context_str += "Context No {}:\n{}\n\n".format(idx + 1, node.get_content(metadata_mode=self._metadata_mode))

            task = program.acall(query_str=self.question_gen_query, context_str=context_str)
            query_tasks.append(task)
            
            # index = SummaryIndex.from_documents(
            #     [
            #         Document(
            #             text=node.get_content(metadata_mode=self._metadata_mode),
            #             metadata=node.metadata,
            #             excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            #             excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            #             relationships=node.relationships,
            #         ) for node in retrieved_nodes
            #     ]
            # )
            # summary_indices.append(index)
            
        responses = await run_jobs(query_tasks, self._show_progress, self._workers)
        
        for run_idx, (response, node_ids, node_indices) in enumerate(zip(responses, node_ids_all_runs, node_indices_all_runs)):
            
            retrieved_nodes = [nodes[node_idx] for node_idx in node_indices]
            
            question_list_str = [gen_question.question for gen_question in response.question_list]
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
                    
                    # index = summary_indices[run_idx]
                    
                    qr_tasks = []
                    
                    for query in cleaned_questions:
                        
                        # if llm_relevance_filter:
                        #     filtered_retrieved_nodes = [node_with_score.node for node_with_score in self._llm_reranker.postprocess_nodes(nodes=retrieved_nodes, query_str=query)]
                        
                        # else:
                        #     filtered_retrieved_nodes = retrieved_nodes
                        
                        # index = SummaryIndex.from_documents(
                        #     [
                        #         Document(
                        #             text=node.get_content(metadata_mode=self._metadata_mode),
                        #             metadata=node.metadata,
                        #             excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                        #             excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        #             relationships=node.relationships,
                        #         ) for node in filtered_retrieved_nodes
                        #     ]
                        # )
                        
                        # qa_query_engine = index.as_query_engine(
                        #     llm=self._qa_llm,
                        #     text_qa_template=self.text_qa_template,
                        # )
                        
                        qr_task = self._afilter_and_query(retrieved_nodes, query, llm_relevance_filter)
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
                            metadata={"reference_nodes_ids": node_ids}
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
                            metadata={"reference_nodes_ids": node_ids}
                        )
                        examples.append(example)

                if add_generated_data_as_examples:
                    added_example = deepcopy(example)
                    added_example.metadata["occurence"] = 0
                    added_example.metadata["example_type"] = "generated"
                    self._examples_bank.append(added_example)
        
        return LabelledRagDataset(examples=examples)

    async def _afilter_relevant_nodes(self, retrieved_nodes: List[BaseNode], query) -> List[BaseNode]:
        retrieved_nodes = [NodeWithScore(node=node, score=0.0) for node in retrieved_nodes]
        nodes_with_score = self._llm_reranker.postprocess_nodes(nodes=retrieved_nodes, query_str=query)
        filtered_nodes = [node_with_score.node for node_with_score in nodes_with_score]
        return filtered_nodes

    async def _afilter_and_query(self, retrieved_nodes: List[BaseNode], query, llm_relevance_filter: bool=False):
        if llm_relevance_filter:
            filtered_retrieved_nodes = await self._afilter_relevant_nodes(retrieved_nodes=retrieved_nodes, query=query)
        else:
            filtered_retrieved_nodes = retrieved_nodes
        
        index = SummaryIndex.from_documents(
            [
                Document(
                    text=node.get_content(metadata_mode=self._metadata_mode),
                    metadata=node.metadata,
                    excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                    relationships=node.relationships,
                ) for node in filtered_retrieved_nodes
            ]
        )
        
        qa_query_engine = index.as_query_engine(
            llm=self._qa_llm,
            text_qa_template=self.text_qa_template,
        )
        response = await qa_query_engine.aquery(query)
        return response

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