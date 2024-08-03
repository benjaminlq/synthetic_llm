from ._modules import NodeRelevancy
from typing import Optional, Sequence, List

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode

from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

import numpy as np
import json

FUNCTION_CALLING_SYSTEM_PROMPT = (
    "You are given a QUERY and a CONTEXT document. Your task is to evaluate if the extracted information "
    "under context is relevant to the topic of the query. Keep your answer concise"
    )

DEFAULT_LLM = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=128)

def load_relevant_nodes_from_storage(
    file_directory: str
    ) -> List[NodeWithScore]:
        
    with open(file_directory, 'r') as f:
        filtered_nodes_dict = json.load(f)

    filtered_nodes = []
    for node_dict in filtered_nodes_dict:
        text_node = TextNode.from_dict(node_dict["node"])
        score = node_dict["score"]
        
        filtered_nodes.append(NodeWithScore(node=text_node, score=score))
    return filtered_nodes

def filter_relevant_nodes_by_topic(
    topic: str,
    nodes: Optional[Sequence[BaseNode]],
    llm: Optional[BaseChatModel] = None,
    embed_model: Optional[BaseEmbedding] = None,
    relevancy_evaluation_prompt: Optional[str] = None,
    relevance_threshold: float = 0.1,
    no_bins: int = 10, min_no_samples_per_bin: int = 15,
    save_folder: Optional[str] = None
    ) -> List[NodeWithScore]:

    embed_model = embed_model or OpenAIEmbedding()
    llm = llm or DEFAULT_LLM
    
    _relevancy_evaluation_prompt = relevancy_evaluation_prompt or FUNCTION_CALLING_SYSTEM_PROMPT
    messages = [
        SystemMessagePromptTemplate.from_template(_relevancy_evaluation_prompt),
        HumanMessagePromptTemplate.from_template("CONTEXT: {context_str}\n\nQUERY: {query_str}")
    ]

    fn_calling_chat_messages = ChatPromptTemplate.from_messages(messages)
    relevance_chain = (
        fn_calling_chat_messages
        | llm.with_structured_output(schema=NodeRelevancy)
        | RunnableLambda(lambda x: 1 if x.relevance.value.upper() == "YES" else 0)
        )

    index = VectorStoreIndex(
        nodes=nodes, embed_model=embed_model
    )
    retriever = index.as_retriever(similarity_top_k=len(nodes), embed_model=embed_model)
    combined_nodes_with_score = retriever.retrieve(topic)

    relevance_scores_by_bin = []
    relevance_hashmap = {}

    binned_nodes_with_score = np.array_split(combined_nodes_with_score, no_bins)
    for nodes_with_score_by_bin in binned_nodes_with_score:
        no_samples_per_bin = min(len(nodes_with_score_by_bin), min_no_samples_per_bin)
        sampled_nodes = np.random.choice(nodes_with_score_by_bin, size=no_samples_per_bin, replace=False)
        total_relevance_score = 0
        for node_with_score in sampled_nodes:
            relevance_score = relevance_chain.invoke(
                {"query_str": topic, "context_str": node_with_score.text}
            )
            total_relevance_score += relevance_score
            relevance_hashmap[node_with_score.node.id_] = relevance_score
        
        relevance_scores_by_bin.append(total_relevance_score / no_samples_per_bin)

    filtered_nodes: List[NodeWithScore] = []

    for score, nodes_with_score_by_bin in zip(relevance_scores_by_bin[::-1], binned_nodes_with_score[::-1]):
        if score < relevance_threshold:
            break
        for node_with_score in nodes_with_score_by_bin:
            if node_with_score.node.id_ in relevance_hashmap:
                if relevance_hashmap[node_with_score.node.id_] == 1:
                    filtered_nodes.append(node_with_score)
            else:
                relevance_score = relevance_chain.invoke(
                    {"query_str": topic, "context_str": node_with_score.text}
                )
                relevance_hashmap[node_with_score.node.id_] = relevance_score
                if relevance_score == 1:
                    filtered_nodes.append(node_with_score)
                    
    for node in filtered_nodes:
        node.node.metadata["query_topic"] = topic
        
    if save_folder:
        filtered_nodes_dict = [node.to_dict() for node in filtered_nodes]
        with open(save_folder, 'w') as f:
            json.dump(filtered_nodes_dict, f)
        
    return filtered_nodes