import os
import random

from llama_index.core.prompts import PromptTemplate
from llama_index.core.indices import SummaryIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv
from config import MAIN_DIR
from prompts import (
    QUESTION_GEN_PROMPT,
    DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS,
    ) 

from custom_pydantic import QuestionList, GeneratedQuestion
from generator import RagDataExampleWithMetadata
from generator import CustomRAGDatasetGenerator
from utils import convert_examples_to_string
from datetime import datetime

load_dotenv()
test_llm = OpenAI(
    model = "gpt-3.5-turbo", max_tokens = 512, temperature=0.5
)

documents = SimpleDirectoryReader(
    input_dir = os.path.join(MAIN_DIR, "tests", "sample_docs")
).load_data()

sentence_splitter = SentenceSplitter(chunk_size=512, paragraph_separator="\n\n")
nodes = sentence_splitter.get_nodes_from_documents(documents)

def test_generate_questions():
    
    blank_text = ""
    sample_node = random.choice(nodes)
    NO_OF_QUESTIONS = 4
    
    question_gen_query = PromptTemplate(QUESTION_GEN_PROMPT).format(num_questions_per_chunk=NO_OF_QUESTIONS)

    sample_document = Document(text=sample_node.text)
    sample_index = SummaryIndex.from_documents([sample_document])
    sample_query_engine = sample_index.as_query_engine(
        llm=test_llm,
        text_qa_template=PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS).partial_format(few_shot_examples=""),
        output_cls=QuestionList
    )
    sample_response = sample_query_engine.query(question_gen_query).response
    
    blank_document = Document(text=blank_text)
    blank_index = SummaryIndex.from_documents([blank_document])
    blank_query_engine = blank_index.as_query_engine(
        llm=test_llm,
        text_qa_template=PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS).partial_format(few_shot_examples=""),
        output_cls=QuestionList
    )
    blank_response = blank_query_engine.query(question_gen_query).response
         
    assert isinstance(sample_response, QuestionList), "Invalid Output Class for sample response"
    assert isinstance(blank_response, QuestionList), "Invalid Output Class for blank response"
    
    assert blank_response.question_list == []
    assert len(sample_response.question_list) == NO_OF_QUESTIONS
    for sample_question in sample_response.question_list:
        assert isinstance(sample_question, GeneratedQuestion)
        
def test_examples():
    labelled_example_list = [
        RagDataExampleWithMetadata(query="query_1", reference_contexts=["context_1-1", "context_1-2"], reference_answer="answer_1"),
        RagDataExampleWithMetadata(query="query_3", reference_contexts=[], reference_answer="answer_3")
    ]
    
    blank_example_list = []
    
    unlabelled_example_list = [
        RagDataExampleWithMetadata(query="query_1", reference_contexts=["context_1-1", "context_1-2"]),
        RagDataExampleWithMetadata(query="query_2", reference_contexts=["context_1-1"])
    ]
    
    assert convert_examples_to_string(blank_example_list) == "", "Must be blank examples"
    _ = convert_examples_to_string(unlabelled_example_list)
    _ = convert_examples_to_string(labelled_example_list)

def test_question_generator():
    question_generator = CustomRAGDatasetGenerator(
        nodes = nodes[:4],
        llm = test_llm,
        num_questions_per_chunk = 2,
        maximum_source_nodes = 2,
        n_shots = 2,
    )
    
    _ = question_generator.generate_dataset_from_nodes(
        use_examples = True,
        reset_examples = True,
        add_generated_data_as_examples = True,
        iterations = 3
    )