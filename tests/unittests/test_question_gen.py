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

from custom_pydantic import QuestionList, GeneratedQuestion

load_dotenv()
test_llm = OpenAI(
    model = "gpt-4o", max_tokens = 512, temperature=0.5
)

documents = SimpleDirectoryReader(
    input_dir = os.path.join(MAIN_DIR, "tests", "sample_docs")
).load_data()

sentence_splitter = SentenceSplitter(chunk_size=512, paragraph_separator="\n\n")
nodes = sentence_splitter.get_nodes_from_documents(documents)

DEFAULT_QUESTION_GENERATION_PROMPT_ZERO_SHOT = """\
Given the context information and not prior knowledge.
generate only questions based on the below query.
{query_str}
-----------
Context:
<START OF CONTEXT>
{context_str}
</END OF CONTEXT>

Generated Questions:
-----------
"""

DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS = """\
Given the context information and not prior knowledge.
generate only questions based on the below instructions.
{query_str}
-----------
{few_shot_examples}
-----------
Context:
<START OF CONTEXT>
{context_str}
</END OF CONTEXT>

Generated Questions:
-----------
"""

QUESTION_GEN_PROMPT = (
    "You are a question generation engine. Your task is to setup {num_questions_per_chunk} "
    "questions based on the facts given inside Context. The questions should be diverse in nature "
    "across the document. Generated questions should be answerable only with reference to information given "
    "within Context. Return empty list if questions cannot be generated to fulfill above requirements."
    )

def test_generate_questions():
    
    blank_text = ""
    sample_node = random.choice(nodes)
    NO_OF_QUESTIONS = 4
    
    question_gen_query = PromptTemplate(QUESTION_GEN_PROMPT).format(num_questions_per_chunk=NO_OF_QUESTIONS)

    sample_document = Document(text=sample_node.text)
    sample_index = SummaryIndex.from_documents([sample_document])
    sample_query_engine = sample_index.as_query_engine(
        llm=test_llm,
        text_qa_template=PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT_ZERO_SHOT),
        output_cls=QuestionList
    )
    sample_response = sample_query_engine.query(question_gen_query).response
    
    blank_document = Document(text=blank_text)
    blank_index = SummaryIndex.from_documents([blank_document])
    blank_query_engine = blank_index.as_query_engine(
        llm=test_llm,
        text_qa_template=PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT_ZERO_SHOT),
        output_cls=QuestionList
    )
    blank_response = blank_query_engine.query(question_gen_query).response
         
    assert isinstance(sample_response, QuestionList), "Invalid Output Class for sample response"
    assert isinstance(blank_response, QuestionList), "Invalid Output Class for blank response"
    
    assert blank_response.question_list == []
    assert len(sample_response.question_list) == NO_OF_QUESTIONS
    for sample_question in sample_response.question_list:
        assert isinstance(sample_question, GeneratedQuestion)