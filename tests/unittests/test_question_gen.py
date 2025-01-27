from llama_index.core.prompts import PromptTemplate
from llama_index.core.indices import SummaryIndex
from llama_index.core.schema import Document

from synthetic_llm.llama_index.llama_index_generator import (
    QUESTION_GEN_PROMPT,
    DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS,
    ) 

from synthetic_llm.node_filter import filter_relevant_nodes_by_topic
from synthetic_llm.llama_index._types import QuestionList, GeneratedQuestion, RagDataExampleWithMetadata
from synthetic_llm.llama_index import CustomRAGDatasetGenerator
from synthetic_llm.llama_index.rag_few_shot import convert_examples_to_string

BLANK_TEXT = ""
NO_OF_QUESTIONS = 4
QUESTION_GEN_QUERY = PromptTemplate(QUESTION_GEN_PROMPT).format(num_questions_per_chunk=NO_OF_QUESTIONS)

def test_generate_questions(server):
    
    test_llm, _, sample_node, _ = server
    
    sample_document = Document(text=sample_node.text)
    sample_index = SummaryIndex.from_documents([sample_document])
    sample_query_engine = sample_index.as_query_engine(
        llm=test_llm,
        text_qa_template=PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS).partial_format(few_shot_examples=BLANK_TEXT),
        output_cls=QuestionList
    )
    
    sample_response = sample_query_engine.query(QUESTION_GEN_QUERY).response
    
    blank_document = Document(text=BLANK_TEXT)
    blank_index = SummaryIndex.from_documents([blank_document])
    blank_query_engine = blank_index.as_query_engine(
        llm=test_llm,
        text_qa_template=PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS).partial_format(few_shot_examples=BLANK_TEXT),
        output_cls=QuestionList
    )
    blank_response = blank_query_engine.query(QUESTION_GEN_QUERY).response
         
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

def test_question_generator(server):
    
    test_llm, _, _, nodes = server
    
    examples = [
        RagDataExampleWithMetadata(
            query="Photosynthesis in plants involves converting light energy into chemical energy, using chlorophyll and other pigments to absorb light. This process is crucial for plant growth and the production of oxygen.",
            reference_answer=QuestionList(
                question_list = [
                    GeneratedQuestion(question = "What is the role of photosynthesis in plant growth?"),
                    GeneratedQuestion(question = "What substance does photosynthesis produce?"),
                    GeneratedQuestion(question = "What energy conversion takes place in photosynthesis?")
                ]).json()
        ),
        RagDataExampleWithMetadata(
            query="The Industrial Revolution, starting in the 18th century, marked a major turning point in history as it led to the development of factories and urbanization.",
            reference_answer=QuestionList(
                question_list = [
                    GeneratedQuestion(question = "How did the Industrial Revolution mark a major turning point in history?"),
                    GeneratedQuestion(question = "When did the Industrial Revolution start?")
                ]).json()
        )
    ]
    
    question_generator = CustomRAGDatasetGenerator(
        nodes=nodes[:4],
        llm=test_llm,
        num_questions_per_chunk=5,
        maximum_source_nodes=2,
        n_shots=2,
        use_function_calling=False,
        few_shot_examples=examples
    )
    
    _ = question_generator.generate_dataset_from_nodes(
        use_examples=True,
        reset_examples=True,
        add_generated_data_as_examples=True,
        iterations=3
    )

def test_question_generator_function_calling(server):
    test_llm, _, _, nodes = server

    examples = [
        RagDataExampleWithMetadata(
            query="Photosynthesis in plants involves converting light energy into chemical energy, using chlorophyll and other pigments to absorb light. This process is crucial for plant growth and the production of oxygen.",
            reference_answer=QuestionList(
                question_list = [
                    GeneratedQuestion(question = "What is the role of photosynthesis in plant growth?"),
                    GeneratedQuestion(question = "What substance does photosynthesis produce?"),
                    GeneratedQuestion(question = "What energy conversion takes place in photosynthesis?")
                ]).json()
        ),
        RagDataExampleWithMetadata(
            query="The Industrial Revolution, starting in the 18th century, marked a major turning point in history as it led to the development of factories and urbanization.",
            reference_answer=QuestionList(
                question_list = [
                    GeneratedQuestion(question = "How did the Industrial Revolution mark a major turning point in history?"),
                    GeneratedQuestion(question = "When did the Industrial Revolution start?")
                ]).json()
        )
    ]

    pydantic_question_generator = CustomRAGDatasetGenerator(
        nodes=nodes[:4],
        llm=test_llm,
        num_questions_per_chunk=5,
        maximum_source_nodes=2,
        n_shots=2,
        use_function_calling=True,
        few_shot_examples=examples
    )

    _ = pydantic_question_generator.generate_dataset_from_nodes(
        use_examples = True,
        reset_examples = True,
        add_generated_data_as_examples = True,
        iterations = 3,
    )
    

    
def test_relevance_filter(server):
    
    test_llm, _, _, nodes = server
    
    pydantic_question_generator = CustomRAGDatasetGenerator(
        nodes=nodes[:4],
        llm=test_llm,
        num_questions_per_chunk=5,
        maximum_source_nodes=2,
        n_shots=2,
        use_function_calling=True
    )
    
    _ = pydantic_question_generator.generate_dataset_from_nodes(
        iterations = 3,
        llm_relevance_filter = True
    )
    
def test_topic_filter(server):
    _, structured_llm , _, nodes = server
    topic: str = "education"
    
    _ = filter_relevant_nodes_by_topic(
        topic=topic,
        nodes=nodes,
        llm=structured_llm
    )