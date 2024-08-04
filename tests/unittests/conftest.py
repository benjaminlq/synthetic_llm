import pytest
import os
import random

from langchain_openai import ChatOpenAI

from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv

@pytest.fixture(scope='session', autouse=True)
def server():
    
    load_dotenv()
    print(os.environ["OPENAI_API_KEY"])
    
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    test_llm = OpenAI(
        model = "gpt-4o", max_tokens = 512, temperature=0.5
    )

    structured_llm = ChatOpenAI(
        model_name="gpt-4o", temperature=0, max_tokens=128
    )

    documents = SimpleDirectoryReader(
        input_dir = os.path.join("tests", "sample_docs")
    ).load_data()

    sentence_splitter = SentenceSplitter(chunk_size=512, paragraph_separator="\n\n")
    nodes = sentence_splitter.get_nodes_from_documents(documents)
    sample_node = random.choice(nodes)
    
    return test_llm, structured_llm, sample_node, nodes