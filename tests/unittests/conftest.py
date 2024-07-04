import pytest
import os
import random

from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv
from config import MAIN_DIR

@pytest.fixture(scope='session', autouse=True)
def server():
    
    load_dotenv()
    test_llm = OpenAI(
        model = "gpt-4o", max_tokens = 512, temperature=0.5
    )

    documents = SimpleDirectoryReader(
        input_dir = os.path.join(MAIN_DIR, "tests", "sample_docs")
    ).load_data()

    sentence_splitter = SentenceSplitter(chunk_size=512, paragraph_separator="\n\n")
    nodes = sentence_splitter.get_nodes_from_documents(documents)
    sample_node = random.choice(nodes)
    
    return test_llm, sample_node, nodes