# Synthetic Data Generation
Synthetic Data Generation using Large Language Model for Fine Tuning and Retrieval Augmented Generation tasks

## Installation
```
git clone https://github.com/benjaminlq/synthetic_llm.git
cd synthetic_llm
pip install -e .
```

## Usage

### Setup API environment
```
import os
import openai

os.environ["OPENAI_API_KEY"] = <YOUR_API_KEY>
openai.api_key = os.environ["OPENAI_API_KEY"]
```

### Ingest Nodes
```
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

reader = SimpleDirectoryReader(input_dir="path/to/directory")
documents = reader.load_data()

node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)
```

### Filter Nodes by your topic
This function uses embeddings and LLM Judge to filter relevant nodes from the nodes library
```
from synthetic_llm.node_filter import filter_relevant_nodes_by_topic
from langchain_openai import ChatOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

structured_extraction_llm = ChatOpenAI(model_name="gpt-4o, temperature=0, max_tokens=64)
embed_model = OpenAIEmbedding() 

topic: str = "interested topic"
nodes_by_topic = filter_relevant_nodes_by_topic(
  topic, nodes,
  llm=structured_extraction_llm,
  embed_model=embed_model,
  no_bins=10, min_no_samples_per_bin=15
)
```

### Generate Synthetic RAG dataset
```
from synthetic_llm.generator import CustomRAGDatasetGenerator
from llama_index.llms.openai import OpenAI
from llama_index.core.llama_dataset import LabelledRagDataset

generator_llm = OpenAI(model = "gpt-4o", max_tokens = 1024, temperature=0.7)

question_generator = CustomRAGDatasetGenerator(
    nodes = nodes_by_topic,
    llm = generator_llm,
    num_questions_per_chunk = 2,
    maximum_source_nodes = 3,
    n_shots = 2,
    use_function_calling=True
)

labelled_rag_dataset: LabelledRagDataset = question_generator.generate_dataset_from_nodes(
                            use_examples = True,
                            reset_examples = True,
                            add_generated_data_as_examples = True,
                            iterations = 50
                          )

```
