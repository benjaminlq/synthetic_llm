from llama_index.core.prompts import PromptTemplate
from generator import RagDataExampleWithMetadata
from typing import List

def convert_examples_to_string(
    examples: List[RagDataExampleWithMetadata],
    example_wrapper: str = "Context:\n{context}\nQuery:\n{query}\nAnswer:\n{answer}\n\n",
    query_key: str = "query",
    context_key: str = "context",
    answer_key: str = "answer"
) -> str:
    example_prompt = PromptTemplate(example_wrapper)
    variables = example_prompt.template_vars
    
    combined_examples_str = ""
    for example in examples:
        variable_mapping = {
            query_key: example.query
        }
        
        combined_contexts_str = ""
        if example.reference_contexts:
            for context_idx, context_content in enumerate(example.reference_contexts):
                context_str = "Context No {context_idx}:\n{context_content}\n".format(
                    context_idx = context_idx,
                    context_content = context_content
                )
                combined_contexts_str += context_str
            
        if context_key in variables:
            variable_mapping[context_key] = combined_contexts_str 
            
        if answer_key in variables:
            variable_mapping[answer_key] = example.reference_answer

        example_str = example_prompt.format(**variable_mapping)
        combined_examples_str += example_str

    return ("EXAMPLES:\n" + combined_examples_str).strip() if combined_examples_str else ""