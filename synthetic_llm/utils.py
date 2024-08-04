from llama_index.core.prompts import PromptTemplate, ChatMessage, MessageRole
from llama_index.core.llama_dataset import LabelledRagDataExample
from typing import List
from pydantic import BaseModel
from synthetic_llm.generator._modules import QuestionList, GeneratedQuestion
from openai.types.chat.chat_completion_message_tool_call import Function, ChatCompletionMessageToolCall
from uuid import uuid4

def format_context(
    reference_contexts: List[str] = []
) -> str:
    combined_contexts_str = ""
    for context_idx, context_content in enumerate(reference_contexts):
        context_str = "Context No {context_idx}:\n{context_content}\n".format(
            context_idx = context_idx,
            context_content = context_content
        )
        combined_contexts_str += context_str
    return combined_contexts_str

def convert_examples_to_string(
    examples: List[LabelledRagDataExample],
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
        
        combined_contexts_str = format_context(example.reference_contexts) if example.reference_contexts else ""
            
        if context_key in variables:
            variable_mapping[context_key] = combined_contexts_str 
            
        if answer_key in variables:
            variable_mapping[answer_key] = example.reference_answer

        example_str = example_prompt.format(**variable_mapping)
        combined_examples_str += example_str

    return ("EXAMPLES:\n" + combined_examples_str).strip() if combined_examples_str else ""   

def convert_examples_to_chat_messages(
    examples: List[LabelledRagDataExample],
    user_template: str = "Context:\n{context}\nQuery:\n{query}\n",
    query_key: str = "query",
    context_key: str = "context",
    answer_cls: BaseModel = QuestionList
) -> str:
    example_messages = []
    user_prompt = PromptTemplate(user_template)
    variables = user_prompt.template_vars
    
    for example in examples:
        variable_mapping = {query_key: example.query}    
        combined_contexts_str = format_context(example.reference_contexts) if example.reference_contexts else ""
            
        if context_key in variables:
            variable_mapping[context_key] = combined_contexts_str 
            
        example_messages.append(
            ChatMessage(content=user_prompt.format(**variable_mapping), role=MessageRole.USER)
        )
        
        tool_call_id = str(uuid4())
        tool_name = answer_cls.__name__
        
        example_messages.extend(
            [
                ChatMessage(
                    content = "", role = MessageRole.ASSISTANT,
                    additional_kwargs={
                        'tool_calls': [ChatCompletionMessageToolCall(id = tool_call_id, function=Function(arguments=example.reference_answer,name=tool_name),
                                                                    type="function")]}),
                ChatMessage(
                    content = "The generated information from tool calling is correct.",
                    role = MessageRole.TOOL,
                    additional_kwargs={"name": tool_name, "tool_call_id": tool_call_id})
            ]
        )

    return example_messages

if __name__ == "__main__":

    from synthetic_llm.generator._modules import RagDataExampleWithMetadata
    example_list = [
        {
            "reference_contexts": ["Andy has 10 apples", "Bob has 3 apples"],
            "reference_answers": [
                "How many apples do Andy and Bob have in total?",
                "How many apples does Andy have more than Bob?",
                "Does Andy have more apples than Bob?",
                "Do both Andy and Bob have the same number of apples?"
            ]
        },
        {
            "reference_contexts": [],
            "reference_answers": []
        }
    ]
    
    examples = []
    for example in example_list:
        reference_answer_str = str(
            QuestionList(
                question_list=[GeneratedQuestion(question=question) for question in example["reference_answers"]]
                ).dict()
            ).replace('\'', '"')
        examples.append(RagDataExampleWithMetadata(reference_contexts=example["reference_contexts"], reference_answer=reference_answer_str))
    
    messages = convert_examples_to_chat_messages(
        examples,
        user_template="Context:\n<START OF CONTEXT>\n{context}\n</END OF CONTEXT>"
    )
    
    for message in messages:
        print([message])
        print("----------")