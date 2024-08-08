from llama_index.core.prompts import PromptTemplate, ChatMessage, MessageRole
from llama_index.core.llama_dataset import LabelledRagDataExample
from typing import List, Dict, Union
from langchain_core.pydantic_v1 import BaseModel
from synthetic_llm.generator._types import QuestionList, GeneratedQuestion
from openai.types.chat.chat_completion_message_tool_call import Function, ChatCompletionMessageToolCall
from uuid import uuid4

def convert_examples_to_string(
    examples: List[Union[BaseModel, Dict]],
    prompt_template: str,
) -> str:
    
    example_prompt = PromptTemplate(prompt_template)
    variables = example_prompt.template_vars
    
    combined_examples_str = ""
    for example in examples:
        if isinstance(example, BaseModel):
            example = example.dict()
            
        # Check that example has all the required prompt placeholders
        for variable in variables:
            assert variable in example, f"{variable} not in example"
        
        example_str = example_prompt.format(**example)
        combined_examples_str += (example_str + "\n\n")

    return ("EXAMPLES:\n" + combined_examples_str).strip() if combined_examples_str else ""   

def convert_examples_to_chat_messages(
    examples: List[Union[BaseModel, Dict]],
    user_template: str,
    answer_cls: BaseModel,
    answer_key: str = "answer"
) -> List[ChatMessage]:
    
    example_messages = []
    user_prompt = PromptTemplate(user_template)
    variables = user_prompt.template_vars
    
    for example in examples:
        if isinstance(example, BaseModel):
            example = example.dict()
             
        # Check that example has all the required prompt placeholders
        for variable in variables:
            assert variable in example, f"{variable} not in example"
        assert answer_key in example, f"{answer_key} not in example"
        
        example_messages.append(
            ChatMessage(content=user_prompt.format(**example), role=MessageRole.USER)
        )
        
        tool_call_id = str(uuid4())
        tool_name = answer_cls.__name__
        
        example_messages.extend(
            [
                ChatMessage(
                    content = "", role = MessageRole.ASSISTANT,
                    additional_kwargs={
                        'tool_calls': [ChatCompletionMessageToolCall(id = tool_call_id, function=Function(arguments=str(example[answer_key]),name=tool_name),
                                                                    type="function")]}),
                ChatMessage(
                    content = "The generated information from tool calling is correct.",
                    role = MessageRole.TOOL,
                    additional_kwargs={"name": tool_name, "tool_call_id": tool_call_id})
            ]
        )

    return example_messages