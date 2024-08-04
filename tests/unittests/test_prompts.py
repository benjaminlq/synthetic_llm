from openai.types.chat.chat_completion_message_tool_call import Function, ChatCompletionMessageToolCall
from llama_index.core.prompts import ChatMessage, MessageRole, ChatPromptTemplate, PromptTemplate
from uuid import uuid4

from synthetic_llm.generator._modules import QuestionList, GeneratedQuestion
from synthetic_llm.generator.llama_index_generator import QUESTION_GEN_PROMPT 
from synthetic_llm.utils import format_context

NO_OF_QUESTIONS = 4
QUESTION_GEN_QUERY = PromptTemplate(QUESTION_GEN_PROMPT).format(num_questions_per_chunk=NO_OF_QUESTIONS)

def test_few_shot_tool_calling(server):
    
    test_llm, _, sample_node, _ = server
    
    SYSTEM_MESSAGE = "Given the context information and not prior knowledge, generate only questions based on the below instructions.\n{query_str}\n-----"
    
    messages = [
        ChatMessage(content=SYSTEM_MESSAGE.format(query_str = QUESTION_GEN_QUERY), role=MessageRole.SYSTEM)
    ]
    
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

    tool_name = QuestionList.__name__

    for example in example_list:
        tool_call_id = str(uuid4())
        reference_contexts = example["reference_contexts"]
        reference_answers = QuestionList(question_list=[GeneratedQuestion(question=question) for question in example["reference_answers"]])
        reference_answers_str = str(reference_answers.dict()).replace('\'', '"')
        messages.extend(
            [
                ChatMessage(
                    content= f"Context:\n<START OF CONTEXT>\n{format_context(reference_contexts)}\n</END OF CONTEXT>",
                    role = MessageRole.USER),
                ChatMessage(
                    content = "",
                    role = MessageRole.ASSISTANT,
                    additional_kwargs={
                        'tool_calls': 
                            [ChatCompletionMessageToolCall(
                                id = tool_call_id,
                                function=Function(arguments=reference_answers_str,name=tool_name),
                                type="function")]
                            }),
                ChatMessage(
                    content = "The generated information from tool calling is correct.",
                    role = MessageRole.TOOL,
                    additional_kwargs={
                        "name": tool_name,
                        "tool_call_id": tool_call_id,
                    }
                )
            ]
        )
    
        
    messages.append(
        ChatMessage(content= "Context:\n<START OF CONTEXT>\n{context_str}\n</END OF CONTEXT>", role = MessageRole.USER)
    )

    chat_prompt = ChatPromptTemplate(messages)
    
    structured_response = test_llm.structured_predict(
        output_cls=QuestionList,
        prompt=chat_prompt,
        context_str=sample_node.text
    )
    
    assert isinstance(structured_response, QuestionList), "Response must be a QuestionList Object"