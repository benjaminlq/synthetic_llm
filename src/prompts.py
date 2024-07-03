DEFAULT_QUESTION_GENERATION_PROMPT_FEW_SHOTS = """\
Given the context information and not prior knowledge.
generate only questions based on the below instructions.
{query_str}
-----------
{few_shot_examples}

EXAMPLES:
Context:

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