# LlamaIndex Synthetic Data Generation

## RAG Dataset Class

* RagExamplePrediction
```
class RagExamplePrediction(BaseLlamaExamplePrediction):

    # Key Attributes
    response: str 
    contexts: Optional[List[str]] 
```

* LabelledRagDataExample
```
class LabelledRagDataExample(BaseLlamaDataExample):

    # Key Attributes

    query: str
    reference_contexts: Optional[List[str]]
    reference_answer: str
```

* LabelledRagDataset
```
class LabelledRagDataset(BaseLlamaDataset[BaseQueryEngine]):
    # Key Methods:
    def _predict_example(
        self, predictor: BaseQueryEngine, example: LabelledRagDataExample, sleep_time_in_seconds: int = 0,
    ) -> RagExamplePrediction:
        """Predict RAG example with a query engine."""
        time.sleep(sleep_time_in_seconds)
        response = predictor.query(example.query)
        return RagExamplePrediction(
            response=str(response), contexts=[s.text for s in response.source_nodes]
        )

    async def _apredict_example(
        self, predictor: BaseQueryEngine, example: LabelledRagDataExample, sleep_time_in_seconds: int,
    ) -> RagExamplePrediction:
```

## RAG Dataset Generator

### Important Prompts:
* Question generation (For generating questions)
```
DEFAULT_QUESTION_GENERATION_PROMPT = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
generate only questions based on the below query.
{query_str}
"""
```

* RAG QA (For generating reference answers)
```
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
```

* Question Generation Query:
```
DEFAULT_QUESTION_GEN_QUERY = (
    "You are a Teacher/Professor. Your task is to setup "
    "{num_questions_per_chunk} questions for an upcoming quiz "
    "examination. The questions should be diverse in nature across "
    "the document. Restrict the questions to the context information "
    "provided."
)
```

### RAGDatasetGenerator Class
* Key attributes:
```
    nodes: List[BaseNode] => List of Nodes to be generated from
    llm: LLM => LLM for both all actions 
    num_questions_per_chunk: int => Number of questions to be generated per node
    text_question_template: PromptTemplate => Contextualized Question Generation Prompt
    text_qa_template: PromptTemplate => RAG Prompt for generating reference answer
    question_gen_query: str => Instruction for question generation prompt
```

* Key methods
```
    async def _agenerate_dataset(
        self,
        nodes: List[BaseNode],
        labelled: bool = False,
    ) -> LabelledRagDataset:
```

Loop through all the nodes and generate queries questions
```
        for node in nodes:
            index = SummaryIndex.from_documents(
                [
                    Document(
                        text=node.get_content(metadata_mode=self._metadata_mode),
                        metadata=node.metadata,
                        excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                        excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        relationships=node.relationships,
                    )
                ],
            )

            query_engine = index.as_query_engine(
                llm=self._llm,
                text_qa_template=self.text_question_template,
                use_async=True,
            )
            task = query_engine.aquery(
                self.question_gen_query,
            )
            query_tasks.append(task)
            summary_indices.append(index)

        responses = await run_jobs(query_tasks, self._show_progress, self._workers)
```

Clean up generated responses
```
        for idx, response in enumerate(responses):
            result = str(response).strip().split("\n")
            cleaned_questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            cleaned_questions = [
                question for question in cleaned_questions if len(question) > 0
            ][: self.num_questions_per_chunk]

            num_questions_generated = len(cleaned_questions)
            if num_questions_generated < self.num_questions_per_chunk:
                warnings.warn(
                    f"Fewer questions generated ({num_questions_generated}) "
                    f"than requested ({self.num_questions_per_chunk})."
                )

            index = summary_indices[idx]
            reference_context = nodes[idx].text
            model_name = self._llm.metadata.model_name
            created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)

```

Perform RAG on the ground truth node to generate reference answer
```
            if labelled:
                index = summary_indices[idx]
                qr_tasks = []
                for query in cleaned_questions:
                    # build summary index off of node (i.e. context)
                    qa_query_engine = index.as_query_engine(
                        llm=self._llm,
                        text_qa_template=self.text_qa_template,
                    )
                    qr_task = qa_query_engine.aquery(query)
                    qr_tasks.append(qr_task)
                answer_responses: List[RESPONSE_TYPE] = await run_jobs(
                    qr_tasks, self._show_progress, self._workers
                )
                for question, answer_response in zip(
                    cleaned_questions, answer_responses
                ):
                    example = LabelledRagDataExample(
                        query=question,
                        reference_answer=str(answer_response),
                        reference_contexts=[reference_context],
                        reference_answer_by=created_by,
                        query_by=created_by,
                    )
                    examples.append(example)

            # Else return without label
            else:
                for query in cleaned_questions:
                    example = LabelledRagDataExample(
                        query=query,
                        reference_answer="",
                        reference_contexts=[reference_context],
                        reference_answer_by=None,
                        query_by=created_by,
                    )
                    examples.append(example)
```

### Limitations (Implement a custom module):
- Single Node generation:
  + Implement Multi Node with random sampling (Done)
  + Implement Counter to track the number of times nodes are used to generate (Done)
- Better Parsing of Questions
  + Use Function Calling to return list of questions instead of free texts (Done)
- Lack of examples
  + Implement a prompt placeholders for dynamically inserting examples (Done)
  + Implement an example bank to store generated examples (SelfInstruct style)
- Evolve to more complicated questions => EvolInstruct
- Extra Steps to ensure context -> query relationships