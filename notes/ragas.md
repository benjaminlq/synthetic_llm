RAGAS RAG Synthetic Dataset Generation 
=========

References: https://github.com/explodinggradients/ragas

- [RAGAS RAG Synthetic Dataset Generation](#ragas-rag-synthetic-dataset-generation)
- [1. Dataset Generator](#1-dataset-generator)
- [2. InMemoryDocumentStore](#2-inmemorydocumentstore)
- [3. Evolution Class:](#3-evolution-class)
  - [Base Class - Evolution](#base-class---evolution)
  - [Simple Evolution](#simple-evolution)
  - [Complex Evolution](#complex-evolution)
  - [MultiContextEvolution](#multicontextevolution)
  - [ReasoningEvolution](#reasoningevolution)
  - [Conditional Evolution](#conditional-evolution)
- [4. Node Filters](#4-node-filters)
  - [Node Filter](#node-filter)
  - [Question Filter](#question-filter)
  - [Evolution Filter](#evolution-filter)

# 1. Dataset Generator

```

DEFAULT_DISTRIBUTION = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}

def generate(
    self, test_size: int, distributions: t.Optional[Distributions] = None,
):
    distributions = distributions or DEFAULT_DISTRIBUTION

    ...
    # Initiate Evolution
    - Set docstore, generation_llm and critic_llm

    for evolution in distributions:
        self.init_evolution(evolution)
        evolution.init(is_async=is_async, run_config=run_config)

        def init_evolution(self, evolution: Evolution) -> None:
            evolution.docstore = self.docstore

            if evolution.generator_llm is None:
                evolution.generator_llm = self.generator_llm

                if evolution.question_filter is None:
                    evolution.question_filter = QuestionFilter(llm=self.critic_llm)
                if evolution.node_filter is None:
                    evolution.node_filter = NodeFilter(llm=self.critic_llm)

                if isinstance(evolution, ComplexEvolution):
                    if evolution.evolution_filter is None:
                        evolution.evolution_filter = EvolutionFilter(llm=self.critic_llm)

    ...

    # Run through all the nodes 
    current_nodes = [
        CurrentNodes(root_node=n, nodes=[n])
        for n in self.docstore.get_random_nodes(k=test_size)
    ]
    total_evolutions = 0
    for evolution, probability in distributions.items():
        for i in sample(range(test_size), round(probability * test_size)):
            exec.submit(evolution.evolve, current_nodes[i], name=f"{evolution.__class__.__name__}-{i}",)
            total_evolutions += 1

    ...

    # Collect
    test_data_rows = exec.results()
    test_data_rows = [r for r in test_data_rows if not is_nan(r)]
    test_dataset = TestDataset(test_data=test_data_rows)

    return test_dataset
```

# 2. InMemoryDocumentStore

```
class InMemoryDocumentStore(DocumentStore):
    splitter: TextSplitter => To split documents
    extractor: t.Optional[Extractor] => To extract key phrases from each node
    embeddings: t.Optional[BaseRagasEmbeddings] => Embedding Model
    nodes: t.List[Node] => Node library to generate queries
    node_embeddings_list: t.List[Embedding] => List of embedded vectors
    node_map: t.Dict[str, Node] => Mapping between node ids and nodes
```

Add list of nodes to node library

```
def add_nodes(self, nodes: t.Sequence[Node], show_progress=True):
    ...
    for node in nodes:
        - Generate Embeddings => attach embedding to self.nodes & append to self.node_embeddings_list
        - Generate Key => attach key phrases to self.nodes

    doc_embeddings = {}
    docs = set([node.filename for node in self.nodes])
    for doc in docs:
        doc_embeddings[doc] = np.mean([for node in self.node if node.filename == doc])
    
    for node in self.nodes:
        node.doc_similarity = similarity(node.embedding, doc_embeddings[node.filename])
```

Random sample based on previous no of occurences:

```
    def get_random_nodes(self, k=1, alpha=0.1) -> t.List[Node]:
        def adjustment_factor(wins, alpha):
            return np.exp(-alpha * wins)

        scores = [adjustment_factor(node.wins, alpha) for node in self.nodes]
        similarity_scores = [node.doc_similarity for node in self.nodes]
        prob = np.array(scores) * np.array(similarity_scores)
        prob = prob / np.sum(prob)

        nodes = rng.choice(np.array(self.nodes), size=k, p=prob).tolist()

        for node in nodes:
            idx = self.nodes.index(node)
            self.nodes[idx].wins += 1
```

- KeyphraseExtractor:
```
# Prompt
keyphrase_extraction_prompt = Prompt(
    name="keyphrase_extraction",
    instruction="Extract the top 3 to 5 keyphrases from the provided text, focusing on the most significant and distinctive aspects.",
    examples=[
        {"text": "A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, has enough energy to escape it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.",
            "output": {"keyphrases": ["Black hole", "Region of spacetime", "Strong gravity", "Light and electromagnetic waves", "Theory of general relativity",]}}
        ...],
    input_keys=["text"],
    output_key="output",
)
```

# 3. Evolution Class:
## Base Class - Evolution
```
class Evolution
    generator_llm: BaseRagasLLM => Used for generating questions
    docstore: t.Optional[DocumentStore] =>  Nodes library
    node_filter: t.Optional[NodeFilter]
    question_filter: t.Optional[QuestionFilter]
    question_answer_prompt: 
    find_relevant_context_prompt: 
    rewrite_invalid_question_prompt:

    @abstractmethod
    async def _aevolve(self, current_tries: int, current_nodes: CurrentNodes) -> EvolutionOutput:
        ...

    async def fix_invalid_question(self, question: str, current_nodes: CurrentNodes, feedback: str):
        prev_node = current_nodes.root_node.prev
        if prev_node is not None:
            current_nodes.nodes.insert(0, prev_node)
            current_nodes.root_node = prev_node
            prompt = self.rewrite_invalid_question_prompt.format(
                question=question,context=self.merge_nodes(current_nodes).page_content,feedback=feedback)
            results = await self.generator_llm.generate(prompt=prompt, is_async=self.is_async)
            question = results.generations[0][0].text.strip()

        return question, current_nodes

    # Method to generate answers from relevant contexts
    async def generate_datarow(
        self, question: str, current_nodes: CurrentNodes, evolution_type: str):

        # Find relevant context from a set of nodes
        node_content = [f"{i+1}\t{n.page_content}" for i, n in enumerate(current_nodes.nodes)]
        results = await self.generator_llm.generate(
            prompt=self.find_relevant_context_prompt.format(
                question=question, contexts=node_content))

        relevant_contexts_result = await json_loader.safe_load(results.generations[0][0].text.strip(), llm=self.generator_llm)
        relevant_context_indices = (
            relevant_contexts_result.get("relevant_contexts", None) 
            if isinstance(relevant_contexts_result, dict) else None)

        if relevant_context_indices is not None:
            relevant_context_indices = [idx for idx in relevant_context_indices if isinstance(idx, int)]

        if relevant_context_indices is None or not relevant_context_indices:
            relevant_context = CurrentNodes(
                root_node=current_nodes.root_node, nodes=current_nodes.nodes
            ) # Return original nodes
        else:
            selected_nodes = [
                current_nodes.nodes[i - 1] for i in relevant_context_indices if i - 1 < len(current_nodes.nodes)]
            relevant_context = (
                CurrentNodes(root_node=selected_nodes[0], nodes=selected_nodes) if selected_nodes else current_nodes)

        merged_nodes = self.merge_nodes(relevant_context)
        results = await self.generator_llm.generate(
            prompt=self.question_answer_prompt.format(question=question, context=merged_nodes.page_content
            )
        )
        answer = await json_loader.safe_load(
            results.generations[0][0].text.strip(), self.generator_llm
        )

        return DataRow(
            question=question.strip('"'),
            contexts=[n.page_content for n in relevant_context.nodes],
            ground_truth=answer,
            evolution_type=evolution_type,
            metadata=[n.metadata for n in relevant_context.nodes],
        )

```

* Find Relevant Contexts Prompt
```
find_relevant_context_prompt = Prompt(
    instruction="Given a question and set of contexts, find the most relevant contexts to answer the question.",
    examples=[
        {
            "question": "What is the capital of France?",
            "contexts": [
                "1. France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.",
                "2. The capital of France is Paris. It is also the most populous city in France, with a population of over 2 million people. Paris is known for its cultural landmarks like the Eiffel Tower and the Louvre Museum.",
                "3. Paris is the capital of France. It is also the most populous city in France, with a population of over 2 million people. Paris is known for its cultural landmarks like the Eiffel Tower and the Louvre Museum.",
            ],
            "output": {"relevant_contexts": [1, 2]},
        },
    ],
    input_keys=["question", "contexts"],
    output_key="output",
)
```

* Rewrite Question Prompt
```
question_rewrite_prompt = Prompt(
    instruction="""Given a context, question and feedback, rewrite the question to improve its clarity and answerability based on the feedback provided.""",
    examples=[
        {
            "context": "The Eiffel Tower was constructed using iron and was originally intended as a temporary exhibit for the 1889 World's Fair held in Paris. Despite its initial temporary purpose, the Eiffel Tower quickly became a symbol of Parisian ingenuity and an iconic landmark of the city, attracting millions of visitors each year. The tower's design, created by Gustave Eiffel, was initially met with criticism from some French artists and intellectuals, but it has since been celebrated as a masterpiece of structural engineering and architectural design.",
            "question": "Who created the design for the Tower?",
            "feedback": "The question asks about the creator of the design for 'the Tower', but it does not specify which tower it refers to. There are many towers worldwide, and without specifying the exact tower, the question is unclear and unanswerable. To improve the question, it should include the name or a clear description of the specific tower in question.",
            "output": "Who created the design for the Eiffel Tower?",
        },
        {
            "context": "'Exploring Zero-Shot Learning in Neural Networks' was published by Smith and Lee in 2021, focusing on the application of zero-shot learning techniques in artificial intelligence.",
            "question": "What datasets were used for the zero-shot evaluations in this study?",
            "feedback": "The question asks about the datasets used for zero-shot evaluations in 'this study', without specifying or providing any details about the study in question. This makes the question unclear for those who do not have access to or knowledge of the specific study. To improve clarity and answerability, the question should specify the study it refers to, or provide enough context about the study for the question to be understood and answered independently.",
            "output": "What datasets were used for the zero-shot evaluations Exploring Zero-Shot Learning in Neural Networks paper?",
        },
    ],
    input_keys=["context", "question", "feedback"],
    output_key="output",
)
```

* Prompt for Contextualized Question Answering

```
class AnswerFormat(BaseModel):
    answer: str
    verdict: int

question_answer_prompt = Prompt(
    instruction="""Answer the question using the information from the given context. Output verdict as '1' if answer is present '-1' if answer is not present in the context.""",
    output_format_instruction=get_json_format_instructions(AnswerFormat),
    examples=[
        {
            "context": """Climate change is significantly influenced by human activities, notably the emission of greenhouse gases from burning fossil fuels. The increased greenhouse gas concentration in the atmosphere traps more heat, leading to global warming and changes in weather patterns.""",
            "question": "How do human activities contribute to climate change?",
            "answer": AnswerFormat.parse_obj({
                "answer": "Human activities contribute to climate change primarily through the emission of greenhouse gases from burning fossil fuels. These emissions increase the concentration of greenhouse gases in the atmosphere, which traps more heat and leads to global warming and altered weather patterns.",
                "verdict": "1",}).dict(),
        },
        {
            "context": """The novel "Pride and Prejudice" by Jane Austen revolves around the character Elizabeth Bennet and her family. The story is set in the 19th century in rural England and deals with issues of marriage, morality, and misconceptions.""",
            "question": "What year was 'Pride and Prejudice' published?",
            "answer": AnswerFormat.parse_obj({
                "answer": "The answer to given question is not present in context",
                "verdict": "-1",}).dict(),
        },
    ],
    input_keys=["context", "question"],
    output_key="answer",
)
```
## Simple Evolution

```
class SimpleEvolution(Evolution):
    seed_question_prompt: Prompt = field(default_factory=lambda: seed_question_prompt)

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:

        # Check that current node is relevant and useful
        merged_node = self.merge_nodes(current_nodes)
        passed = await self.node_filter.filter(merged_node)
        if not passed["score"]:
            current_nodes = self._get_new_random_node()
            return await self.aretry_evolve(
                current_tries, current_nodes, update_count=False
            )

        # Based on topic and node content
        results = await self.generator_llm.generate(
            prompt=self.seed_question_prompt.format(
                context=merged_node.page_content,
                keyphrase=rng.choice(np.array(merged_node.keyphrases), size=1)[0],
            )
        )
        seed_question = results.generations[0][0].text

        # Check that the question quality is avalid
        is_valid_question, feedback = await self.question_filter.filter(seed_question)

        if not is_valid_question:
            # get more context to rewrite question
            seed_question, current_nodes = await self.fix_invalid_question(
                seed_question, current_nodes, feedback
            )
            logger.info("rewritten question: %s", seed_question)
            is_valid_question, _ = await self.question_filter.filter(seed_question)
            if not is_valid_question:
                # retry with new nodes added
                current_nodes = self._get_new_random_node()
                return await self.aretry_evolve(current_tries, current_nodes)

        return seed_question, current_nodes, "simple"
```

* Seed Question Prompt

```
seed_question_prompt = Prompt(
    name="seed_question",
    instruction="Generate a question that can be fully answered from given context. The question should be formed using topic",
    examples=[
        {
            "context": "Photosynthesis in plants involves converting light energy into chemical energy, using chlorophyll and other pigments to absorb light. This process is crucial for plant growth and the production of oxygen.",
            "keyphrase": "Photosynthesis",
            "question": "What is the role of photosynthesis in plant growth?",
        },
        {
            "context": "The Industrial Revolution, starting in the 18th century, marked a major turning point in history as it led to the development of factories and urbanization.",
            "keyphrase": "Industrial Revolution",
            "question": "How did the Industrial Revolution mark a major turning point in history?",
        },
        {
            "context": "The process of evaporation plays a crucial role in the water cycle, converting water from liquid to vapor and allowing it to rise into the atmosphere.",
            "keyphrase": "Evaporation",
            "question": "Why is evaporation important in the water cycle?",
        },
    ],
    input_keys=["context", "keyphrase"],
    output_key="question",
    output_type="str",
)
```

## Complex Evolution

```
class ComplexEvolution(Evolution):
    se: t.Optional[SimpleEvolution] = field(default=None, repr=False)
    evolution_filter: t.Optional[EvolutionFilter] = field(default=None, repr=False)
    compress_question_prompt: Prompt = field(default_factory=lambda: compress_question_prompt)

    def init(self, is_async: bool = True, run_config: t.Optional[RunConfig] = None):

        self.se = SimpleEvolution(
            generator_llm=self.generator_llm,
            docstore=self.docstore,
            node_filter=self.node_filter,
            question_filter=self.question_filter,
        )

        assert self.node_filter is not None, "node filter cannot be None"
        if self.evolution_filter is None:
            self.evolution_filter = EvolutionFilter(self.node_filter.llm)

    async def _acomplex_evolution(
        self, current_tries: int, current_nodes: CurrentNodes, question_prompt: Prompt
    ):

        # Get a simple question first
        simple_question, current_nodes, _ = await self.se._aevolve(
            current_tries, current_nodes
        )

        merged_node = self.merge_nodes(current_nodes)

        # Generic Evolve Step to modify simple question
        result = await self.generator_llm.generate(
            prompt=question_prompt.format(
                question=simple_question, context=merged_node.page_content
            )
        )
        reasoning_question = result.generations[0][0].text.strip()
        is_valid_question, feedback = await self.question_filter.filter(
            reasoning_question
        )
        if not is_valid_question:
            reasoning_question, current_nodes = await self.fix_invalid_question(
                reasoning_question, current_nodes, feedback
            )
            is_valid_question, _ = await self.question_filter.filter(reasoning_question)
            if not is_valid_question:
                current_nodes = self.se._get_new_random_node()
                return await self.aretry_evolve(current_tries, current_nodes)

        # compress the question
        compressed_question = await self._transform_question(
            prompt=self.compress_question_prompt, question=reasoning_question
        )

        assert self.evolution_filter is not None, "evolution filter cannot be None"
        if await self.evolution_filter.filter(simple_question, compressed_question):
            current_nodes = self.se._get_new_random_node()
            return await self.aretry_evolve(current_tries, current_nodes)

        return compressed_question, current_nodes
```

* Compress Question Prompt

```
compress_question_prompt = Prompt(
    name="compress_question",
    instruction="""Rewrite the following question to make it more indirect and shorter while retaining the essence of the original question.
    The goal is to create a question that conveys the same meaning but in a less direct manner. The rewritten question should shorter so use abbreviation wherever possible.""",
    examples=[
        {
            "question": "What is the distance between the Earth and the Moon?",
            "output": "How far is the Moon from Earth?",
        },
        {
            "question": "What ingredients are required to bake a chocolate cake?",
            "output": "What's needed for a chocolate cake?",
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="str",
    language="english",
)
```

## MultiContextEvolution

* Class

```
class MultiContextEvolution(ComplexEvolution):
    multi_context_question_prompt: Prompt = field(
        default_factory=lambda: multi_context_question_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        # Generate simple question first
        simple_question, current_nodes, _ = await self.se._aevolve(
            current_tries, current_nodes
        )

        # Find a similar node to add to multi nodes list
        merged_node = self.merge_nodes(current_nodes)
        similar_node = self.docstore.get_similar(merged_node, top_k=1)
        if not similar_node:
            new_random_nodes = self.docstore.get_random_nodes(k=1)
            current_nodes = CurrentNodes(
                root_node=new_random_nodes[0], nodes=new_random_nodes
            )
            return await self.aretry_evolve(current_tries, current_nodes)
        else:
            assert isinstance(similar_node[0], Node), "similar_node must be a Node"
            current_nodes.nodes.append(similar_node[0])

        # Combine content of context1 and context2
        prompt = self.multi_context_question_prompt.format(
            question=simple_question,
            context1=merged_node.page_content,
            context2=similar_node[0].page_content,
        )
        results = await self.generator_llm.generate(prompt=prompt)
        question = results.generations[0][0].text.strip()  

        # Validate step
        is_valid_question, feedback = await self.question_filter.filter(question)
        if not is_valid_question:
            question, current_nodes = await self.fix_invalid_question(
                question, current_nodes, feedback
            )
            is_valid_question, _ = await self.question_filter.filter(question)

            if not is_valid_question:
                # retry with new nodes added
                current_nodes = self.se._get_new_random_node()
                return await self.aretry_evolve(current_tries, current_nodes)

        compressed_question = await self._transform_question(
            prompt=self.compress_question_prompt, question=question
        )

        if await self.evolution_filter.filter(simple_question, compressed_question):
            # retry
            current_nodes = self.se._get_new_random_node()
            return await self.aretry_evolve(current_tries, current_nodes)

        return compressed_question, current_nodes, "multi_context"
```

* Prompt
  
```
multi_context_question_prompt = Prompt(
    instruction="""
    The task is to rewrite and complicate the given question in a way that answering it requires information derived from both context1 and context2. 
    Follow the rules given below while rewriting the question.
        1. The rewritten question should not be very long. Use abbreviation wherever possible.
        2. The rewritten question must be reasonable and must be understood and responded by humans.
        3. The rewritten question must be fully answerable from information present in context1 and context2. 
        4. Read and understand both contexts and rewrite the question so that answering requires insight from both context1 and context2.
        5. phrases like 'based on the provided context','according to the context?',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What process turns plants green?",
            "context1": "Chlorophyll is the pigment that gives plants their green color and helps them photosynthesize.",
            "context2": "Photosynthesis in plants typically occurs in the leaves where chloroplasts are concentrated.",
            "output": "In which plant structures does the pigment responsible for their verdancy facilitate energy production?",
        },
        {
            "question": "How do you calculate the area of a rectangle?",
            "context1": "The area of a shape is calculated based on the shape's dimensions. For rectangles, this involves multiplying the length and width.",
            "context2": "Rectangles have four sides with opposite sides being equal in length. They are a type of quadrilateral.",
            "output": "What multiplication involving equal opposites yields a quadrilateral's area?",
        },
    ],
    input_keys=["question", "context1", "context2"],
    output_key="output",
    output_type="str",
    language="english",
)
```

## ReasoningEvolution

* Class
```
class ReasoningEvolution(ComplexEvolution):
    reasoning_question_prompt: Prompt = field(
        default_factory=lambda: reasoning_question_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        result = await self._acomplex_evolution(
            current_tries, current_nodes, self.reasoning_question_prompt
        )
        return result[0], result[1], "reasoning"
```

* Prompt
```
reasoning_question_prompt = Prompt(
    name="reasoning_question",
    instruction="""Complicate the given question by rewriting question into a multi-hop reasoning question based on the provided context.
    Answering the question should require the reader to make multiple logical connections or inferences using the information available in given context.
    Rules to follow when rewriting question:
    1. Ensure that the rewritten question can be answered entirely from the information present in the contexts.
    2. Do not frame questions that contains more than 15 words. Use abbreviation wherever possible.
    3. Make sure the question is clear and unambiguous.
    4. phrases like 'based on the provided context','according to the context',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.",
            "output": "Linking the Eiffel Tower and administrative center, which city stands as both?",
        },
        {
            "question": "What does the append() method do in Python?",
            "context": "In Python, lists are used to store multiple items in a single variable. Lists are one of 4 built-in data types used to store collections of data. The append() method adds a single item to the end of a list.",
            "output": "If a list represents a variable collection, what method extends it by one item?",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="english",
)
```

## Conditional Evolution

* Class
```
@dataclass
class ConditionalEvolution(ComplexEvolution):
    conditional_question_prompt: Prompt = field(
        default_factory=lambda: conditional_question_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        result = await self._acomplex_evolution(
            current_tries, current_nodes, self.conditional_question_prompt
        )
        return result[0], result[1], "conditional"
```

* Prompt
```
conditional_question_prompt = Prompt(
    name="conditional_question",
    instruction="""Rewrite the provided question to increase its complexity by introducing a conditional element.
    The goal is to make the question more intricate by incorporating a scenario or condition that affects the context of the question.
    Follow the rules given below while rewriting the question.
        1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
        2. The rewritten question must be reasonable and must be understood and responded by humans.
        3. The rewritten question must be fully answerable from information present context.
        4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What is the function of the roots of a plant?",
            "context": "The roots of a plant absorb water and nutrients from the soil, anchor the plant in the ground, and store food.",
            "output": "What dual purpose do plant roots serve concerning soil nutrients and stability?",
        },
        {
            "question": "How do vaccines protect against diseases?",
            "context": "Vaccines protect against diseases by stimulating the body's immune response to produce antibodies, which recognize and combat pathogens.",
            "output": "How do vaccines utilize the body's immune system to defend against pathogens?",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="english",
)
```

# 4. Node Filters

## Node Filter

* Class
```
class NodeFilter(Filter):
    threshold: float = 1.5
    context_scoring_prompt: Prompt = field(
        default_factory=lambda: context_scoring_prompt
    )

    async def filter(self, node: Node) -> t.Dict:
        prompt = self.context_scoring_prompt.format(context=node.page_content)
        results = await self.llm.generate(prompt=prompt)
        output = results.generations[0][0].text.strip()
        output = await context_scoring_parser.aparse(output, prompt, self.llm)
        output = output.dict() if output is not None else {}
        output["score"] = sum(output.values()) / len(output.values())
        logger.debug("context scoring: %s", output)
        output.update({"score": output.get("score", 0) >= self.threshold})
        return output

```

* Context Scoring Prompt
```

class ContextScoring(BaseModel):
    clarity: int
    depth: int
    structure: int
    relevance: int

context_scoring_prompt = Prompt(
    instruction="""
    Given a context, perform the following task and output the answer in VALID JSON format: Assess the provided context and assign a numerical score of 1 (Low), 2 (Medium), or 3 (High) for each of the following criteria in your JSON response:

    clarity: Evaluate the precision and understandability of the information presented. High scores (3) are reserved for contexts that are both precise in their information and easy to understand. Low scores (1) are for contexts where the information is vague or hard to comprehend.
    depth: Determine the level of detailed examination and the inclusion of innovative insights within the context. A high score indicates a comprehensive and insightful analysis, while a low score suggests a superficial treatment of the topic.
    structure: Assess how well the content is organized and whether it flows logically. High scores are awarded to contexts that demonstrate coherent organization and logical progression, whereas low scores indicate a lack of structure or clarity in progression.
    relevance: Judge the pertinence of the content to the main topic, awarding high scores to contexts tightly focused on the subject without unnecessary digressions, and low scores to those that are cluttered with irrelevant information.
    Structure your JSON output to reflect these criteria as keys with their corresponding scores as values
    """,
    output_format_instruction=get_json_format_instructions(ContextScoring),
    examples=[
        {
            "context": "The Pythagorean theorem is a fundamental principle in geometry. It states that in a right-angled triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides. This can be written as a^2 + b^2 = c^2 where c represents the length of the hypotenuse, and a and b represent the lengths of the other two sides.",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 1, "structure": 3, "relevance": 3}
            ).dict(),
        },
        {
            "context": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 2, "structure": 3, "relevance": 3}
            ).dict(),
        },
    ],
    input_keys=["context"],
    output_key="output",
    output_type="json"
)
```

## Question Filter

* Class

```
class QuestionFilter(Filter):
    llm: BaseRagasLLM
    filter_question_prompt: Prompt = field(
        default_factory=lambda: filter_question_prompt
    )

    async def filter(self, question: str) -> t.Tuple[bool, str]:
        prompt = self.filter_question_prompt.format(question=question)
        results = await self.llm.generate(prompt=prompt)
        results = results.generations[0][0].text.strip()
        results = await question_filter_parser.aparse(results, prompt, self.llm)
        results = results.dict() if results is not None else {}
        logger.debug("filtered question: %s", results)
        return results.get("verdict") == 1, results.get("feedback", "")
```

* Question Filter Prompt
```

class QuestionFilter(BaseModel):
    feedback: str
    verdict: int

filter_question_prompt = Prompt(
    instruction="""
    Asses the given question for clarity and answerability given enough domain knowledge, consider the following criteria:
    1.Independence: Can the question be understood and answered without needing additional context or access to external references not provided within the question itself? Questions should be self-contained, meaning they do not rely on specific documents, tables, or prior knowledge not shared within the question.
    2.Clear Intent: Is it clear what type of answer or information the question seeks? The question should convey its purpose without ambiguity, allowing for a direct and relevant response.
    Based on these criteria, assign a verdict of "1" if a question is specific, independent, and has a clear intent, making it understandable and answerable based on the details provided. Assign "0" if it fails to meet one or more of these criteria due to vagueness, reliance on external references, or ambiguity in intent.
    Provide feedback and a verdict in JSON format, including suggestions for improvement if the question is deemed unclear. Highlight aspects of the question that contribute to its clarity or lack thereof, and offer advice on how it could be reframed or detailed for better understanding and answerability.
    """,
    output_format_instruction=get_json_format_instructions(QuestionFilter),
    examples=[
        {
            "question": "What is the discovery about space?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "The question is too vague and broad, asking for a 'discovery about space' without specifying any particular aspect, time frame, or context of interest. This could refer to a wide range of topics, from the discovery of new celestial bodies to advancements in space travel technology. To improve clarity and answerability, the question could specify the type of discovery (e.g., astronomical, technological), the time frame (e.g., recent, historical), or the context (e.g., within a specific research study or space mission).",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "How does ALMA-13B-R perform compared to other translation models in the WMT'23 study, based on the results in context1 and context2?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "This question asks for a comparison of the ALMA-13B-R model's performance against other translation models within the WMT'23 study, specifically referring to results in 'context1' and 'context2'. While it clearly specifies the model of interest (ALMA-13B-R) and the study (WMT'23), it assumes access to and understanding of 'context1' and 'context2' without explaining what these contexts entail. This makes the question unclear for those not familiar with the WMT'23 study or these specific contexts. To improve clarity and answerability for a broader audience, the question could benefit from defining or describing 'context1' and 'context2' or explaining the criteria used for comparison in these contexts.",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "How do KIWI-XXL and XCOMET compare to the gold standard references in Table 1 in terms of evaluation scores, translation model performance, and success rate in surpassing the references?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "The question requests a comparison between KIWI-XXL and XCOMET models and gold standard references in 'Table 1', focusing on evaluation scores, translation model performance, and success rates in surpassing the references. It specifies the models and criteria for comparison, making the intent clear. However, the question assumes access to 'Table 1' without providing its content or context, making it unclear for those without direct access to the source material. To be clearer and more answerable for a general audience, the question could include a brief description of the content or key findings of 'Table 1', or alternatively, frame the question in a way that does not rely on specific, unpublished documents.",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question": "What is the configuration of UL2 training objective in OpenMoE and why is it a better choice for pre-training?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "The question asks for the configuration of the UL2 training objective within the OpenMoE framework and the rationale behind its suitability for pre-training. It is clear in specifying the topic of interest (UL2 training objective, OpenMoE) and seeks detailed information on both the configuration and the reasons for its effectiveness in pre-training. However, the question might be challenging for those unfamiliar with the specific terminology or the context of OpenMoE and UL2. For broader clarity and answerability, it would be helpful if the question included a brief explanation or context about OpenMoE and the UL2 training objective, or clarified the aspects of pre-training effectiveness it refers to (e.g., efficiency, accuracy, generalization).",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question": "What is the detailed configuration of the UL2 training objective in OpenMoE, based on the provided context?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "The question seeks detailed information on the UL2 training objective's configuration within the OpenMoE framework, mentioning 'the provided context' without actually including or describing this context within the query. This makes the question unclear for those who do not have access to the unspecified context. For the question to be clear and answerable, it needs to either include the relevant context directly within the question or be framed in a way that does not require external information. Detailing the specific aspects of the configuration of interest (e.g., loss functions, data augmentation techniques) could also help clarify the query.",
                    "verdict": 0,
                }
            ).dict(),
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="english",
)
```

## Evolution Filter

* Class

```
class EvolutionFilter(Filter):
    llm: BaseRagasLLM
    evolution_elimination_prompt: Prompt = field(
        default_factory=lambda: evolution_elimination_prompt
    )

    async def filter(self, simple_question: str, compressed_question: str) -> bool:
        prompt = self.evolution_elimination_prompt.format(
            question1=simple_question, question2=compressed_question
        )
        results = await self.llm.generate(prompt=prompt)
        results = results.generations[0][0].text.strip()
        results = await evolution_elimination_parser.aparse(results, prompt, self.llm)
        results = results.dict() if results is not None else {}
        logger.debug("evolution filter: %s", results)
        return results.get("verdict") == 1
```

* Evolution Filter Prompt

```
class EvolutionElimination(BaseModel):
    reason: str
    verdict: int

evolution_elimination_prompt = Prompt(
    instruction="""Check if the given two questions are equal based on following requirements:
    1. They have same constraints and requirements.
    2. They have same depth and breadth of the inquiry.
    Output verdict as 1 if they are equal and 0 if they are not""",
    output_format_instruction=get_json_format_instructions(EvolutionElimination),
    examples=[
        {
            "question1": "What are the primary causes of climate change?",
            "question2": "What factors contribute to global warming?",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "While both questions deal with environmental issues, 'climate change' encompasses broader changes than 'global warming', leading to different depths of inquiry.",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question1": "How does photosynthesis work in plants?",
            "question2": "Can you explain the process of photosynthesis in plants?",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "Both questions ask for an explanation of the photosynthesis process in plants, sharing the same depth, breadth, and requirements for the answer.",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question1": "What are the health benefits of regular exercise?",
            "question2": "Can you list the advantages of exercising regularly for health?",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "Both questions seek information about the positive effects of regular exercise on health. They require a similar level of detail in listing the health benefits.",
                    "verdict": 1,
                }
            ).dict(),
        },
    ],
    input_keys=["question1", "question2"],
    output_key="output",
    output_type="json",
    language="english",
)
```