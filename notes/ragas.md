# RAGAS RAG Synthetic Dataset Generation 

References: https://github.com/explodinggradients/ragas

## Key Classes:

### 1. InMemoryDocumentStore

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
    output_type="json",
)
```
