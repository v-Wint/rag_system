# Hierarchy-Native RAG for Local-Global Routing

The system builds a top-level schema of the knowledge base and uses it to guide retrieval and provide structural context to the LLM, letting the router decide per-query whether it needs schema, chunks, or neither - addressing the classic RAG failure mode where local (factual) questions retrieve fine but global (structural/thematic) questions don't. Unlike GraphRAG or RAPTOR, which construct new hierarchical structure via LLM extraction or recursive clustering, this system reads the hierarchy that well-organized note-taking tools already impose - trading generality (it assumes structured input) for a much cheaper indexing step (no graph construction, no clustering).

The same hierarchy powers a second, index-free strategy: instead of embedding anything, an agent walks the document tree directly at query time with a single expand tool, navigating titles and previews down to the relevant leaves and answering from what it gathers.


## RAG architecture
### Chunking (schema-guided strategy)
The system leverages the hierarchical tree-like structure of many of the note-taking methods to create a schema of the whole knowledge base with individual chunks as leaves. Each chunk gets its path embedded alongside the text itself for more informed retrieval and reasoning.

### Inference (schema-guided strategy)
During inference router LLM gets token-truncated schema and the user's prompt and:
1. Rewrites the prompt for clarity, to be passed to the main model
2. Rewrites the prompt into a retrieval-optimized query for the vector database
3. Decides which type of the prompt it is: general, fact or schema question

General questions are general greetings, trivia or facts that have little chance of appearing in the schema, so no retrieval occurs.
Fact questions are the questions that don't need schema for the answer. The retrieval for those questions is just top-k chunks.
Schema questions are most likely to be answered by looking at the schema itself, so the system retrieves it alongside top-k chunks.

Retrieval is vector search in Qdrant followed by cross-encoder reranking.

### Agentic traversal
The agentic strategy skips embeddings and schema entirely. The agent gets a single tool, expand, plus a system prompt that defines the behavior: it starts with the top-level overview of the whole knowledge base, expands branches by id (each call returns a size-bounded breadth-first listing of titles, leaf previews and sizes), reads leaves for their full content, and stops once it has gathered enough. A final synthesis step writes the answer from the gathered content. Because traversal is agent-guided, the same loop handles fact, schema and general questions - the agent decides whether and where to explore.

## Application architecture
The application relies on the Feature Training Inference architecture and consists of the following ZenML [pipelines](/pipelines/):
1. etl_pipeline(data_dir) - extracts and cleans documents from the directory, converts them to markdown format if needed, and puts them into the MongoDB warehouse. The system tracks each document's hash value to synchronize warehouse with the data directory
2. feature_pipeline(config) - reconciles changed and deleted documents against the current index, rebuilds what changed, and materializes the strategy's index: hierarchical and recursive embed their chunks into Qdrant (the former also persisting the pruned schema), while agentic embeds nothing since the tree alone is its retrieval index
3. inference_pipeline(queries, config) - builds the inference graph or chain from the config, runs the queries, and logs the graph steps through MLflow
4. evaluation_pipeline(dataset_name, config) - loads the golden dataset from the questions/ directory, runs inference on each question, persists predictions to MongoDB so a run is crash-resumable, then computes the metrics (router accuracy and the custom answer-correctness score) and logs them to MLflow

## Stack
- **Orchestration**: ZenML (pipeline structure, lineage), LangGraph (inference-time routing and agent loops)
- **Experiment tracking**: MLflow
- **Evaluation**: Ragas
- **Document store**: MongoDB
- **Vector DB**: Qdrant
- **Embeddings**: HuggingFace
- **LLMs**: DeepSeek
