# Schema-Guided RAG system for Local-Global Routing
The system builds a top-level schema of the knowledge base and uses it to guide retrieval and provide structural context to the LLM, letting the router decide per-query whether it needs schema, chunks, or neither - addressing the classic RAG failure mode where local (factual) questions retrieve fine but global (structural/thematic) questions don't. Unlike GraphRAG or RAPTOR, which construct new hierarchical structure via LLM extraction or recursive clustering, this system reads the hierarchy that well-organized note-taking tools already impose - trading generality (it assumes structured input) for a much cheaper indexing step (no graph construction, no clustering).

## RAG architecture
### Chunking
The system leverages the hierarchical tree-like structure of many of the note-taking methods to create a schema of the whole knowledge base with individual chunks as leaves. Each chunk gets its path embedded alongside the text itself for more informed retrieval and reasoning.
### Inference
During inference router LLM gets token-truncated schema and the user's prompt and:
1. Rewrites the prompt for clarity, to be passed to the main model
2. Rewrites the prompt into a retrieval-optimized query for the vector database
3. Decides which type of the prompt it is: general, fact or schema question

General questions are general greetings, trivia or facts that have little chance of appearing in the schema, so no retrieval occurs.
Fact questions are the questions that don't need schema for the answer. The retrieval for those questions is just top-k chunks.
Schema questions are most likely to be answered by looking at the schema itself, so the system retrieves it alongside top-k chunks.

Retrieval includes document search in the database and then cross encoder's reranking.

## Application architecture
The application relies on the Feature Training Inference architecture and consists of the following ZenML [pipelines](/pipelines/):
1. etl_pipeline(data_dir) - extracts documents from the directory, converts them to markdown format if needed, and puts them into the MongoDB warehouse. The system tracks each document's hash value to synchronize warehouse with the data directory
2. feature_pipeline(documents, deleted_rel_paths, config) - retrieves changed and deleted documents from MongoDB warehouse (or gets them from etl pipeline output), cleans them, performs chunking, embedding and saving the chunks in the vector database. Schema is saved in MongoDB
3. inference_pipeline(queries, config) - performs inference by building the LangGraph according to the config and invoking the queries, logs graph steps through MLflow
4. evaluation_pipeline(dataset_path, config) - first loads golden dataset at dataset_path and performs inference on each of those, saves them into MongoDB, so that the intermediate results are not lost in the case of a crash. After all the predictions are done it uses Ragas to compute metrics, which are saved in MLflow

## Stack
- **Orchestration**: ZenML (pipeline structure, lineage), LangGraph (inference-time routing)
- **Experiment tracking**: MLflow
- **Evaluation**: Ragas
- **Document store**: MongoDB
- **Vector DB**: Qdrant
- **Embeddings**: HuggingFace
- **LLMs**: Groq
