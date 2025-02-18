# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot implementation using LlamaIndex and various LLM models. This project implements a sophisticated chatbot that can process documents, create embeddings, and provide context-aware responses.

## Features

- Document processing and chunking with customizable sizes
- Multiple LLM model support (OpenAI GPT-4 and Ollama)
- Hugging Face embeddings integration
- BM25 retrieval system
- Context-aware chat engine with memory
- Multiple query engines including:
  - Basic retriever query engine
  - Summary query engine
  - Function calling agent

## Requirements

### Dependencies

```
llama_hub
llama-index
llama-index-core
llama-index-retrievers-bm25
llama-index-llms-ollama
pydantic
transformers
llama-index-llms-openai
llama-index-embeddings-huggingface
```

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install llama_hub llama-index llama-index-core llama-index-retrievers-bm25 llama-index-llms-ollama pydantic transformers llama-index-llms-openai llama-index-embeddings-huggingface
```

## Configuration

### File Paths
```python
DATA_FILE_PATH = './output_text_file.txt'
FIXED_CHUNK_FOLDER = './chunkraw/fixed_size'
```

### Model Setup
The project supports multiple LLM and embedding models:

```python
# LLM Options:
# 1. Ollama
llm = Ollama(model="llama3.1:8B", temperature=0.1, request_timeout=300)
# 2. OpenAI
llm = OpenAI(model="gpt-4o-mini", temperature=0.1, api_key="your_api_key")

# Embedding Options:
# 1. MiniLM
embeddings = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
# 2. CDE Small
embeddings = HuggingFaceEmbedding(model_name="jxm/cde-small-v2", trust_remote_code=True, max_length=768)
```

## Usage

### 1. Initialize the System
```python
# Set chunk size and overlap
chunk_size = 100
overlap = 10

# Load index and create agents
retriever, query_engine, chat_agent, func_agent = load_index(chunk_size=chunk_size, overlap=overlap)
```

### 2. Chat Interface
```python
# Stream chat response
response = chat_agent.stream_chat("Your question here")

# Print retrieved nodes and response
print("Retrieved nodes from the index:")
for i in response.source_nodes:
   print(f"retriever: {i}")

print("Response from the chat engine")
for token in response.response_gen:
   print(token, end="")
```

## Features in Detail

### Document Processing
- Fixed-size chunking with configurable overlap
- Sentence splitting for natural text segmentation
- Support for both fixed and semantic splitting

### Retrieval System
- BM25 retriever with configurable parameters
- Similarity postprocessing with cutoff threshold
- Context-aware response synthesis

### Chat Engine
- Memory buffer for conversation context
- Custom prompt templates for different scenarios
- System prompts for maintaining conversation style
- Support for both chat and query modes

## Customization

### Chunk Size Configuration
- Larger chunks: More context but potential ambiguity in retrieval
- Smaller chunks: More precise but might miss context
- Default recommendation: chunk_size=100, overlap=10

### Response Templates
The system includes customizable prompt templates for:
- Context processing
- Response refinement
- Question condensing
- System behavior

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Specify your license here]
