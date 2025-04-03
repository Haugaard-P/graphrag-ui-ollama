# GraphRAG - Graph-based Retrieval Augmented Generation

A powerful application that combines graph databases with retrieval augmented generation to provide enhanced question answering capabilities over your documents.

## Overview

GraphRAG leverages Neo4j as a graph database to store and query document chunks, entities, and their relationships. This approach enhances traditional RAG systems by incorporating graph structures to capture relationships between different pieces of information, enabling more contextual and accurate responses.

## Features

- **Document Processing**: Upload and process documents in various formats (PDF, DOCX, TXT, CSV)
- **Knowledge Graph Construction**: Automatically extract entities and build a knowledge graph
- **Vector Embeddings**: Generate embeddings for semantic similarity search
- **Graph Visualization**: Interactive visualization of the knowledge graph
- **Conversational Interface**: Chat with your documents using natural language
- **Multi-hop Reasoning**: Connect information across different documents through graph traversal
- **Source Attribution**: Responses include references to source documents

## Architecture

The application consists of the following components:

1. **Streamlit UI**: Web interface for uploading documents and chatting
2. **Neo4j Database**: Graph database for storing documents, chunks, entities, and relationships
3. **Ollama Integration**: Connection to a local LLM for embeddings and text generation
4. **GraphRAG Retriever**: Custom retriever that combines vector similarity with graph traversal

## Prerequisites

- Docker and Docker Compose
- Access to a local Ollama instance

## Quick Start

1. Clone this repository
2. Make sure your Ollama instance is running
3. Create and configure your environment file:
   ```bash
   cp .env-example .env
   ```
   Then edit the `.env` file to set your Ollama URL and model:
   ```env
   OLLAMA_BASE_URL=http://your-ollama-ip:11434  # Replace with your Ollama IP
   MODEL=llama3.1                               # Specify your preferred model
   NEO4J_URI=bolt://neo4j:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=graphragpassword
   ```
4. Start the application with Docker Compose:

```bash
docker-compose up -d
```

5. Access the web interface at http://localhost:8501
6. Upload documents using the file uploader in the sidebar
7. Process documents to add them to the knowledge graph
8. View the knowledge graph in the "Knowledge Graph" tab
9. Ask questions about your documents in the chat interface

## Configuration

### Environment Variables

1. Create a `.env` file in the root directory:
   ```bash
   cp .env-example .env
   ```

2. Configure the following variables:

   ```env
   # Ollama Configuration
   OLLAMA_BASE_URL=http://your-ollama-ip:11434  # URL of your Ollama instance
   MODEL=llama3.1                               # Model to use (must be installed in Ollama)

   # Neo4j Configuration
   NEO4J_URI=bolt://neo4j:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=graphragpassword
   ```

   - **OLLAMA_BASE_URL**: Your Ollama instance URL
     - Local: `http://localhost:11434`
     - Remote: `http://<ollama-ip>:11434`
     - Docker: Use host machine IP
   
   - **MODEL**: Ollama model name (default: `llama3.1`)

### Configuration Files

You can further customize the application by modifying:

- **docker-compose.yml**: Services, ports, and environment variables
- **app.py**: Application logic, UI, and retrieval mechanisms

## Performance Optimization

The application includes several optimizations:

1. **Batch Processing**: Efficient document processing in batches
2. **Caching**: Reduces computation time for expensive operations
3. **Optimized Queries**: Faster retrieval with optimized Neo4j queries
4. **Resource Allocation**: Appropriate Docker container resource limits

## Accessing Neo4j Browser

Access the Neo4j browser at http://localhost:7474 to explore the graph database:

- Username: `neo4j`
- Password: `graphragpassword`

This interface allows you to run Cypher queries and visualize the knowledge graph structure.

## Troubleshooting

### Connection Issues

If you encounter connection issues with Neo4j, the application will automatically try different connection methods:

1. Using the container name: `bolt://neo4j:7687`
2. Using localhost: `bolt://localhost:7687`
3. Using IP address: `bolt://127.0.0.1:7687`

### Memory Issues

If you encounter memory issues, you can adjust the memory allocation in the `docker-compose.yml` file:

```yaml
deploy:
  resources:
    limits:
      memory: 2G  # Adjust as needed
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
