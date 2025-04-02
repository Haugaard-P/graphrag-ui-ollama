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
- Access to a local Ollama instance running on http://<ollama ip>:11434 (or modify the configuration to use your own Ollama instance)

## Quick Start

1. Clone this repository
2. Make sure your Ollama instance is running
3. Start the application with Docker Compose:

```bash
docker-compose up -d
```

4. Access the web interface at http://localhost:8501
5. Upload documents using the file uploader in the sidebar
6. Process documents to add them to the knowledge graph
7. View the knowledge graph in the "Knowledge Graph" tab
8. Ask questions about your documents in the chat interface

## Configuration

You can customize the application by modifying the following files:

- **docker-compose.yml**: Configure Docker services, ports, and environment variables
- **app.py**: Modify the application logic, UI, or retrieval mechanisms
- **.env**: Set environment variables (create this file if needed)

### Environment Variables

- `OLLAMA_BASE_URL`: URL of your Ollama instance
- `NEO4J_URI`: URI for connecting to Neo4j
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password

## Performance Optimization

The application includes several optimizations for better performance:

1. **Batch Processing**: Documents are processed in batches for better efficiency
2. **Caching**: Expensive operations are cached to reduce computation time
3. **Optimized Queries**: Neo4j queries are optimized for faster retrieval
4. **Resource Allocation**: Docker containers are configured with appropriate resource limits

## Accessing Neo4j Browser

You can access the Neo4j browser directly at http://localhost:7474 to explore the graph database. Use the following credentials:

- Username: neo4j
- Password: graphragpassword

This will allow you to run Cypher queries and visualize the graph structure directly in the Neo4j browser.

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
