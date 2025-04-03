import os
import streamlit as st
import tempfile
import numpy as np
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    CSVLoader, 
    Docx2txtLoader
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship

# Set page configuration with minimal UI for better performance
st.set_page_config(
    page_title="GraphRAG", 
    page_icon="fsi.jpg", 
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for faster loading
)

# Disable Streamlit's automatic scrolling for better performance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Logo styling */
    .stApp > header {
        background-color: transparent !important;
    }
    .stApp > header img {
        width: 100px;
        margin-top: -50px;
        margin-bottom: -50px;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.5rem;
    }
    .stMarkdown p {
        margin-bottom: 0.5rem;
    }
    .stChatMessage {
        padding: 0.5rem;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Neo4j connection with fallback options
@st.cache_resource
def get_neo4j_driver():
    # Try different connection methods
    connection_methods = [
        {"uri": os.environ.get("NEO4J_URI", "bolt://neo4j:7687")},
        {"uri": "bolt://localhost:7687"},
        {"uri": "bolt://127.0.0.1:7687"}
    ]
    
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "graphragpassword")
    
    for method in connection_methods:
        try:
            uri = method["uri"]
            st.info(f"Trying to connect to Neo4j at {uri}...")
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Test connection
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                st.success(f"Connected to Neo4j database at {uri} with {count} nodes")
            
            # Store the successful URI for py2neo to use
            os.environ["SUCCESSFUL_NEO4J_URI"] = uri
            return driver
        except Exception as e:
            st.warning(f"Failed to connect to Neo4j at {uri}: {str(e)}")
    
    st.error("All Neo4j connection attempts failed")
    return None

# Function to get py2neo Graph object
@st.cache_resource
def get_py2neo_graph():
    # Use the URI that worked for the driver
    uri = os.environ.get("SUCCESSFUL_NEO4J_URI", os.environ.get("NEO4J_URI", "bolt://neo4j:7687"))
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "graphragpassword")
    
    try:
        st.info(f"Connecting to Neo4j with py2neo at {uri}...")
        graph = Graph(uri, auth=(user, password))
        st.success(f"Connected to Neo4j with py2neo at {uri}")
        return graph
    except Exception as e:
        st.error(f"Error connecting to Neo4j with py2neo: {str(e)}")
        return None

# Custom GraphRAG retriever class using Neo4j - optimized for performance
class GraphRAGRetriever:
    def __init__(self, neo4j_driver, embeddings_model):
        self.driver = neo4j_driver
        self.embeddings_model = embeddings_model
        # Cache for query results
        self.cache = {}
        self.cache_size = 50  # Maximum number of cached queries
    
    # Remove caching from the method since it can't hash 'self'
    def get_relevant_documents(self, query):
        from langchain.schema import Document
        
        # Check cache first
        if query in self.cache:
            return self.cache[query]
        
        # Get query embedding
        query_embedding = self.embeddings_model.embed_query(query)
        
        # Handle different embedding formats
        if hasattr(query_embedding, 'tolist'):
            query_embedding_list = query_embedding.tolist()
        elif isinstance(query_embedding, list):
            query_embedding_list = query_embedding
        else:
            # Convert to list if it's another type
            query_embedding_list = list(query_embedding) if hasattr(query_embedding, '__iter__') else [float(query_embedding)]
        
        # Convert embedding to string for Cypher query
        query_embedding_str = str(query_embedding_list)
        
        # Extract keywords from the query for text-based matching
        import re
        # Remove common stop words and extract meaningful keywords
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of'}
        query_keywords = [word.lower() for word in re.findall(r'\b\w+\b', query) 
                         if word.lower() not in stop_words and len(word) > 2]
        
        # Create a Cypher query that searches for chunks only from the selected group
        cypher_query = """
        MATCH (d:Document {group: $current_group})-[:CONTAINS]->(c:Chunk)
        WHERE c.embedding IS NOT NULL
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        WITH c, d, collect(DISTINCT e.name) AS entities,
             CASE
               WHEN any(keyword IN $keywords WHERE toLower(c.content) CONTAINS toLower(keyword)) THEN 2.0
               ELSE 1.0
             END AS relevance_score
        // First, get top chunks per document to ensure document diversity
        WITH d, c, entities, relevance_score
        ORDER BY relevance_score DESC
        // Group by document and collect top chunks for each document
        WITH d, collect({chunk: c, score: relevance_score, entities: entities})[0..3] AS top_chunks_per_doc
        // Unwind to get individual chunks back
        UNWIND top_chunks_per_doc AS chunk_data
        WITH chunk_data.chunk AS c, d, chunk_data.entities AS entities, chunk_data.score AS relevance_score
        // Return results from all documents, sorted by relevance
        RETURN 
            c.content AS content, 
            c.id AS id, 
            d.name AS source, 
            d.id AS doc_id,
            relevance_score AS score,
            entities
        ORDER BY relevance_score DESC
        LIMIT 15
        """
        
        try:
            # Ensure current group exists
            if "current_group" not in st.session_state:
                st.session_state.current_group = "default"
            
            with self.driver.session() as session:
                result = session.run(cypher_query, 
                                    query_embedding=query_embedding_str,
                                    keywords=query_keywords,
                                    current_group=st.session_state.current_group)
                
                # Convert to Documents with enhanced metadata
                documents = []
                for record in result:
                    # Create enhanced metadata with entities and document ID
                    metadata = {
                        "source": record["source"],
                        "score": record["score"],
                        "chunk_id": record["id"],
                        "doc_id": record["doc_id"],  # Include document ID in metadata
                        "entities": record["entities"]
                    }
                    
                    doc = Document(
                        page_content=record["content"],
                        metadata=metadata
                    )
                    documents.append(doc)
                
                # Update cache
                if len(self.cache) >= self.cache_size:
                    # Remove oldest entry if cache is full
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                
                self.cache[query] = documents
                return documents
        except Exception as e:
            st.error(f"Error retrieving documents from Neo4j: {str(e)}")
            return []

# Function to initialize Neo4j database with constraints and indexes
def initialize_neo4j_database(driver):
    try:
        with driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            
            # Create indexes
            session.run("CREATE INDEX document_name IF NOT EXISTS FOR (d:Document) ON (d.name)")
            session.run("CREATE INDEX chunk_content IF NOT EXISTS FOR (c:Chunk) ON (c.content)")
            
            # Skip vector index creation since it requires Neo4j Enterprise Edition
            # We'll use a simpler approach for retrieval
            pass
                
        st.success("Neo4j database initialized with constraints and indexes")
        return True
    except Exception as e:
        st.error(f"Error initializing Neo4j database: {str(e)}")
        return False

# Configure LLM and embeddings
@st.cache_resource
def get_llm():
    try:
        # Use the newer API for creating the LLM
        from langchain_community.llms.ollama import Ollama
        return Ollama(
            base_url="http://192.168.6.150:11434",
            model="llama3.1",
            temperature=0.7,  # Slightly higher temperature for more detailed responses
            num_ctx=4096,  # Larger context window for longer responses
            top_k=10,  # More diverse token selection
            repeat_penalty=1.1  # Reduce repetition
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

@st.cache_resource
def get_embeddings():
    try:
        from langchain_community.embeddings.ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url="http://192.168.6.150:11434", model="llama3.1")
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

# Initialize session state variables
if "neo4j_driver" not in st.session_state:
    st.session_state.neo4j_driver = get_neo4j_driver()
    if st.session_state.neo4j_driver:
        initialize_neo4j_database(st.session_state.neo4j_driver)
if "py2neo_graph" not in st.session_state:
    st.session_state.py2neo_graph = get_py2neo_graph()
if "embeddings_model" not in st.session_state:
    st.session_state.embeddings_model = get_embeddings()
if "llm" not in st.session_state:
    st.session_state.llm = get_llm()
if "graph_retriever" not in st.session_state:
    if st.session_state.neo4j_driver and st.session_state.embeddings_model:
        st.session_state.graph_retriever = GraphRAGRetriever(
            neo4j_driver=st.session_state.neo4j_driver,
            embeddings_model=st.session_state.embeddings_model
        )
# Initialize chat-related session state variables (must happen before any UI elements)
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {"default": []}
if "current_group" not in st.session_state:
    st.session_state.current_group = "default"
if st.session_state.current_group not in st.session_state.chat_histories:
    st.session_state.chat_histories[st.session_state.current_group] = []

# Keep chat histories synchronized with current group
current_group = st.session_state.current_group
if current_group not in st.session_state.chat_histories:
    st.session_state.chat_histories[current_group] = []
if "conversation_ready" not in st.session_state:
    # Check if Neo4j is connected and models are loaded
    if (st.session_state.neo4j_driver and 
        st.session_state.embeddings_model and 
        st.session_state.llm and 
        st.session_state.graph_retriever):
        # Check if there are documents in the database
        try:
            with st.session_state.neo4j_driver.session() as session:
                result = session.run("MATCH (d:Document) RETURN count(d) as count")
                count = result.single()["count"]
                st.session_state.conversation_ready = count > 0
        except:
            st.session_state.conversation_ready = False
    else:
        st.session_state.conversation_ready = False

# Function to load and process documents - optimized for performance with Neo4j
def process_document(file, file_type):
    # Check if Neo4j is connected
    if not st.session_state.neo4j_driver:
        return False, "Neo4j database is not connected"
    
    # Use a context manager to ensure the temp file is properly cleaned up
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
        # Write file content in binary mode for better performance
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Start a progress bar
        progress_bar = st.progress(0)
        
        # Load document based on file type - with progress updates
        progress_bar.progress(10, text="Loading document...")
        if file_type == "pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_type == "docx":
            loader = Docx2txtLoader(tmp_file_path)
        elif file_type == "csv":
            loader = CSVLoader(tmp_file_path)
        else:  # Default to text
            loader = TextLoader(tmp_file_path)
        
        documents = loader.load()
        progress_bar.progress(30, text="Document loaded, splitting into chunks...")
        
        # Split documents into chunks with optimized parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,  # Reduced overlap for better performance
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Explicit separators for better chunking
        )
        chunks = text_splitter.split_documents(documents)
        progress_bar.progress(50, text="Chunks created, building knowledge graph...")
        
        # Get embeddings model
        embeddings_model = get_embeddings()
        if embeddings_model is None:
            return False, "Failed to initialize embeddings model"
        
        # Store embeddings model in session state
        st.session_state.embeddings_model = embeddings_model
        
        # Add to Neo4j graph with progress update and embeddings
        add_to_neo4j_graph(chunks, file.name, embeddings_model)
        progress_bar.progress(80, text="Knowledge graph built with embeddings...")
        
        # Initialize LLM
        llm = get_llm()
        if llm is None:
            return False, "Failed to initialize LLM"
        
        # Store LLM in session state
        st.session_state.llm = llm
        
        progress_bar.progress(90, text="Initializing GraphRAG retriever...")
        
        try:
            # Create GraphRAG retriever
            graph_retriever = GraphRAGRetriever(
                neo4j_driver=st.session_state.neo4j_driver,
                embeddings_model=embeddings_model
            )
            
            # Store retriever in session state
            st.session_state.graph_retriever = graph_retriever
            
            # Mark conversation as ready
            st.session_state.conversation_ready = True
            
            # Complete the progress bar
            progress_bar.progress(100, text="Document processing complete!")
            
        except Exception as e:
            return False, f"Error creating GraphRAG retriever: {str(e)}"
        
        # Clear the progress bar after completion
        progress_bar.empty()
        
        return True, f"Successfully processed {file.name} ({len(chunks)} chunks created)"
    except Exception as e:
        return False, f"Error processing {file.name}: {str(e)}"
    finally:
        os.unlink(tmp_file_path)

# Function to add document chunks to Neo4j graph with embeddings - optimized for performance
def add_to_neo4j_graph(chunks, filename, embeddings_model):
    # Generate a unique document ID using timestamp and a random component
    import uuid
    import time
    
    # Create a unique ID based on timestamp and random UUID
    timestamp = int(time.time())
    random_component = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
    doc_id = f"doc_{timestamp}_{random_component}"
    
    # Process chunks in batches for embedding - batch size of 10 for better performance
    batch_size = 10
    all_chunk_nodes = []
    all_entity_nodes = {}
    all_relationships = []
    
    # Create document node
    # Ensure current group exists
    if "current_group" not in st.session_state:
        st.session_state.current_group = "default"
    
    # Ensure chat history exists for current group
    if "chat_histories" in st.session_state and st.session_state.current_group not in st.session_state.chat_histories:
        st.session_state.chat_histories[st.session_state.current_group] = []
    
    doc_node = Node("Document", 
                   id=doc_id, 
                   name=filename, 
                   group=st.session_state.current_group,
                   created_at=pd.Timestamp.now().isoformat())
    
    # Use py2neo for batch operations
    graph = st.session_state.py2neo_graph
    
    # Process chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        # Get embeddings for batch
        chunk_texts = [chunk.page_content for chunk in batch_chunks]
        chunk_embeddings = embeddings_model.embed_documents(chunk_texts)
        
        # Process each chunk in the batch
        for j, (chunk, embedding) in enumerate(zip(batch_chunks, chunk_embeddings)):
            chunk_index = i + j
            chunk_id = f"chunk_{doc_id}_{chunk_index}"
            
            # Truncate content preview for performance
            if len(chunk.page_content) > 100:
                content_preview = chunk.page_content[:100] + "..."
            else:
                content_preview = chunk.page_content
            
            # Handle embedding format - ensure it's a list for Neo4j
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            elif isinstance(embedding, list):
                embedding_list = embedding
            else:
                # Convert to list if it's another type
                embedding_list = list(embedding) if hasattr(embedding, '__iter__') else [float(embedding)]
            
            # Create chunk node with embedding
            chunk_node = Node("Chunk",
                             id=chunk_id,
                             content=chunk.page_content,
                             preview=content_preview,
                             embedding=embedding_list,  # Store as list for Neo4j
                             index=chunk_index)
            
            all_chunk_nodes.append(chunk_node)
            
            # Create relationship from document to chunk
            contains_rel = Relationship(doc_node, "CONTAINS", chunk_node)
            all_relationships.append(contains_rel)
            
            # Extract entities more efficiently
            entities = set()
            # Use a more efficient tokenization approach
            words = chunk.page_content.lower().split()
            for word in words:
                word = word.strip(".,;:!?()[]{}\"'").lower()
                if len(word) > 5 and word.isalpha() and len(entities) < 5:
                    entities.add(word)
            
            # Process entities
            for entity in entities:
                # Check if we've already created this entity in this batch
                if entity not in all_entity_nodes:
                    # Check if entity exists in database
                    entity_node = graph.nodes.match("Entity", name=entity).first()
                    if not entity_node:
                        entity_node = Node("Entity", name=entity)
                        all_entity_nodes[entity] = entity_node
                    else:
                        all_entity_nodes[entity] = entity_node
                
                # Create relationship from chunk to entity
                mentions_rel = Relationship(chunk_node, "MENTIONS", all_entity_nodes[entity])
                all_relationships.append(mentions_rel)
    
    # Use a single transaction for better performance
    tx = graph.begin()
    
    # Add document node
    tx.create(doc_node)
    
    # Add all chunk nodes in a single batch
    for node in all_chunk_nodes:
        tx.create(node)
    
    # Add all new entity nodes in a single batch
    for entity_node in all_entity_nodes.values():
        if not entity_node.identity:  # Only create if it's a new node
            tx.create(entity_node)
    
    # Add all relationships in a single batch
    for rel in all_relationships:
        tx.create(rel)
    
    # Commit all changes in a single transaction
    graph.commit(tx)

# Function to visualize the graph from Neo4j - highly optimized for performance
@st.cache_data(ttl=600)  # Cache for 10 minutes for better performance
def visualize_neo4j_graph():
    # Check if Neo4j is connected
    if not st.session_state.neo4j_driver:
        return "Neo4j database is not connected. Please check your connection."
    
    # Check if there are documents in the database
    with st.session_state.neo4j_driver.session() as session:
        result = session.run("MATCH (d:Document) RETURN count(d) as count")
        count = result.single()["count"]
        if count == 0:
            return "No graph data available. Please add documents first."
    
    # Create a pyvis network with physics options for better performance
    # Use remote CDN resources to avoid issues with Chrome/Safari
    net = Network(notebook=True, width="100%", height="600px", bgcolor="#222222", font_color="white", cdn_resources="remote")
    
    # Optimize physics for better performance - reduce complexity
    net.barnes_hut(
        gravity=-3000,  # Less gravity for faster rendering
        central_gravity=0.2,  # Less central gravity
        spring_length=100,  # Shorter springs
        spring_strength=0.01,  # Weaker springs for faster stabilization
        damping=0.09,  # More damping for faster stabilization
        overlap=0.5  # Allow more overlap for faster rendering
    )
    
    # Increase node limit to match chunk limit
    max_nodes = 1000  # Match the total chunk limit
    
    # Query Neo4j for nodes and relationships - optimized query with limits
    with st.session_state.neo4j_driver.session() as session:
        # Get document nodes - always include these
        doc_query = """
        MATCH (d:Document)
        RETURN d.id AS doc_id, d.name AS doc_name
        """
        
        doc_result = session.run(doc_query)
        
        # Process document nodes
        doc_nodes = set()
        for record in doc_result:
            if record["doc_id"] and record["doc_id"] not in doc_nodes:
                net.add_node(record["doc_id"], 
                            label=record["doc_name"], 
                            title=record["doc_name"],
                            color="#FF6B6B", 
                            size=25)
                doc_nodes.add(record["doc_id"])
        
        # First, count the total number of documents to distribute chunks evenly
        count_query = """
        MATCH (d:Document)
        RETURN count(d) AS doc_count
        """
        count_result = session.run(count_query)
        doc_count = count_result.single()["doc_count"]
        
        # Calculate chunks per document (with a minimum of 10)
        total_limit = 1000
        chunks_per_doc = max(10, total_limit // doc_count if doc_count > 0 else 10)
        
        # Get chunks for each document - ensure all documents have representation
        chunk_query = f"""
        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
        WITH d, c, c.preview AS preview, c.index AS index
        ORDER BY d.id, index
        WITH d, collect({{chunk_id: c.id, preview: preview, index: index}})[0..{chunks_per_doc}] AS doc_chunks
        UNWIND doc_chunks AS chunk_data
        RETURN 
            d.id AS doc_id,
            chunk_data.chunk_id AS chunk_id, 
            chunk_data.preview AS chunk_preview, 
            chunk_data.index AS chunk_index
        LIMIT {total_limit}
        """
        
        chunk_result = session.run(chunk_query)
        
        # Process chunk nodes and document-chunk edges
        chunk_nodes = set()
        doc_chunk_edges = set()
        
        # First, organize chunks by document to ensure all documents have representation
        chunks_by_doc = {}
        for record in chunk_result:
            doc_id = record["doc_id"]
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            
            chunks_by_doc[doc_id].append({
                "chunk_id": record["chunk_id"],
                "chunk_preview": record["chunk_preview"],
                "chunk_index": record["chunk_index"]
            })
        
        # Ensure each document has at least some chunks shown
        for doc_id, chunks in chunks_by_doc.items():
            # Add at least 5 chunks per document (or all if less than 5)
            for i, chunk in enumerate(chunks[:min(5, len(chunks))]):
                if chunk["chunk_id"] and chunk["chunk_id"] not in chunk_nodes:
                    net.add_node(chunk["chunk_id"], 
                                label=f"C{chunk['chunk_index']}",  # Shorter label
                                title=chunk["chunk_preview"],
                                color="#4ECDC4", 
                                size=10)  # Smaller size
                    chunk_nodes.add(chunk["chunk_id"])
                    
                    # Add document-chunk edge
                    edge = (doc_id, chunk["chunk_id"])
                    if edge not in doc_chunk_edges:
                        net.add_edge(doc_id, chunk["chunk_id"], width=0.5)  # Thinner edges
                        doc_chunk_edges.add(edge)
            
            # Add remaining chunks if we haven't reached the limit
            for chunk in chunks[5:]:
                if len(chunk_nodes) >= max_nodes:
                    break
                    
                if chunk["chunk_id"] and chunk["chunk_id"] not in chunk_nodes:
                    net.add_node(chunk["chunk_id"], 
                                label=f"C{chunk['chunk_index']}",  # Shorter label
                                title=chunk["chunk_preview"],
                                color="#4ECDC4", 
                                size=10)  # Smaller size
                    chunk_nodes.add(chunk["chunk_id"])
                    
                    # Add document-chunk edge
                    edge = (doc_id, chunk["chunk_id"])
                    if edge not in doc_chunk_edges:
                        net.add_edge(doc_id, chunk["chunk_id"], width=0.5)  # Thinner edges
                        doc_chunk_edges.add(edge)
        
        # Get a limited number of entities
        entity_query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE c.id IN $chunk_ids
        RETURN DISTINCT
            c.id AS chunk_id,
            e.name AS entity_name
        LIMIT 50
        """
        
        entity_result = session.run(entity_query, chunk_ids=list(chunk_nodes))
        
        # Process entity nodes and chunk-entity edges
        entity_nodes = set()
        chunk_entity_edges = set()
        
        for record in entity_result:
            # Add entity node if not already added and we haven't reached the limit
            if record["entity_name"] and record["entity_name"] not in entity_nodes and len(entity_nodes) < max_nodes:
                # Truncate long entity names
                entity_label = record["entity_name"]
                if len(entity_label) > 10:
                    entity_label = entity_label[:8] + ".."
                
                net.add_node(record["entity_name"], 
                            label=entity_label,
                            title=record["entity_name"],
                            color="#FFE66D", 
                            size=5)  # Smaller size
                entity_nodes.add(record["entity_name"])
                
                # Add chunk-entity edge
                if record["chunk_id"]:
                    edge = (record["chunk_id"], record["entity_name"])
                    if edge not in chunk_entity_edges:
                        net.add_edge(record["chunk_id"], record["entity_name"], width=0.3)  # Thinner edges
                        chunk_entity_edges.add(edge)
    
    # Set additional options for better performance
    net.set_options("""
    {
      "physics": {
        "stabilization": {
          "iterations": 100,
          "updateInterval": 50,
          "fit": true
        },
        "maxVelocity": 30,
        "minVelocity": 5,
        "solver": "barnesHut",
        "timestep": 0.5
      },
      "interaction": {
        "hover": true,
        "navigationButtons": false,
        "keyboard": false,
        "zoomView": true
      },
      "rendering": {
        "clustering": false
      }
    }
    """)
    
    # Use a more efficient approach to save and read the HTML
    html_file = "graph.html"
    net.save_graph(html_file)
    
    # Read file in binary mode for better performance
    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()
    
    return html

# Initialize file queue session state variables
if "file_queue" not in st.session_state:
    st.session_state.file_queue = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Initialize session state for chat and groups (must happen first, before any UI elements)
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {"default": []}
if "current_group" not in st.session_state:
    st.session_state.current_group = "default"
elif st.session_state.current_group not in st.session_state.chat_histories:
    st.session_state.chat_histories[st.session_state.current_group] = []

# Main UI - simplified for better performance
col1, col2 = st.columns([1, 5])
with col1:
    st.image("fsi.jpg", width=100)
with col2:
    st.title("GraphRAG")

# Group selector and chat input at the top level
def get_available_groups():
    groups = ["default"]
    if st.session_state.neo4j_driver:
        with st.session_state.neo4j_driver.session() as session:
            result = session.run("""
            MATCH (d:Document)
            RETURN DISTINCT d.group as group
            ORDER BY d.group
            """)
            db_groups = [record["group"] for record in result]
            if db_groups:
                groups = db_groups
    return groups

# Function to handle new group creation
def handle_new_group():
    if st.session_state.new_group_name:
        new_group = st.session_state.new_group_name.strip()
        if new_group:
            st.session_state.current_group = new_group
            st.session_state.chat_histories[new_group] = []
            st.session_state.selected_group = new_group  # Update selectbox
            st.experimental_rerun()  # Force UI update

# Function to handle group changes
def handle_group_change():
    if st.session_state.selected_group == "Create new group...":
        return  # Let the text input handle new group creation
    st.session_state.current_group = st.session_state.selected_group
    st.session_state.chat_histories.setdefault(st.session_state.current_group, [])

# Get available groups including any pending new group
groups = get_available_groups()

# Add current group if it exists and isn't in the list
if "current_group" in st.session_state and st.session_state.current_group not in groups:
    groups.append(st.session_state.current_group)

groups.append("Create new group...")

# Initialize new group name state if not exists
if "new_group_name" not in st.session_state:
    st.session_state.new_group_name = ""

# Group selector with change handler
try:
    current_index = groups.index(st.session_state.current_group)
except ValueError:
    current_index = 0
    st.session_state.current_group = "default"

selected_group = st.selectbox(
    "Select document group",
    options=groups,
    key="selected_group",
    index=current_index,
    on_change=handle_group_change
)

# Show new group input if "Create new group..." is selected 
if selected_group == "Create new group...":
    new_group = st.text_input("Enter new group name", key="new_group_name")
    if new_group and new_group.strip():
        new_group = new_group.strip()
        if new_group != "Create new group...":
            st.session_state.current_group = new_group
            st.session_state.chat_histories[new_group] = []
            # Force refresh to show the new group
            st.experimental_rerun()
elif selected_group != st.session_state.current_group:
    st.session_state.current_group = selected_group
    if selected_group not in st.session_state.chat_histories:
        st.session_state.chat_histories[selected_group] = []

# Chat input must be at the root level
user_query = st.chat_input(f"Ask a question about documents in group: {st.session_state.current_group}")

# Create tabs for different views
tab1, tab2 = st.tabs(["Chat", "Knowledge Graph"])

# Chat tab
with tab1:
    # Check if conversation is ready
    if st.session_state.conversation_ready:
        # Create a container for chat messages
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            # Ensure current group exists in chat histories
            if st.session_state.current_group not in st.session_state.chat_histories:
                st.session_state.chat_histories[st.session_state.current_group] = []
            
            current_chat_history = st.session_state.chat_histories[st.session_state.current_group]
            if not current_chat_history:
                st.info(f"No chat history for group: {st.session_state.current_group}")
            else:
                for i, message in enumerate(current_chat_history):
                    if message["role"] == "user":
                        st.chat_message("user").write(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message["content"])
                        
                        # Show source documents with diversity information if available
                        if "sources" in message:
                            # Count unique documents
                            doc_sources = set([source['source'] for source in message["sources"]])
                            
                            with st.expander(f"Sources (from {len(doc_sources)} documents)"):
                                # Group sources by document
                                sources_by_doc = {}
                                for source in message["sources"]:
                                    doc_name = source['source']
                                    if doc_name not in sources_by_doc:
                                        sources_by_doc[doc_name] = []
                                    sources_by_doc[doc_name].append(source)
                                
                                # Display sources grouped by document
                                for doc_name, doc_sources in sources_by_doc.items():
                                    st.markdown(f"### Document: {doc_name}")
                                    for source in doc_sources:
                                        st.markdown(f"*Score: {source['score']:.2f}*")
                                        st.markdown(source['content'])
                                        st.markdown("---")
        
        # Process chat input
        if user_query and st.session_state.conversation_ready:
            # Add user message to current group's chat history
            st.session_state.chat_histories[st.session_state.current_group].append({
                "role": "user", 
                "content": user_query
            })
            
            # Display user message
            st.chat_message("user").write(user_query)
            
            # Get response from LLM with sources
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get relevant documents
                        docs = st.session_state.graph_retriever.get_relevant_documents(user_query)
                        
                        # Create a list of sources for display
                        sources = []
                        for doc in docs:
                            sources.append({
                                "content": doc.page_content,
                                "source": doc.metadata.get("source", "Unknown"),
                                "score": doc.metadata.get("score", 0.0),
                                "entities": doc.metadata.get("entities", [])
                            })
                        
                        # Create a prompt with the retrieved documents
                        prompt = f"""
                        As an expert assistant, provide a detailed and extensive answer to the following question using the provided context. Your response should be thorough and well-organized, following these guidelines:

                        1. Begin with a comprehensive overview that sets the stage for your detailed explanation
                        2. Structure your response with clear sections and descriptive headings
                        3. Incorporate direct quotes and specific examples from the source documents, citing them as "Document X"
                        4. Deep dive into key concepts, explaining their relationships and implications
                        5. Provide extensive explanations with supporting details, avoiding brief or surface-level answers
                        6. Create smooth transitions between topics to maintain a cohesive narrative
                        7. Draw connections between different sources when they support or complement each other
                        8. Include practical examples or applications where relevant
                        9. End with a thorough conclusion that summarizes the key insights

                        If information is missing from the context, clearly identify:
                        - What specific information is needed
                        - Why this information would enhance the answer
                        - How it relates to the available information

                        Aim to be thorough and expansive in your response, providing as much relevant detail as possible.

                        Question: {user_query}

                        Context:
                        """
                        
                        # Add context from retrieved documents
                        for i, doc in enumerate(docs[:5]):  # Limit to top 5 docs for better performance
                            prompt += f"\n---\nDocument {i+1} (from {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}\n"
                        
                        # Get response from LLM
                        response = st.session_state.llm.invoke(prompt)
                        
                        # Display response
                        st.write(response)
                        
                        # Show sources with document diversity information
                        if sources:
                            # Count unique documents
                            doc_sources = set([source['source'] for source in sources])
                            
                            with st.expander(f"Sources (from {len(doc_sources)} documents)"):
                                # Group sources by document
                                sources_by_doc = {}
                                for source in sources:
                                    doc_name = source['source']
                                    if doc_name not in sources_by_doc:
                                        sources_by_doc[doc_name] = []
                                    sources_by_doc[doc_name].append(source)
                                
                                # Display sources grouped by document
                                for doc_name, doc_sources in sources_by_doc.items():
                                    st.markdown(f"### Document: {doc_name}")
                                    for source in doc_sources:
                                        st.markdown(f"*Score: {source['score']:.2f}*")
                                        st.markdown(source['content'])
                                        st.markdown("---")
                        
                        # Add assistant message to current group's chat history with sources
                        st.session_state.chat_histories[st.session_state.current_group].append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.session_state.chat_histories[st.session_state.current_group].append({
                            "role": "assistant", 
                            "content": f"I'm sorry, I encountered an error: {str(e)}"
                        })
    else:
        # Show message if conversation is not ready
        st.info("Please upload documents to start chatting.")

# Knowledge Graph tab
with tab2:
    st.header("Knowledge Graph Visualization")
    
    # Add document statistics
    if st.session_state.conversation_ready:
        with st.session_state.neo4j_driver.session() as session:
            # Get counts by group
            result = session.run("""
            MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
            WITH d.group as group, 
                 count(DISTINCT d) as doc_count,
                 count(DISTINCT c) as chunk_count,
                 count(DISTINCT e) as entity_count
            RETURN group, doc_count, chunk_count, entity_count
            ORDER BY group
            """)
            
            stats = []
            total_docs = 0
            total_chunks = 0
            total_entities = 0
            
            for record in result:
                group = record["group"] or "default"
                stats.append({
                    "group": group,
                    "docs": record["doc_count"],
                    "chunks": record["chunk_count"],
                    "entities": record["entity_count"]
                })
                total_docs += record["doc_count"]
                total_chunks += record["chunk_count"]
                total_entities += record["entity_count"]
            
            # Display total statistics
            st.subheader("Total Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", total_docs)
            with col2:
                st.metric("Total Chunks", total_chunks)
            with col3:
                st.metric("Total Entities", total_entities)
            
            # Display statistics by group
            if stats:
                st.subheader("Statistics by Group")
                # Create DataFrame for better display
                df_stats = pd.DataFrame(stats)
                df_stats.set_index("group", inplace=True)
                
                # Display as a table
                st.dataframe(df_stats)
            
            # Display document distribution by group
            st.subheader("Document Distribution by Group")
            doc_stats_result = session.run("""
            MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
            WITH d.group as group, d.name as name, count(c) as chunk_count
            RETURN group, name, chunk_count
            ORDER BY group, chunk_count DESC
            """)
            
            doc_stats = []
            for record in doc_stats_result:
                doc_stats.append({
                    "Group": record["group"] or "default",
                    "Document": record["name"],
                    "Chunks": record["chunk_count"]
                })
            
            if doc_stats:
                # Create DataFrame for better display
                df = pd.DataFrame(doc_stats)
                
                # Create grouped bar chart
                df_pivot = df.pivot(index='Document', columns='Group', values='Chunks')
                df_pivot.fillna(0, inplace=True)
                
                # Display as a stacked bar chart
                st.bar_chart(df_pivot)
                
                # Also show detailed table
                with st.expander("Show detailed distribution"):
                    st.dataframe(df.set_index(["Group", "Document"]))
    
    # Check if there are documents in the database
    if st.session_state.conversation_ready:
        # Visualize the graph
        with st.spinner("Generating knowledge graph visualization..."):
            html = visualize_neo4j_graph()
            st.components.v1.html(html, height=600)
    else:
        st.info("Please upload documents to visualize the knowledge graph.")

# Function to process the file queue
def process_queue():
    if not st.session_state.file_queue:
        st.session_state.processing = False
        return
    
    # Get the next file from the queue
    file_info = st.session_state.file_queue[0]
    if file_info["status"] != "processing":
        # Update status to processing and rerun to show updated status
        file_info["status"] = "processing"
        st.session_state.file_queue[0] = file_info
        st.rerun()
        
    file = file_info["file"]
    file_type = file_info["type"]
    
    try:
        # Process the file
        success, message = process_document(file, file_type)
        
        # Update status based on result
        if success:
            file_info["status"] = "completed"
            file_info["message"] = message
        else:
            file_info["status"] = "error"
            file_info["message"] = message
        
        # Move from queue to processed
        st.session_state.processed_files.append(file_info)
        st.session_state.file_queue.pop(0)
        
        # Continue processing the queue
        if st.session_state.file_queue:
            st.rerun()
        else:
            st.session_state.processing = False
            
    except Exception as e:
        file_info["status"] = "error"
        file_info["message"] = str(e)
        st.session_state.processed_files.append(file_info)
        st.session_state.file_queue.pop(0)
        
        # Continue processing the queue
        if st.session_state.file_queue:
            st.rerun()
        else:
            st.session_state.processing = False

# Function to delete a document and all its associated data from Neo4j
def delete_document(doc_id, doc_name):
    try:
        # Use a transaction to ensure atomicity
        with st.session_state.neo4j_driver.session() as session:
            # First, get all chunks and entities related to this document
            result = session.run("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
            RETURN collect(distinct c.id) as chunk_ids, collect(distinct e.name) as entity_names
            """, doc_id=doc_id)
            
            record = result.single()
            chunk_ids = record["chunk_ids"] if record and record["chunk_ids"] else []
            entity_names = record["entity_names"] if record and record["entity_names"] else []
            
            # Delete the document and all its relationships and related nodes
            # This is done in a specific order to maintain referential integrity
            
            # 1. Delete MENTIONS relationships between chunks and entities
            session.run("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(c:Chunk)-[r:MENTIONS]->(e:Entity)
            DELETE r
            """, doc_id=doc_id)
            
            # 2. Delete CONTAINS relationships between document and chunks
            session.run("""
            MATCH (d:Document {id: $doc_id})-[r:CONTAINS]->(c:Chunk)
            DELETE r
            """, doc_id=doc_id)
            
            # 3. Delete all chunks belonging to this document
            session.run("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(c:Chunk)
            DELETE c
            """, doc_id=doc_id)
            
            # 4. Delete the document itself
            session.run("""
            MATCH (d:Document {id: $doc_id})
            DELETE d
            """, doc_id=doc_id)
            
            # 5. Clean up orphaned entities (entities not connected to any chunks)
            session.run("""
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-()
            DELETE e
            """)
            
            return True, f"Successfully deleted document: {doc_name}"
    except Exception as e:
        return False, f"Error deleting document: {str(e)}"

# Sidebar for document management - add upload and delete functionality
with st.sidebar:
    st.header("Document Management")
    
    if not st.session_state.neo4j_driver:
        st.error("Neo4j database is not connected. Please check your connection.")
    else:
        # Group management
        st.subheader("Document Groups")
        
        # Get current groups combining database groups and any pending new group
        with st.session_state.neo4j_driver.session() as session:
            result = session.run("""
            MATCH (d:Document)
            RETURN DISTINCT d.group as group
            ORDER BY d.group
            """)
            groups = [record["group"] for record in result]
            if not groups:
                groups = ["default"]
            
            # Add current group if it's not in the database yet
            if st.session_state.current_group not in groups:
                groups.append(st.session_state.current_group)
        
        # Ensure consistent group handling in sidebar
        if st.session_state.current_group not in groups and st.session_state.current_group != "Create new group...":
            st.session_state.current_group = "default"

        # Group selector for upload with validation
        selected_upload_group = st.selectbox(
            "Select group for upload",
            options=groups + ["Create new group..."],
            key="upload_group",
            index=groups.index(st.session_state.current_group) if st.session_state.current_group in groups else 0
        )

        # New group input with proper state handling
        if selected_upload_group == "Create new group...":
            new_group = st.text_input("Enter new group name")
            if new_group and new_group.strip():
                new_group = new_group.strip()
                st.session_state.current_group = new_group
                st.session_state.chat_histories[new_group] = []
                st.experimental_rerun()
        else:
            st.session_state.current_group = selected_upload_group
            if selected_upload_group not in st.session_state.chat_histories:
                st.session_state.chat_histories[selected_upload_group] = []
            
            # File uploader with group selection
            uploaded_files = st.file_uploader(
                f"Upload documents to group: {st.session_state.current_group}",
                type=["pdf", "txt", "docx", "csv"],
                accept_multiple_files=True,
                help="Upload PDF, TXT, DOCX, or CSV files to process"
            )
            
            # Add files to queue when uploaded
            if uploaded_files:
                for file in uploaded_files:
                    # Check if file is already in queue or processed
                    file_names = [f["name"] for f in st.session_state.file_queue + st.session_state.processed_files]
                    if file.name not in file_names:
                        # Get file type from extension
                        file_type = file.name.split(".")[-1].lower()
                        if file_type not in ["pdf", "txt", "docx", "csv"]:
                            file_type = "txt"  # Default to text
                        
                        # Add to queue
                        st.session_state.file_queue.append({
                            "name": file.name,
                            "file": file,
                            "type": file_type,
                            "status": "queued",
                            "message": "Waiting to be processed"
                        })
            
            # Process queue button
            if st.session_state.file_queue:
                if not st.session_state.processing:
                    if st.button("Process Queue", type="primary"):
                        st.session_state.processing = True
                        process_queue()
                elif st.session_state.processing:
                    # Add a spinner while processing
                    with st.spinner(f"Processing {st.session_state.file_queue[0]['name']}..."):
                        process_queue()
            
            # Show queue status
            if st.session_state.file_queue or st.session_state.processed_files:
                st.subheader("File Queue")
            
            # Show files in queue
            for i, file_info in enumerate(st.session_state.file_queue):
                status_color = "blue"
                if file_info["status"] == "processing":
                    status_color = "orange"
                
                st.markdown(f"""
                <div style="padding: 5px; margin-bottom: 5px; border-left: 3px solid {status_color};">
                    <strong>{file_info["name"]}</strong><br/>
                    <small>Status: {file_info["status"]}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Show processed files
            for i, file_info in enumerate(st.session_state.processed_files):
                status_color = "green" if file_info["status"] == "completed" else "red"
                
                st.markdown(f"""
                <div style="padding: 5px; margin-bottom: 5px; border-left: 3px solid {status_color};">
                    <strong>{file_info["name"]}</strong><br/>
                    <small>Status: {file_info["status"]}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Show documents in database
            st.subheader("Documents in Database")
            
            with st.session_state.neo4j_driver.session() as session:
                # Group documents by their groups
                result = session.run("""
                MATCH (d:Document)
                RETURN d.id as id, d.name as name, d.group as group, d.created_at as created_at
                ORDER BY d.group, d.created_at DESC
                """)
                
                # Organize documents by group
                docs_by_group = {}
                for record in result:
                    group = record["group"] or "default"
                    if group not in docs_by_group:
                        docs_by_group[group] = []
                    docs_by_group[group].append({
                        "id": record["id"],
                        "name": record["name"],
                        "created_at": record["created_at"]
                    })
                
                st.subheader("Documents in Database")
                if not docs_by_group:
                    st.info("No documents in database. Upload documents to get started.")
                for group in sorted(docs_by_group.keys()):
                    st.subheader(group)
                    for doc in docs_by_group[group]:
                        st.markdown(f"📄 **{doc['name']}**")
                        if st.button("🗑️", key=f"del_{doc['id']}"):
                            success, msg = delete_document(doc['id'], doc['name'])
                            if success:
                                st.success(msg)
                                st.experimental_rerun()
                            else:
                                st.error(msg)
                        st.markdown("---")
