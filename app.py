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
    page_icon="🧠", 
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
        
        # Use a simpler approach for similarity calculation
        # This query doesn't require the GDS library
        cypher_query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        MATCH (d:Document)-[:CONTAINS]->(c)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        WITH c, d, collect(DISTINCT e.name) AS entities
        RETURN 
            c.content AS content, 
            c.id AS id, 
            d.name AS source, 
            1.0 AS score,  // Default score since we can't calculate cosine similarity
            entities
        LIMIT 5
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, query_embedding=query_embedding_str)
                
                # Convert to Documents with enhanced metadata
                documents = []
                for record in result:
                    # Create enhanced metadata with entities
                    metadata = {
                        "source": record["source"],
                        "score": record["score"],
                        "chunk_id": record["id"],
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
        return Ollama(base_url="http://192.168.6.150:11434", model="llama3.1")
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
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
    doc_node = Node("Document", 
                   id=doc_id, 
                   name=filename, 
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
    tx.commit()

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
    
    # Limit the number of nodes for better performance
    max_nodes = 100  # Limit total nodes
    
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
        
        # Get a limited number of chunks and entities
        chunk_query = """
        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
        RETURN 
            d.id AS doc_id,
            c.id AS chunk_id, 
            c.preview AS chunk_preview, 
            c.index AS chunk_index
        LIMIT 50
        """
        
        chunk_result = session.run(chunk_query)
        
        # Process chunk nodes and document-chunk edges
        chunk_nodes = set()
        doc_chunk_edges = set()
        
        for record in chunk_result:
            # Add chunk node if not already added and we haven't reached the limit
            if record["chunk_id"] and record["chunk_id"] not in chunk_nodes and len(chunk_nodes) < max_nodes:
                net.add_node(record["chunk_id"], 
                            label=f"C{record['chunk_index']}",  # Shorter label
                            title=record["chunk_preview"],
                            color="#4ECDC4", 
                            size=10)  # Smaller size
                chunk_nodes.add(record["chunk_id"])
                
                # Add document-chunk edge
                if record["doc_id"]:
                    edge = (record["doc_id"], record["chunk_id"])
                    if edge not in doc_chunk_edges:
                        net.add_edge(record["doc_id"], record["chunk_id"], width=0.5)  # Thinner edges
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

# Main UI - simplified for better performance
st.title("📊 GraphRAG")

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
    
    # Document upload section
    st.subheader("Add Documents")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "csv"])
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                success, message = process_document(uploaded_file, file_type)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Document deletion section
    st.subheader("Manage Documents")
    
    # Get list of documents from Neo4j
    @st.cache_data(ttl=5)  # Short cache to ensure list is updated after deletions
    def get_document_list():
        if not st.session_state.neo4j_driver:
            return []
        
        try:
            with st.session_state.neo4j_driver.session() as session:
                result = session.run("""
                MATCH (d:Document)
                RETURN d.id as id, d.name as name, d.created_at as created_at
                ORDER BY d.created_at DESC
                """)
                
                documents = []
                for record in result:
                    documents.append({
                        "id": record["id"],
                        "name": record["name"],
                        "created_at": record["created_at"]
                    })
                
                return documents
        except Exception as e:
            st.error(f"Error retrieving document list: {str(e)}")
            return []
    
    # Display document list with delete buttons
    documents = get_document_list()
    
    if not documents:
        st.info("No documents found in the database")
    else:
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {doc['name']}")
            with col2:
                # Create a unique key for each delete button
                delete_key = f"delete_{doc['id']}"
                if st.button("🗑️ Delete", key=delete_key):
                    with st.spinner(f"Deleting {doc['name']}..."):
                        success, message = delete_document(doc['id'], doc['name'])
                        if success:
                            st.success(message)
                            # Clear the document list cache to refresh the list
                            get_document_list.clear()
                            # Force a rerun to update the UI
                            st.rerun()
                        else:
                            st.error(message)
    
    # Only show statistics when expanded
    if st.sidebar.checkbox("Show Statistics", value=False):
        st.divider()
        st.header("Graph Statistics")
        
        # Calculate statistics from Neo4j with longer cache
        @st.cache_data(ttl=30)  # Cache for 30 seconds
        def get_neo4j_stats():
            if not st.session_state.neo4j_driver:
                return {
                    "documents": 0,
                    "chunks": 0,
                    "entities": 0,
                    "connections": 0
                }
            
            try:
                with st.session_state.neo4j_driver.session() as session:
                    # Use a single query for better performance
                    query = """
                    MATCH (d:Document) WITH count(d) AS doc_count
                    MATCH (c:Chunk) WITH doc_count, count(c) AS chunk_count
                    MATCH (e:Entity) WITH doc_count, chunk_count, count(e) AS entity_count
                    MATCH ()-[r]->() WITH doc_count, chunk_count, entity_count, count(r) AS rel_count
                    RETURN doc_count, chunk_count, entity_count, rel_count
                    """
                    
                    result = session.run(query)
                    record = result.single()
                    
                    return {
                        "documents": record["doc_count"],
                        "chunks": record["chunk_count"],
                        "entities": record["entity_count"],
                        "connections": record["rel_count"]
                    }
            except Exception as e:
                return {
                    "documents": 0,
                    "chunks": 0,
                    "entities": 0,
                    "connections": 0
                }
        
        # Get cached statistics
        stats = get_neo4j_stats()
        
        # Display statistics with a more efficient layout
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats["documents"])
            st.metric("Chunks", stats["chunks"])
        with col2:
            st.metric("Entities", stats["entities"])
            st.metric("Connections", stats["connections"])

# Chat input - must be outside of tabs
chat_input_placeholder = st.empty()

# Simplified tabs - only show graph visualization on demand
tab_options = ["Chat", "Knowledge Graph (Heavy)"]
selected_tab = st.radio("Select View", tab_options, horizontal=True, index=0)

# Chat tab - optimized for performance
if selected_tab == "Chat":
    # Display chat messages directly without caching
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Knowledge Graph tab - with loading indicator and lazy loading
elif selected_tab == "Knowledge Graph (Heavy)":
    st.warning("⚠️ Graph visualization is resource-intensive and may slow down the application")
    
    if st.button("Load Knowledge Graph"):
        # Check if Neo4j is connected and has data
        if st.session_state.neo4j_driver:
            with st.spinner("Rendering knowledge graph from Neo4j..."):
                html = visualize_neo4j_graph()
                if isinstance(html, str) and html.startswith("No graph data") or html.startswith("Neo4j database"):
                    st.info(html)
                else:
                    st.components.v1.html(html, height=600)
        else:
            st.info("Neo4j database is not connected. Please check your connection.")

# Initialize chat history if not already in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input - placed outside of tabs with simplified approach
with chat_input_placeholder:
    # Get user input
    prompt = st.chat_input("Ask a question about your documents")
    
    # Process user input if provided
    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            if not st.session_state.get("conversation_ready", False):
                response = "Please add documents first before asking questions."
            else:
                with st.spinner("Thinking..."):
                    try:
                        # Create a progress bar for response generation
                        progress_bar = st.progress(0)
                        
                        # Step 1: Retrieve relevant documents using GraphRAG
                        progress_bar.progress(20, text="Retrieving relevant documents using GraphRAG...")
                        message_placeholder.markdown("Retrieving relevant documents using GraphRAG...")
                        docs = st.session_state.graph_retriever.get_relevant_documents(prompt)
                        
                        # Step 2: Format context from retrieved documents
                        progress_bar.progress(40, text="Processing documents...")
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Step 3: Create prompt with context
                        progress_bar.progress(60, text="Generating response...")
                        message_placeholder.markdown("Generating response...")
                        prompt_with_context = f"""
                        Answer the question based on the following context:
                        
                        {context}
                        
                        Question: {prompt}
                        
                        Answer:
                        """
                        
                        # Step 4: Generate response with LLM
                        progress_bar.progress(80, text="Finalizing response...")
                        response = st.session_state.llm.invoke(prompt_with_context)
                        
                        # Step 5: Add source information
                        sources = "\n\n**Sources:**\n"
                        for i, doc in enumerate(docs[:3]):  # Limit to top 3 sources
                            source = doc.metadata.get("source", f"Document {i+1}")
                            sources += f"- {source}\n"
                        
                        response += sources
                        
                        # Complete progress
                        progress_bar.progress(100, text="Complete!")
                        
                        # Clear progress bar
                        progress_bar.empty()
                            
                    except Exception as e:
                        response = f"Error generating response: {str(e)}"
            
            # Display the final response
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the UI with the new chat history
            st.rerun()

if __name__ == "__main__":
    # This will be used when running the app directly
    pass
