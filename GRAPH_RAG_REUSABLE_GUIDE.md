# Graph RAG - Reusable Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Step-by-Step Implementation](#step-by-step-implementation)
   - [Setup & Configuration](#setup--configuration)
   - [Document Ingestion](#document-ingestion)
   - [Query & Retrieval](#query--retrieval)
5. [Code Examples](#code-examples)
6. [Integration Checklist](#integration-checklist)

---

## Overview

**Graph RAG** (Retrieval-Augmented Generation with Knowledge Graphs) structures document knowledge as a graph:
- Extracting **entities** and **relationships** from documents using LLM
- Storing them in a **Neo4j graph database**
- Enabling **graph-traversal search** that understands semantic connections
- Supporting multi-hop reasoning across related concepts

**When to use Graph RAG:**
- Documents with rich entity relationships (e.g., business processes, technical systems)
- Multi-hop reasoning queries (e.g., "How does A relate to B, which connects to C?")
- Need for explainable connections between concepts

---

## Core Concepts

### 1. Entity Extraction
Identify key entities in text:
- **Types**: Person, Organization, Location, Product, Concept, etc. (customize for your domain)
- **Properties**: Canonical name, type, confidence score
- **Example**: "customer onboarding" → `Customer_Onboarding` (Process)

### 2. Relation Extraction
Identify relationships between entities:
- **Types**: RELATED_TO, DEPENDS_ON, PART_OF, etc. (customize for your domain)
- **Properties**: Source, target, type, confidence
- **Example**: `Customer_Onboarding` → RELATED_TO → `Account_Setup`

### 3. Graph Storage (Neo4j)
Store entities as nodes and relations as edges:
```cypher
(Entity {name: "Customer_Onboarding", type: "Process"})
  -[RELATED_TO]->
(Entity {name: "Account_Setup", type: "Process"})
```

### 4. Graph Search
Traverse the graph to find related entities:
- **Direct lookup**: Find entities mentioned in query
- **Graph expansion**: Discover connected entities through relationships
- **Path finding**: Identify multi-hop connections

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Document Text                                                │
│       ↓                                                       │
│  1. Chunk Text (split into manageable pieces)                │
│       ↓                                                       │
│  2. LLM Entity Extraction (per chunk)                         │
│       ├─→ Entities (names, types)                            │
│       └─→ Relations (source→target, type)                    │
│       ↓                                                       │
│  3. Store in Neo4j                                            │
│       ├─→ Entity nodes                                       │
│       ├─→ Relationship edges                                 │
│       └─→ Link to source chunks                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  User Query                                                   │
│       ↓                                                       │
│  1. Query Understanding (LLM)                                 │
│       └─→ Extract entities mentioned in query                │
│       ↓                                                       │
│  2. Graph Search                                              │
│       ├─→ Find matching entities in graph                    │
│       └─→ Traverse relationships to find related entities    │
│       ↓                                                       │
│  3. Retrieve Source Chunks                                    │
│       └─→ Get text chunks mentioning these entities          │
│       ↓                                                       │
│  4. LLM Answer Generation                                     │
│       └─→ Use retrieved chunks + graph context               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Implementation

### Setup & Configuration

#### 1. Install Dependencies

```bash
pip install neo4j openai pydantic
```

#### 2. Configure Neo4j Connection

```python
# config.py
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
NEO4J_DATABASE = "neo4j"

# Graph extraction settings
GRAPH_EXTRACTION_MODEL = "gpt-4"  # or gpt-3.5-turbo
GRAPH_EXTRACTION_TEMPERATURE = 0.0
GRAPH_MIN_CONFIDENCE = 0.7
```

#### 3. Initialize Neo4j Client

```python
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def execute_write(self, query, parameters=None):
        with self.driver.session() as session:
            session.run(query, parameters or {})

# Initialize client
neo4j_client = Neo4jClient(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD
)
```

#### 4. Create Neo4j Indexes

```python
def setup_neo4j_indexes(client):
    """Create necessary indexes and constraints."""
    
    # Unique constraint on entity canonical_name
    client.execute_write("""
        CREATE CONSTRAINT entity_canonical_name IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE
    """)
    
    # Index on entity_type for filtering
    client.execute_write("""
        CREATE INDEX entity_type_idx IF NOT EXISTS
        FOR (e:Entity) ON (e.entity_type)
    """)
    
    # Index on confidence for filtering
    client.execute_write("""
        CREATE INDEX entity_confidence_idx IF NOT EXISTS
        FOR (e:Entity) ON (e.confidence)
    """)
    
    # Unique constraint on chunk ID
    client.execute_write("""
        CREATE CONSTRAINT chunk_id IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.id IS UNIQUE
    """)

# Setup indexes
setup_neo4j_indexes(neo4j_client)
```

---

### Document Ingestion

#### Step 1: Define Data Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Entity(BaseModel):
    """Extracted entity from text."""
    name: str  # Normalized entity name (e.g., "Machine_Learning")
    type: str  # Entity type (e.g., Concept, Person, Organization)
    confidence: float  # Confidence score 0.0 to 1.0
    chunk_id: Optional[str] = None

class Relation(BaseModel):
    """Extracted relation between entities."""
    source: str  # Source entity name
    target: str  # Target entity name
    relation_type: str  # Relation type (e.g., RELATED_TO, DEPENDS_ON)
    confidence: float  # Confidence score 0.0 to 1.0
```

#### Step 2: LLM Prompts for Extraction

```python
# System prompt for entity extraction - customize for your domain
GRAPH_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph builder. Extract entities and relationships from text.

## Instructions
1. Identify key entities (people, organizations, concepts, etc.)
2. Normalize entity names (e.g., "AI" and "Artificial Intelligence" → "Artificial_Intelligence")
3. Extract clear relationships between entities
4. Assign confidence scores (0.0-1.0) based on evidence clarity
5. Return results in JSON format

## Output Format
{
  "entities": [
    {
      "name": "Entity_Name",
      "type": "Person/Organization/Concept/etc",
      "confidence": 0.9
    }
  ],
  "relations": [
    {
      "source": "Entity_A",
      "target": "Entity_B",
      "relation_type": "RELATED_TO",
      "confidence": 0.85
    }
  ]
}

Be precise and conservative."""

# User prompt template
GRAPH_EXTRACTION_USER_PROMPT = """Extract entities and relationships from the following text.

**Text:**
{text}

Return only valid JSON."""
```

#### Step 3: Implement Entity Extractor

```python
import json
import re
from typing import List, Tuple, Dict, Any
from openai import AzureOpenAI

class GraphEntityExtractor:
    """Extracts entities and relations from text using LLM."""
    
    def __init__(self, openai_client, model="gpt-4", temperature=0.0):
        self.client = openai_client
        self.model = model
        self.temperature = temperature
    
    def extract(self, text: str, chunk_id: str) -> Tuple[List[Entity], List[Relation], Dict[str, int]]:
        """
        Extract entities and relations from text chunk.
        
        Args:
            text: Text to extract from
            chunk_id: Unique chunk identifier
        
        Returns:
            Tuple of (entities, relations, token_usage)
        """
        # Build user prompt
        user_prompt = GRAPH_EXTRACTION_USER_PROMPT.format(text=text)
        
        # Build messages
        messages = [
            {"role": "system", "content": GRAPH_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        response_text = response.choices[0].message.content
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Parse JSON response
        entities, relations = self._parse_response(response_text, chunk_id)
        
        return entities, relations, token_usage
    
    def _parse_response(self, response_text: str, chunk_id: str) -> Tuple[List[Entity], List[Relation]]:
        """Parse LLM JSON response into structured entities and relations."""
        # Extract JSON from response (handles markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(.*?)```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response_text
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {response_text[:200]}")
            return [], []
        
        # Parse entities
        entities = []
        for ent_data in data.get("entities", []):
            entity = Entity(
                name=ent_data.get("name", ""),
                type=ent_data.get("type", "Concept"),
                confidence=float(ent_data.get("confidence", 0.5)),
                chunk_id=chunk_id
            )
            entities.append(entity)
        
        # Parse relations
        relations = []
        for rel_data in data.get("relations", []):
            relation = Relation(
                source=rel_data.get("source", ""),
                target=rel_data.get("target", ""),
                relation_type=rel_data.get("relation_type", "RELATED_TO"),
                confidence=float(rel_data.get("confidence", 0.5))
            )
            relations.append(relation)
        
        return entities, relations
```

#### Step 4: Store Entities in Neo4j

```python
class GraphStore:
    """Handles Neo4j graph storage operations."""
    
    def __init__(self, neo4j_client):
        self.client = neo4j_client
    
    def upsert_entity(self, entity: Entity, doc_id: str):
        """
        Upsert entity node in Neo4j.
        Creates or updates entity, links to source chunk and document.
        """
        query = """
        // Merge entity node
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.confidence = $confidence,
            e.created_at = datetime()
        ON MATCH SET
            e.confidence = CASE 
                WHEN $confidence > e.confidence THEN $confidence 
                ELSE e.confidence 
            END,
            e.updated_at = datetime()
        
        // Link to chunk
        WITH e
        MERGE (c:Chunk {id: $chunk_id})
        MERGE (e)-[r:MENTIONED_IN]->(c)
        
        // Link to document
        WITH e, c
        MERGE (d:Document {id: $doc_id})
        MERGE (c)-[:PART_OF]->(d)
        
        RETURN e.name as name
        """
        
        params = {
            "name": entity.name,
            "type": entity.type,
            "confidence": entity.confidence,
            "chunk_id": entity.chunk_id,
            "doc_id": doc_id
        }
        
        self.client.execute_write(query, params)
    
    def upsert_relation(self, relation: Relation):
        """
        Upsert relationship between entities.
        """
        rel_type = relation.relation_type.upper().replace(" ", "_")
        
        query = f"""
        // Find source and target entities
        MATCH (source:Entity {{name: $source}})
        MATCH (target:Entity {{name: $target}})
        
        // Merge relationship
        MERGE (source)-[r:{rel_type}]->(target)
        ON CREATE SET
            r.confidence = $confidence,
            r.created_at = datetime()
        ON MATCH SET
            r.confidence = CASE 
                WHEN $confidence > r.confidence THEN $confidence 
                ELSE r.confidence 
            END,
            r.updated_at = datetime()
        
        RETURN type(r) as relation_type
        """
        
        params = {
            "source": relation.source,
            "target": relation.target,
            "confidence": relation.confidence
        }
        
        try:
            self.client.execute_write(query, params)
        except Exception as e:
            print(f"Failed to upsert relation {relation.source}->{relation.target}: {e}")
            # Don't raise - relations can fail if entities don't exist
```

#### Step 5: Complete Ingestion Pipeline

```python
def ingest_document(
    doc_id: str,
    text: str,
    extractor: GraphEntityExtractor,
    graph_store: GraphStore,
    chunk_size: int = 1000,
    min_confidence: float = 0.7
) -> Dict[str, Any]:
    """
    Complete pipeline to ingest document into graph.
    
    Args:
        doc_id: Unique document identifier
        text: Document text to process
        extractor: GraphEntityExtractor instance
        graph_store: GraphStore instance
        chunk_size: Chunk size in characters
        min_confidence: Minimum confidence threshold
    
    Returns:
        Dictionary with ingestion statistics
    """
    # Step 1: Chunk text
    chunks = chunk_text(text, chunk_size=chunk_size)
    print(f"Created {len(chunks)} chunks for doc {doc_id}")
    
    # Step 2: Extract entities & relations from each chunk
    all_entities = []
    all_relations = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}:{i}"
        
        # Extract
        entities, relations, tokens = extractor.extract(
            text=chunk_text,
            chunk_id=chunk_id
        )
        
        # Accumulate tokens
        for key in total_tokens:
            total_tokens[key] += tokens[key]
        
        # Filter by confidence
        entities = [e for e in entities if e.confidence >= min_confidence]
        relations = [r for r in relations if r.confidence >= min_confidence]
        
        all_entities.extend(entities)
        all_relations.extend(relations)
        
        print(f"Chunk {i}: {len(entities)} entities, {len(relations)} relations")
    
    # Step 3: Upsert entities to Neo4j
    print(f"Upserting {len(all_entities)} entities to Neo4j...")
    for entity in all_entities:
        graph_store.upsert_entity(entity, doc_id)
    
    # Step 4: Upsert relations to Neo4j
    print(f"Upserting {len(all_relations)} relations to Neo4j...")
    for relation in all_relations:
        graph_store.upsert_relation(relation)
    
    return {
        "doc_id": doc_id,
        "chunks_processed": len(chunks),
        "entities_extracted": len(all_entities),
        "relations_extracted": len(all_relations),
        "token_usage": total_tokens
    }


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Simple text chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks
```

#### Complete Ingestion Example

```python
# Initialize components
openai_client = AzureOpenAI(
    api_key="your_api_key",
    api_version="2024-02-15-preview",
    azure_endpoint="your_endpoint"
)

neo4j_client = Neo4jClient(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

extractor = GraphEntityExtractor(openai_client, model="gpt-4")
graph_store = GraphStore(neo4j_client)

# Ingest a document
document_text = """
Machine learning is a subset of artificial intelligence.
Deep learning is a type of machine learning that uses neural networks.
Neural networks are inspired by biological neurons in the brain.
"""

result = ingest_document(
    doc_id="doc-001",
    text=document_text,
    extractor=extractor,
    graph_store=graph_store,
    chunk_size=500,
    min_confidence=0.7
)

print(f"Ingestion complete: {result}")
```

---

### Query & Retrieval

#### Step 1: Query Understanding with LLM

```python
# Simple query understanding prompt
QUERY_UNDERSTANDING_PROMPT = """Extract key entities from this query that might exist in a knowledge graph.

Query: {query}

Return JSON with entities:
{
  "entities": ["Entity_Name_1", "Entity_Name_2"]
}"""


def understand_query(
    query: str,
    openai_client,
    model: str = "gpt-4"
) -> List[str]:
    """
    Use LLM to extract entities mentioned in query.
    
    Args:
        query: User's search query
        openai_client: OpenAI client
        model: LLM model to use
    
    Returns:
        List of entity names
    """
    prompt = QUERY_UNDERSTANDING_PROMPT.format(query=query)
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    response_text = response.choices[0].message.content
    
    # Parse JSON
    try:
        data = json.loads(response_text)
        return data.get("entities", [])
    except:
        return []
```

#### Step 2: Graph Traversal Search

```python
def graph_search(
    entity_names: List[str],
    neo4j_client,
    max_hops: int = 2
) -> List[Dict[str, Any]]:
    """
    Search graph by traversing from query entities.
    
    Args:
        entity_names: Entity names from query
        neo4j_client: Neo4j client
        max_hops: Maximum relationship hops (default: 2)
    
    Returns:
        List of relevant chunks with graph context
    """
    if not entity_names:
        return []
    
    # Cypher query to find entities and traverse relationships
    query = f"""
    // Find starting entities
    MATCH (start:Entity)
    WHERE start.name IN $entity_names
    
    // Traverse relationships (up to max_hops)
    MATCH path = (start)-[*0..{max_hops}]-(related:Entity)
    
    // Find chunks mentioning these entities
    MATCH (related)-[:MENTIONED_IN]->(chunk:Chunk)
    
    // Return chunk IDs and entity context
    RETURN DISTINCT
        chunk.id as chunk_id,
        collect(DISTINCT related.name) as entities,
        [rel IN relationships(path) | type(rel)] as path
    LIMIT 50
    """
    
    params = {"entity_names": entity_names}
    results = neo4j_client.execute_query(query, params)
    
    return results
```

#### Step 3: Retrieve Chunk Text

```python
def get_chunk_text(chunk_ids: List[str], storage) -> Dict[str, str]:
    """
    Retrieve actual text content for chunks.
    This would connect to your document storage (database, Redis, etc.)
    
    Args:
        chunk_ids: List of chunk IDs
        storage: Your storage client
    
    Returns:
        Dictionary mapping chunk_id to text content
    """
    # Implementation depends on your storage system
    # Example: return storage.get_chunks(chunk_ids)
    pass
```

#### Step 4: Generate Answer with Graph Context

```python
# Simple answer generation prompt
ANSWER_PROMPT = """Answer this question using only the provided information.

Question: {query}

Retrieved Information:
{chunks}

Entities mentioned: {entities}

Answer:"""


def generate_answer(
    query: str,
    chunk_results: List[Dict[str, Any]],
    chunk_texts: Dict[str, str],
    openai_client,
    model: str = "gpt-4"
) -> str:
    """
    Generate answer using LLM with chunks and graph context.
    
    Args:
        query: User's question
        chunk_results: Results from graph_search
        chunk_texts: Actual text content for chunks
        openai_client: OpenAI client
        model: LLM model to use
    
    Returns:
        Generated answer
    """
    # Format chunks
    chunks_text = []
    all_entities = set()
    
    for result in chunk_results:
        chunk_id = result["chunk_id"]
        text = chunk_texts.get(chunk_id, "")
        chunks_text.append(f"[{chunk_id}] {text}")
        all_entities.update(result.get("entities", []))
    
    chunks_str = "\n\n".join(chunks_text)
    entities_str = ", ".join(all_entities)
    
    # Build prompt
    prompt = ANSWER_PROMPT.format(
        query=query,
        chunks=chunks_str,
        entities=entities_str
    )
    
    # Generate answer
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content
```

#### Complete Query Example

```python
# Complete graph search pipeline
query = "What is the relationship between machine learning and neural networks?"

# Step 1: Understand query
entity_names = understand_query(query, openai_client)
print(f"Query entities: {entity_names}")

# Step 2: Graph search
graph_results = graph_search(entity_names, neo4j_client, max_hops=2)
print(f"Found {len(graph_results)} chunks")

# Step 3: Get chunk texts
chunk_ids = [r["chunk_id"] for r in graph_results]
chunk_texts = get_chunk_text(chunk_ids, your_storage)

# Step 4: Generate answer
answer = generate_answer(
    query=query,
    chunk_results=graph_results,
    chunk_texts=chunk_texts,
    openai_client=openai_client,
    model="gpt-4"
)

print(f"\nAnswer:\n{answer}")
```

---

## Integration Checklist

### Prerequisites
- [ ] Neo4j database running (bolt://localhost:7687)
- [ ] OpenAI or Azure OpenAI API access
- [ ] Python 3.8+ with required packages

### Setup Steps
1. [ ] Install dependencies: `pip install neo4j openai pydantic`
2. [ ] Configure Neo4j connection settings
3. [ ] Initialize Neo4j client and create indexes
4. [ ] Set up LLM client (OpenAI/Azure)
5. [ ] Initialize extractor and graph store

### Ingestion Pipeline
1. [ ] Implement text chunking logic
2. [ ] Configure extraction prompts for your domain
3. [ ] Set up entity extractor with LLM
4. [ ] Implement graph store methods (upsert_entity, upsert_relation)
5. [ ] Create ingestion pipeline function
6. [ ] Test with sample documents

### Query Pipeline
1. [ ] Implement query understanding (entity extraction from query)
2. [ ] Implement graph traversal search
3. [ ] Set up chunk text retrieval from your storage
4. [ ] Implement answer generation with graph context
5. [ ] Test with sample queries

### Optimization
- [ ] Add error handling and retry logic for LLM calls
- [ ] Implement batch processing for large documents
- [ ] Monitor token usage and costs
- [ ] Add logging for debugging
- [ ] Cache frequently accessed entities

---

## Key Takeaways

1. **Entity Extraction is Critical**: Quality of extracted entities directly impacts search accuracy. Customize prompts for your domain.

2. **Normalize Entity Names**: Use consistent naming (e.g., "Machine_Learning" not "ML" or "machine learning") to build a coherent graph.

3. **Graph Traversal Enables Discovery**: Multi-hop searches can find related concepts that weren't explicitly mentioned in the query.

4. **Start Simple**: Begin with basic entity and relation types. Expand as you understand your domain better.

5. **Monitor LLM Costs**: Entity extraction uses LLM for every chunk. Consider batch processing and caching for large documents.

6. **Test Iteratively**: Start with small documents, validate results, then scale up.

---

## Additional Resources

- Neo4j Cypher Documentation: https://neo4j.com/docs/cypher-manual/
- OpenAI API Documentation: https://platform.openai.com/docs/

---

**Version**: 1.0  
**Last Updated**: December 2025
