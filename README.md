# Context Graph Demo

A demonstration project showing how to build and use **Context Graphs** with Neo4j for AI-powered decision tracing in financial institutions.

## What is a Context Graph?

Context Graphs are the infrastructure layer that enables AI agents to make enterprise-grade decisions. They capture **decision traces** - the tribal knowledge, exception logic, and reasoning that traditionally lives only in human experts' heads.

Key concepts:
- **State Clock**: What is true now (traditional databases)
- **Event Clock**: What happened, when, and why (context graphs)
- **Decision Traces**: Full reasoning, risk factors, confidence scores, and causal relationships

## Key Features

This demo showcases three key differentiators of Neo4j for context graphs:

### 1. Data Model Fit
Neo4j is the natural substrate for context graphs - entities, decisions, and causal relationships map directly to nodes and relationships.

### 2. GDS Magic with FastRP
Graph Data Science algorithms provide **structural/topological similarity** that's impossible on Postgres without a PhD in graph theory:
- **FastRP**: 75,000x faster than node2vec, captures structural embeddings
- **KNN**: Find k-nearest neighbors based on graph structure
- **Node Similarity**: Compare neighborhood patterns (fraud detection)
- **Louvain**: Community detection for decision clusters
- **PageRank**: Influence scoring for decision impact

### 3. Vector Embeddings + Graph
Combine semantic similarity (text embeddings) with structural similarity (FastRP) for hybrid search that understands both meaning and context.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│  FastAPI + SDK  │────▶│  Neo4j + GDS    │
│  Chakra UI v3   │     │  Claude Agent   │     │  Vector Search  │
│  NVL Graphs     │     │  9 MCP Tools    │     │  FastRP/KNN     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+
- Neo4j AuraDS instance (or local Neo4j Enterprise with GDS plugin)
- Anthropic API Key
- OpenAI API Key (for embeddings)

## Quick Start

### 1. Clone and Setup Environment

```bash
cd context-graph

# Create environment file
cat > .env << 'EOF'
# Neo4j Connection
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Anthropic API Key (for Claude Agent SDK)
ANTHROPIC_API_KEY=your_anthropic_key

# OpenAI API Key (for text embeddings)
OPENAI_API_KEY=your_openai_key
EOF
```

### 2. Install Backend Dependencies

```bash
cd backend

# Create virtual environment and install dependencies with uv
uv venv
uv pip install -e .
```

### 3. Generate Sample Data

```bash
cd backend
source .venv/bin/activate
export $(grep -v '^#' ../.env | xargs)
python scripts/generate_sample_data.py
```

This creates:
- 200 persons
- 350 accounts  
- 2000 transactions
- 600 decisions with causal chains
- 50 organizations
- 30 employees
- 15 policies

### 4. Start Backend

```bash
cd backend
source .venv/bin/activate
export $(grep -v '^#' ../.env | xargs)
uvicorn app.main:app --port 8000
```

Backend runs at http://localhost:8000

### 5. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:3000

## Using Neo4j AuraDS (Recommended)

For the best experience with GDS algorithms:

1. Create an AuraDS instance at https://console.neo4j.io
2. Note the connection URI (format: `neo4j+s://xxxx.databases.neo4j.io`)
3. Update your `.env` with the connection details

AuraDS includes all GDS algorithms (FastRP, KNN, Node Similarity, Louvain, PageRank) without additional configuration.

## Using Local Neo4j with Docker

If you have a Neo4j Enterprise license:

```bash
docker-compose up -d
```

Wait for Neo4j to be ready at http://localhost:7474, then update `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## Demo Scenarios

### Scenario 1: Credit Decision with Precedent Lookup

```
User: "Should we approve a $50K credit line increase for John Smith?"

Agent:
1. Searches customer profile
2. Finds similar past decisions via FastRP embeddings
3. Cites relevant precedents
4. Makes recommendation with confidence score
5. Records decision trace with full reasoning
```

### Scenario 2: Fraud Pattern Detection

```
User: "Analyze account #12345 for fraud patterns"

Agent:
1. Uses FastRP embeddings to compare with known fraud typologies
2. Applies Node Similarity to find structural matches
3. Visualizes suspicious patterns in graph
4. Returns risk score with explanation
```

### Scenario 3: Exception Request with Audit Trail

```
User: "We need to override the trading limit for XYZ Corp"

Agent:
1. Finds past exceptions and their outcomes
2. Traces causal chain of similar exceptions
3. Records exception with justification and precedent links
4. Links to applicable policies
```


Best demo messages:**

1. **Credit Decision with Precedent Lookup:**
```
Should we approve a credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase.
```
(Jessica Norris has the highest risk score at 0.77, which will trigger precedent searches and policy checks)

2. **Fraud Pattern Analysis:**
```
Analyze the fraud decision history for Jacob Fitzpatrick and find similar cases
```
(High-risk customer with multiple source systems)

3. **Causal Chain Investigation:**
```
What decisions have been made about Amanda Smith and what caused them?
```
(She has the most decisions - 7 total - good for showing decision traces)

4. **Policy-Based Decision:**
```
A customer wants to make a $15,000 wire transfer. What policies apply and are there similar past decisions?
```
(Triggers High-Value Transaction Review policy and precedent search)

5. **Exception Request:**
```
We need to override the trading limit for Katherine Miller. Find precedents for similar exceptions.
```
(Shows exception handling with precedent lookup)

**The most compelling single message that demonstrates multiple benefits:**

```
Should we approve a credit limit increase for Jessica Norris? She has a high risk score and is requesting $25,000. Find similar past decisions and applicable policies.
```

This message will:
- Search for the customer (demonstrating entity lookup)
- Find her risk score and decision history
- Search for precedent decisions using semantic/structural similarity
- Look up applicable policies (Credit Limit Policy)
- Show the causal reasoning chain

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /api/customers/search?query=` | Search customers |
| `GET /api/customers/{id}` | Get customer details |
| `GET /api/customers/{id}/decisions` | Get customer decisions |
| `GET /api/decisions/{id}` | Get decision details |
| `GET /api/decisions/{id}/similar` | Find similar decisions |
| `GET /api/decisions/{id}/causal-chain` | Get causal chain |
| `GET /api/graph` | Get graph visualization data |
| `GET /api/graph/statistics` | Get graph statistics |
| `GET /api/policies` | List policies |
| `POST /api/chat` | Chat with AI agent |

## Project Structure

```
context-graph/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── agent.py             # Claude Agent SDK with MCP tools
│   │   ├── context_graph_client.py  # Neo4j operations
│   │   ├── gds_client.py        # GDS algorithms
│   │   ├── vector_client.py     # Vector search
│   │   └── models/              # Pydantic models
│   └── scripts/
│       └── generate_sample_data.py
├── frontend/
│   ├── app/                     # Next.js pages
│   ├── components/
│   │   ├── ChatInterface.tsx    # AI chat with inline graphs
│   │   ├── ContextGraphView.tsx # NVL visualization
│   │   └── DecisionTracePanel.tsx
│   └── lib/
│       └── api.ts               # API client
├── cypher/
│   ├── schema.cypher            # Neo4j schema
│   └── gds_projections.cypher   # GDS algorithms
├── docker-compose.yml           # Local Neo4j setup
└── .env                         # Environment variables
```

## Agent Tools (MCP)

The Claude Agent has access to 9 custom tools:

| Tool | Description |
|------|-------------|
| `search_customer` | Search customers by name, email, account number |
| `get_customer_decisions` | Get all decisions about a customer |
| `find_similar_decisions` | FastRP-based structural similarity search |
| `find_precedents` | Semantic + structural precedent search |
| `get_causal_chain` | Trace causes and effects of a decision |
| `record_decision` | Create new decision trace with reasoning |
| `detect_fraud_patterns` | Graph-based fraud analysis |
| `get_policy` | Get current policy rules |
| `execute_cypher` | Read-only Cypher for custom analysis |
| `get_schema` | Retrieve the current Neo4j schema |

## Neo4j Data Model

### Core Entities
- `Person` - Customer/employee with FastRP embedding
- `Account` - Bank/trading accounts with risk tiers
- `Transaction` - Financial transactions with fraud scores
- `Organization` - Companies with sanctions status

### Decision Trace Nodes
- `Decision` - Core decision with reasoning, confidence
- `DecisionContext` - State snapshot at decision time
- `Precedent` - Links to similar past decisions
- `Policy` - Rules governing decisions
- `Exception` - Documented exceptions

### Key Relationships
```
(:Decision)-[:CAUSED]->(:Decision)
(:Decision)-[:INFLUENCED]->(:Decision)
(:Decision)-[:PRECEDENT_FOR]->(:Decision)
(:Decision)-[:ABOUT]->(:Person|:Account|:Transaction)
(:Decision)-[:APPLIED_POLICY]->(:Policy)
(:Decision)-[:GRANTED_EXCEPTION]->(:Exception)
```

## GDS Algorithms

### FastRP Embeddings
```cypher
CALL gds.fastRP.mutate('decision-graph', {
  embeddingDimension: 128,
  iterationWeights: [0.0, 1.0, 1.0, 0.8, 0.6],
  mutateProperty: 'fastrp_embedding'
})
```

### Hybrid Search (Semantic + Structural)
```cypher
// Find decisions similar in both meaning and graph structure
CALL db.index.vector.queryNodes('decision_semantic_idx', 10, $query_embedding)
YIELD node, score AS semantic_score
WITH node, semantic_score
CALL db.index.vector.queryNodes('decision_fastrp_idx', 10, node.fastrp_embedding)
YIELD node AS similar, score AS structural_score
RETURN similar, (semantic_score + structural_score) / 2 AS combined_score
ORDER BY combined_score DESC
```

## References

- [AI's Trillion-Dollar Opportunity: Context Graphs](https://foundationcapital.com/context-graphs-ais-trillion-dollar-opportunity/) - Foundation Capital
- [Neo4j FastRP Documentation](https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/fastrp/)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- [NVL Visualization Library](https://neo4j.com/docs/nvl/)

## License

MIT
