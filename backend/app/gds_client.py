"""
Neo4j Graph Data Science (GDS) client.
Implements FastRP, KNN, Node Similarity, Louvain, and PageRank.
"""

from typing import Optional

from neo4j import GraphDatabase

from .config import config


class GDSClient:
    """Neo4j GDS client for graph algorithms."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.username, config.neo4j.password),
        )
        self.database = config.neo4j.database
        self.fastrp_dimensions = config.fastrp_dimensions

    def close(self):
        self.driver.close()

    # ============================================
    # GRAPH PROJECTION MANAGEMENT
    # ============================================

    def create_decision_graph_projection(self) -> dict:
        """Create the decision graph projection for GDS algorithms."""
        with self.driver.session(database=self.database) as session:
            # Drop if exists
            session.run("CALL gds.graph.drop('decision-graph', false) YIELD graphName")

            result = session.run(
                """
                CALL gds.graph.project(
                    'decision-graph',
                    ['Decision', 'Person', 'Account', 'Transaction', 'Organization', 'Policy', 'Employee'],
                    {
                        ABOUT: {orientation: 'UNDIRECTED'},
                        CAUSED: {orientation: 'NATURAL', properties: ['confidence']},
                        INFLUENCED: {orientation: 'NATURAL', properties: ['weight']},
                        PRECEDENT_FOR: {orientation: 'NATURAL', properties: ['similarity_score']},
                        OWNS: {orientation: 'UNDIRECTED'},
                        MADE_BY: {orientation: 'NATURAL'},
                        APPLIED_POLICY: {orientation: 'NATURAL'},
                        FROM_ACCOUNT: {orientation: 'NATURAL'},
                        TO_ACCOUNT: {orientation: 'NATURAL'}
                    }
                ) YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """
            )
            record = result.single()
            return dict(record) if record else {}

    def create_entity_graph_projection(self) -> dict:
        """Create the entity graph projection for fraud detection."""
        with self.driver.session(database=self.database) as session:
            # Drop if exists
            session.run("CALL gds.graph.drop('entity-graph', false) YIELD graphName")

            result = session.run(
                """
                CALL gds.graph.project(
                    'entity-graph',
                    ['Person', 'Account', 'Transaction', 'Organization'],
                    {
                        OWNS: {orientation: 'UNDIRECTED'},
                        FROM_ACCOUNT: {orientation: 'NATURAL'},
                        TO_ACCOUNT: {orientation: 'NATURAL'},
                        INVOLVING: {orientation: 'UNDIRECTED'},
                        RELATED_TO: {orientation: 'UNDIRECTED'}
                    }
                ) YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """
            )
            record = result.single()
            return dict(record) if record else {}

    def list_graph_projections(self) -> list[dict]:
        """List all graph projections."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.graph.list()
                YIELD graphName, nodeCount, relationshipCount, creationTime
                RETURN graphName, nodeCount, relationshipCount, creationTime
                """
            )
            return [dict(record) for record in result]

    # ============================================
    # FASTRP EMBEDDINGS
    # ============================================

    def generate_fastrp_embeddings(
        self,
        graph_name: str = "decision-graph",
        node_labels: Optional[list[str]] = None,
    ) -> dict:
        """Generate FastRP embeddings for nodes."""
        node_labels = node_labels or ["Decision", "Person", "Account", "Transaction"]

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.fastRP.mutate($graph_name, {
                    embeddingDimension: $dimensions,
                    iterationWeights: [0.0, 1.0, 1.0, 0.8, 0.6],
                    normalizationStrength: 0.5,
                    randomSeed: 42,
                    mutateProperty: 'fastrp_embedding'
                }) YIELD nodePropertiesWritten, computeMillis
                RETURN nodePropertiesWritten, computeMillis
                """,
                {
                    "graph_name": graph_name,
                    "dimensions": self.fastrp_dimensions,
                },
            )
            record = result.single()
            return dict(record) if record else {}

    def write_fastrp_embeddings(
        self,
        graph_name: str = "decision-graph",
        node_labels: Optional[list[str]] = None,
    ) -> dict:
        """Write FastRP embeddings back to the database."""
        node_labels = node_labels or ["Decision", "Person", "Account", "Transaction"]

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.graph.nodeProperties.write($graph_name, ['fastrp_embedding'], $node_labels)
                YIELD propertiesWritten
                RETURN propertiesWritten
                """,
                {"graph_name": graph_name, "node_labels": node_labels},
            )
            record = result.single()
            return dict(record) if record else {}

    # ============================================
    # K-NEAREST NEIGHBORS (KNN)
    # ============================================

    def find_similar_decisions_knn(
        self,
        decision_id: str,
        limit: int = 10,
        graph_name: str = "decision-graph",
    ) -> list[dict]:
        """Find similar decisions using KNN on FastRP embeddings."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.knn.stream($graph_name, {
                    nodeLabels: ['Decision'],
                    nodeProperties: ['fastrp_embedding'],
                    topK: $limit,
                    sampleRate: 1.0,
                    randomSeed: 42
                }) YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) AS decision1, gds.util.asNode(node2) AS decision2, similarity
                WHERE decision1.id = $decision_id
                RETURN decision2.id AS id,
                       decision2.decision_type AS decision_type,
                       decision2.category AS category,
                       decision2.reasoning_summary AS reasoning_summary,
                       decision2.decision_timestamp AS decision_timestamp,
                       similarity
                ORDER BY similarity DESC
                """,
                {
                    "graph_name": graph_name,
                    "decision_id": decision_id,
                    "limit": limit,
                },
            )
            return [dict(record) for record in result]

    def run_knn_all(
        self,
        graph_name: str = "decision-graph",
        node_label: str = "Decision",
        top_k: int = 5,
    ) -> dict:
        """Run KNN on all nodes and create SIMILAR_TO relationships."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.knn.mutate($graph_name, {
                    nodeLabels: [$node_label],
                    nodeProperties: ['fastrp_embedding'],
                    topK: $top_k,
                    mutateRelationshipType: 'SIMILAR_TO',
                    mutateProperty: 'score'
                }) YIELD relationshipsWritten, computeMillis
                RETURN relationshipsWritten, computeMillis
                """,
                {
                    "graph_name": graph_name,
                    "node_label": node_label,
                    "top_k": top_k,
                },
            )
            record = result.single()
            return dict(record) if record else {}

    # ============================================
    # NODE SIMILARITY
    # ============================================

    def find_similar_accounts(
        self,
        account_id: str,
        limit: int = 10,
        similarity_cutoff: float = 0.5,
        graph_name: str = "entity-graph",
    ) -> list[dict]:
        """Find accounts with similar neighborhood structures."""
        # Ensure the graph projection exists
        if graph_name == "entity-graph":
            self._ensure_entity_graph_exists()

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.nodeSimilarity.stream($graph_name, {
                    nodeLabels: ['Account'],
                    topK: $limit,
                    similarityCutoff: $cutoff
                }) YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) AS account1, gds.util.asNode(node2) AS account2, similarity
                WHERE account1.id = $account_id
                RETURN account2.id AS id,
                       account2.account_number AS account_number,
                       account2.account_type AS account_type,
                       account2.risk_tier AS risk_tier,
                       similarity
                ORDER BY similarity DESC
                """,
                {
                    "graph_name": graph_name,
                    "account_id": account_id,
                    "limit": limit,
                    "cutoff": similarity_cutoff,
                },
            )
            return [dict(record) for record in result]

    def find_potential_duplicates(
        self,
        similarity_cutoff: float = 0.7,
        graph_name: str = "entity-graph",
    ) -> list[dict]:
        """Find potential duplicate persons using Node Similarity."""
        # Ensure the graph projection exists
        if graph_name == "entity-graph":
            self._ensure_entity_graph_exists()

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.nodeSimilarity.stream($graph_name, {
                    nodeLabels: ['Person'],
                    topK: 10,
                    similarityCutoff: $cutoff
                }) YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) AS person1, gds.util.asNode(node2) AS person2, similarity
                WHERE person1.id < person2.id
                RETURN person1.id AS person1_id,
                       person1.name AS person1_name,
                       person1.source_systems AS person1_sources,
                       person2.id AS person2_id,
                       person2.name AS person2_name,
                       person2.source_systems AS person2_sources,
                       similarity
                ORDER BY similarity DESC
                LIMIT 20
                """,
                {"graph_name": graph_name, "cutoff": similarity_cutoff},
            )
            return [dict(record) for record in result]

    # ============================================
    # FRAUD PATTERN DETECTION
    # ============================================

    def _ensure_entity_graph_exists(self) -> None:
        """Ensure the entity-graph projection exists, creating it if necessary."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.graph.exists('entity-graph') YIELD exists
                RETURN exists
                """
            )
            record = result.single()
            if not record or not record["exists"]:
                self.create_entity_graph_projection()

    def detect_fraud_patterns(
        self,
        account_id: Optional[str] = None,
        similarity_threshold: float = 0.7,
        graph_name: str = "entity-graph",
    ) -> list[dict]:
        """Detect accounts with similar structures to known fraud cases."""
        # Ensure the graph projection exists
        if graph_name == "entity-graph":
            self._ensure_entity_graph_exists()

        with self.driver.session(database=self.database) as session:
            if account_id:
                # Check specific account against fraud patterns
                result = session.run(
                    """
                    MATCH (target:Account {id: $account_id})
                    MATCH (fraud:Account)-[:FROM_ACCOUNT|TO_ACCOUNT]-(t:Transaction)
                    WHERE t.status = 'flagged'
                    WITH target, fraud, count(t) AS flagged_count
                    WHERE flagged_count >= 2
                    CALL gds.nodeSimilarity.stream($graph_name, {
                        nodeLabels: ['Account'],
                        topK: 5,
                        similarityCutoff: $threshold
                    }) YIELD node1, node2, similarity
                    WITH target, fraud, gds.util.asNode(node1) AS a1, gds.util.asNode(node2) AS a2, similarity
                    WHERE (a1.id = target.id AND a2.id = fraud.id)
                       OR (a1.id = fraud.id AND a2.id = target.id)
                    RETURN DISTINCT target.id AS target_id,
                           target.account_number AS target_account,
                           fraud.id AS fraud_case_id,
                           fraud.account_number AS fraud_account,
                           similarity AS structural_similarity
                    ORDER BY similarity DESC
                    """,
                    {
                        "graph_name": graph_name,
                        "account_id": account_id,
                        "threshold": similarity_threshold,
                    },
                )
            else:
                # Find all accounts similar to known fraud cases
                result = session.run(
                    """
                    MATCH (fraud:Account)-[:FROM_ACCOUNT|TO_ACCOUNT]-(t:Transaction)
                    WHERE t.status = 'flagged'
                    WITH fraud, count(t) AS flagged_count
                    WHERE flagged_count >= 2
                    WITH collect(fraud) AS fraud_accounts
                    CALL gds.nodeSimilarity.stream($graph_name, {
                        nodeLabels: ['Account'],
                        topK: 10,
                        similarityCutoff: $threshold
                    }) YIELD node1, node2, similarity
                    WITH fraud_accounts, gds.util.asNode(node1) AS a1, gds.util.asNode(node2) AS a2, similarity
                    WHERE a1 IN fraud_accounts AND NOT a2 IN fraud_accounts
                    RETURN a2.id AS suspect_id,
                           a2.account_number AS suspect_account,
                           a2.risk_tier AS current_risk_tier,
                           a1.id AS similar_fraud_id,
                           similarity AS structural_similarity
                    ORDER BY similarity DESC
                    LIMIT 20
                    """,
                    {"graph_name": graph_name, "threshold": similarity_threshold},
                )
            return [dict(record) for record in result]

    # ============================================
    # LOUVAIN COMMUNITY DETECTION
    # ============================================

    def detect_decision_communities(
        self,
        graph_name: str = "decision-graph",
    ) -> list[dict]:
        """Detect communities of related decisions using Louvain."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.louvain.stream($graph_name, {
                    nodeLabels: ['Decision'],
                    relationshipTypes: ['CAUSED', 'INFLUENCED', 'PRECEDENT_FOR']
                }) YIELD nodeId, communityId
                WITH gds.util.asNode(nodeId) AS decision, communityId
                WITH communityId,
                     count(decision) AS decision_count,
                     collect(DISTINCT decision.decision_type) AS decision_types,
                     collect(DISTINCT decision.category) AS categories,
                     collect(decision.id)[0..5] AS sample_decision_ids
                ORDER BY decision_count DESC
                LIMIT 20
                RETURN communityId, decision_count, decision_types, categories, sample_decision_ids
                """,
                {"graph_name": graph_name},
            )
            return [dict(record) for record in result]

    def write_community_ids(
        self,
        graph_name: str = "decision-graph",
    ) -> dict:
        """Write community IDs to decision nodes."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.louvain.mutate($graph_name, {
                    nodeLabels: ['Decision'],
                    relationshipTypes: ['CAUSED', 'INFLUENCED', 'PRECEDENT_FOR'],
                    mutateProperty: 'community_id'
                }) YIELD communityCount, modularity, computeMillis
                RETURN communityCount, modularity, computeMillis
                """,
                {"graph_name": graph_name},
            )
            record = result.single()
            return dict(record) if record else {}

    # ============================================
    # PAGERANK - INFLUENCE SCORING
    # ============================================

    def calculate_influence_scores(
        self,
        graph_name: str = "decision-graph",
    ) -> list[dict]:
        """Calculate PageRank influence scores for decisions."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.pageRank.stream($graph_name, {
                    nodeLabels: ['Decision'],
                    relationshipTypes: ['CAUSED', 'INFLUENCED'],
                    maxIterations: 20,
                    dampingFactor: 0.85
                }) YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS decision, score
                WHERE decision.decision_type IN ['exception', 'override', 'escalation']
                RETURN decision.id AS id,
                       decision.decision_type AS decision_type,
                       decision.category AS category,
                       decision.reasoning_summary AS reasoning_summary,
                       score AS influence_score
                ORDER BY score DESC
                LIMIT 20
                """,
                {"graph_name": graph_name},
            )
            return [dict(record) for record in result]

    def write_influence_scores(
        self,
        graph_name: str = "decision-graph",
    ) -> dict:
        """Write PageRank scores to decision nodes."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL gds.pageRank.mutate($graph_name, {
                    nodeLabels: ['Decision'],
                    relationshipTypes: ['CAUSED', 'INFLUENCED'],
                    mutateProperty: 'influence_score'
                }) YIELD nodePropertiesWritten, computeMillis
                RETURN nodePropertiesWritten, computeMillis
                """,
                {"graph_name": graph_name},
            )
            record = result.single()
            return dict(record) if record else {}


# Singleton instance
gds_client = GDSClient()
