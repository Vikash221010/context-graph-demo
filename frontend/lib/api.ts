import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Types
export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
}

export interface GraphRelationship {
  id: string;
  type: string;
  startNodeId: string;
  endNodeId: string;
  properties: Record<string, unknown>;
}

export interface GraphData {
  nodes: GraphNode[];
  relationships: GraphRelationship[];
}

export interface Decision {
  id: string;
  decision_type: string;
  category: string;
  reasoning: string;
  reasoning_summary?: string;
  outcome?: string;
  confidence?: number;
  confidence_score?: number;
  risk_factors: string[];
  timestamp?: string;
  decision_timestamp?: string;
  made_by?: string;
  status: string;
  source_system?: string;
  target_types?: string[];
}

export interface Customer {
  id: string;
  first_name: string;
  last_name: string;
  email: string;
  risk_score: number;
  customer_since: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  tool_calls?: Array<{
    name: string;
    input: Record<string, unknown>;
    output?: unknown;
  }>;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  tool_calls: Array<{
    name: string;
    input: Record<string, unknown>;
    output?: unknown;
  }>;
  decisions_made: string[];
}

export interface SimilarDecision {
  decision: Decision;
  similarity_score: number;
  similarity_type: string;
}

export interface CausalChain {
  decision_id: string;
  causes: Decision[];
  effects: Decision[];
  depth: number;
}

// API Functions

// Chat
export async function sendChatMessage(
  message: string,
  conversationHistory: ChatMessage[] = [],
): Promise<ChatResponse> {
  const response = await api.post("/api/chat", {
    message,
    conversation_history: conversationHistory,
  });
  return response.data;
}

// Agent context for transparency/debugging
export interface AgentContext {
  system_prompt: string;
  model: string;
  available_tools: string[];
  mcp_server: string;
}

// Streaming chat event types
export interface StreamAgentContextEvent {
  type: "agent_context";
  context: AgentContext;
}

export interface StreamTextEvent {
  type: "text";
  content: string;
}

export interface StreamToolUseEvent {
  type: "tool_use";
  name: string;
  input: Record<string, unknown>;
}

export interface StreamToolResultEvent {
  type: "tool_result";
  name: string;
  output: unknown;
}

export interface StreamDoneEvent {
  type: "done";
  tool_calls: Array<{
    name: string;
    input: Record<string, unknown>;
    output?: unknown;
  }>;
  decisions_made: string[];
}

export interface StreamErrorEvent {
  type: "error";
  error: string;
}

export type StreamEvent =
  | StreamAgentContextEvent
  | StreamTextEvent
  | StreamToolUseEvent
  | StreamToolResultEvent
  | StreamDoneEvent
  | StreamErrorEvent;

// Streaming chat with SSE
export async function* streamChatMessage(
  message: string,
  conversationHistory: ChatMessage[] = [],
): AsyncGenerator<StreamEvent, void, unknown> {
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      conversation_history: conversationHistory,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        // Parse event type for next data line
        continue;
      }
      if (line.startsWith("data:")) {
        const data = line.slice(5).trim();
        if (data) {
          try {
            const parsed = JSON.parse(data);
            console.log("SSE parsed data:", parsed);
            // Determine event type from data content
            if ("system_prompt" in parsed && "available_tools" in parsed) {
              // Agent context event
              yield { type: "agent_context", context: parsed };
            } else if ("content" in parsed && !("type" in parsed)) {
              yield { type: "text", content: parsed.content };
            } else if ("error" in parsed) {
              yield { type: "error", error: parsed.error };
            } else if (parsed.tool_calls !== undefined) {
              yield {
                type: "done",
                tool_calls: parsed.tool_calls,
                decisions_made: parsed.decisions_made || [],
              };
            } else if (
              "name" in parsed &&
              Object.prototype.hasOwnProperty.call(parsed, "output")
            ) {
              // Tool result - check for "output" key existence (can be null)
              console.log("Yielding tool_result:", parsed.name, parsed.output);
              yield {
                type: "tool_result",
                name: parsed.name,
                output: parsed.output,
              };
            } else if ("name" in parsed && "input" in parsed) {
              yield {
                type: "tool_use",
                name: parsed.name,
                input: parsed.input,
              };
            } else {
              console.log("Unhandled SSE data:", parsed);
            }
          } catch (e) {
            console.error("Failed to parse SSE data:", data, e);
          }
        }
      }
    }
  }
}

// Customers
export async function searchCustomers(
  query: string,
  limit = 10,
): Promise<Customer[]> {
  const response = await api.get("/api/customers/search", {
    params: { query, limit },
  });
  return response.data;
}

export async function getCustomer(customerId: string): Promise<Customer> {
  const response = await api.get(`/api/customers/${customerId}`);
  return response.data;
}

export async function getCustomerDecisions(
  customerId: string,
): Promise<Decision[]> {
  const response = await api.get(`/api/customers/${customerId}/decisions`);
  return response.data;
}

// Decisions
export async function getDecision(decisionId: string): Promise<Decision> {
  const response = await api.get(`/api/decisions/${decisionId}`);
  return response.data;
}

export async function listDecisions(
  category?: string,
  decisionType?: string,
  limit = 20,
): Promise<Decision[]> {
  const response = await api.get("/api/decisions", {
    params: {
      category,
      decision_type: decisionType,
      limit,
    },
  });
  return response.data.decisions;
}

export async function getSimilarDecisions(
  decisionId: string,
  limit = 5,
  similarityType: "structural" | "semantic" | "hybrid" = "hybrid",
): Promise<SimilarDecision[]> {
  const response = await api.get(`/api/decisions/${decisionId}/similar`, {
    params: { limit, similarity_type: similarityType },
  });
  return response.data;
}

export async function getCausalChain(
  decisionId: string,
  depth = 3,
): Promise<CausalChain> {
  const response = await api.get(`/api/decisions/${decisionId}/causal-chain`, {
    params: { depth },
  });
  return response.data;
}

export async function recordDecision(
  decision: Partial<Decision>,
): Promise<Decision> {
  const response = await api.post("/api/decisions", decision);
  return response.data;
}

// Policies
export async function getPolicies(): Promise<
  Array<{ id: string; name: string; rules: string[] }>
> {
  const response = await api.get("/api/policies");
  return response.data;
}

export async function getPolicy(
  policyId: string,
): Promise<{ id: string; name: string; rules: string[] }> {
  const response = await api.get(`/api/policies/${policyId}`);
  return response.data;
}

// Graph Visualization
export async function getGraphData(
  centerNodeId?: string,
  depth = 2,
  nodeTypes?: string[],
): Promise<GraphData> {
  const response = await api.get("/api/graph", {
    params: {
      center_node_id: centerNodeId,
      depth,
      node_types: nodeTypes?.join(","),
    },
  });
  return response.data;
}

export interface RelationshipPattern {
  from_label: string;
  rel_type: string;
  to_label: string;
  count: number;
}

export interface GraphSchema {
  node_labels: string[];
  node_counts: Record<string, number>;
  relationship_types: string[];
  relationship_counts: Record<string, number>;
  relationship_patterns: RelationshipPattern[];
  property_keys: string[];
  indexes: Array<{
    name: string;
    type: string;
    labels_or_types: string[];
    properties: string[];
    state: string;
  }>;
  constraints: Array<{
    name: string;
    type: string;
    labels_or_types: string[];
    properties: string[];
  }>;
}

export async function getGraphSchema(): Promise<GraphSchema> {
  const response = await api.get("/api/graph/schema");
  return response.data;
}

// Convert schema to GraphData for visualization
export function schemaToGraphData(schema: GraphSchema): GraphData {
  const nodes: GraphNode[] = [];
  const relationships: GraphRelationship[] = [];

  // Create nodes for each label
  schema.node_labels.forEach((label) => {
    nodes.push({
      id: `label_${label}`,
      labels: [label],
      properties: {
        name: label,
        count: schema.node_counts[label] || 0,
        isSchemaNode: true,
      },
    });
  });

  // Create relationships from patterns
  schema.relationship_patterns.forEach((pattern, index) => {
    relationships.push({
      id: `rel_${index}_${pattern.from_label}_${pattern.rel_type}_${pattern.to_label}`,
      type: pattern.rel_type,
      startNodeId: `label_${pattern.from_label}`,
      endNodeId: `label_${pattern.to_label}`,
      properties: {
        count: pattern.count,
        isSchemaRelationship: true,
      },
    });
  });

  return { nodes, relationships };
}

export async function getDecisionGraph(
  decisionId: string,
  depth = 2,
): Promise<GraphData> {
  const response = await api.get(`/api/graph/decision/${decisionId}`, {
    params: { depth },
  });
  return response.data;
}

// Expand a node to get all connected nodes
export async function expandNode(
  nodeId: string,
  limit = 50,
): Promise<GraphData> {
  const response = await api.get(
    `/api/graph/expand/${encodeURIComponent(nodeId)}`,
    {
      params: { limit },
    },
  );
  return response.data;
}

// Get relationships between a set of nodes
export async function getRelationshipsBetween(
  nodeIds: string[],
): Promise<GraphRelationship[]> {
  const response = await api.post("/api/graph/relationships", nodeIds);
  return response.data.relationships;
}

// GDS Analytics
export async function runFastRP(): Promise<{ nodes_updated: number }> {
  const response = await api.post("/api/gds/fastrp");
  return response.data;
}

export async function detectCommunities(): Promise<
  Array<{ community_id: number; members: string[] }>
> {
  const response = await api.get("/api/gds/communities");
  return response.data;
}

export async function getInfluenceScores(
  limit = 20,
): Promise<Array<{ id: string; score: number }>> {
  const response = await api.get("/api/gds/influence", {
    params: { limit },
  });
  return response.data;
}

export async function detectFraudPatterns(accountId: string): Promise<{
  risk_score: number;
  patterns: string[];
  similar_accounts: string[];
}> {
  const response = await api.get(`/api/gds/fraud-patterns/${accountId}`);
  return response.data;
}

// Vector Search
export async function semanticSearch(
  query: string,
  nodeType = "Decision",
  limit = 10,
): Promise<Array<{ id: string; score: number; content: string }>> {
  const response = await api.post("/api/vector/search", {
    query,
    node_type: nodeType,
    limit,
  });
  return response.data;
}

export async function hybridSearch(
  query: string,
  nodeType = "Decision",
  limit = 10,
): Promise<
  Array<{
    id: string;
    semantic_score: number;
    structural_score: number;
    combined_score: number;
  }>
> {
  const response = await api.post("/api/vector/hybrid-search", {
    query,
    node_type: nodeType,
    limit,
  });
  return response.data;
}

export default api;
