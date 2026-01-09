"use client";

import { useState, useEffect } from "react";
import {
  Box,
  Text,
  Flex,
  Badge,
  VStack,
  HStack,
  Heading,
  Separator,
  Spinner,
  Button,
} from "@chakra-ui/react";
import {
  getSimilarDecisions,
  getCausalChain,
  listDecisions,
  type Decision,
  type SimilarDecision,
  type CausalChain,
} from "@/lib/api";

interface DecisionTracePanelProps {
  decision: Decision | null;
  onDecisionSelect: (decision: Decision) => void;
  graphDecisions?: Decision[]; // Decisions from the graph visualization
}

const DECISION_TYPE_COLORS: Record<string, string> = {
  approval: "green",
  rejection: "red",
  escalation: "purple",
  exception: "yellow",
  override: "orange",
  credit_approval: "green",
  credit_denial: "red",
  fraud_alert: "red",
  fraud_cleared: "green",
  trading_approval: "blue",
  trading_halt: "orange",
  exception_granted: "yellow",
  exception_denied: "red",
};

const CATEGORY_COLORS: Record<string, string> = {
  fraud: "red",
  credit: "blue",
  compliance: "purple",
  trading: "cyan",
  account_management: "green",
  support: "orange",
};

export function DecisionTracePanel({
  decision,
  onDecisionSelect,
  graphDecisions = [],
}: DecisionTracePanelProps) {
  const [similarDecisions, setSimilarDecisions] = useState<SimilarDecision[]>(
    [],
  );
  const [causalChain, setCausalChain] = useState<CausalChain | null>(null);
  const [recentDecisions, setRecentDecisions] = useState<Decision[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingRecent, setLoadingRecent] = useState(true);

  // Load recent decisions when no decision is selected and no graph decisions
  useEffect(() => {
    if (!decision && graphDecisions.length === 0) {
      const fetchRecentDecisions = async () => {
        setLoadingRecent(true);
        try {
          const decisions = await listDecisions(undefined, undefined, 15);
          setRecentDecisions(decisions);
        } catch (error) {
          console.error("Failed to fetch recent decisions:", error);
        } finally {
          setLoadingRecent(false);
        }
      };
      fetchRecentDecisions();
    } else {
      setLoadingRecent(false);
    }
  }, [decision, graphDecisions.length]);

  useEffect(() => {
    if (!decision) {
      setSimilarDecisions([]);
      setCausalChain(null);
      return;
    }

    const fetchData = async () => {
      setLoading(true);
      try {
        const [similar, chain] = await Promise.all([
          getSimilarDecisions(decision.id, 5, "hybrid"),
          getCausalChain(decision.id, 2),
        ]);
        setSimilarDecisions(similar);
        setCausalChain(chain);
      } catch (error) {
        console.error("Failed to fetch decision data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [decision]);

  // Determine which decisions to show in the list
  const decisionsToShow =
    graphDecisions.length > 0 ? graphDecisions : recentDecisions;
  const listTitle =
    graphDecisions.length > 0 ? "Decisions in Graph" : "Recent Decisions";
  const listDescription =
    graphDecisions.length > 0
      ? "Decisions visible in the Context Graph. Click to view details, or double-click nodes in the graph to expand."
      : "Click a decision to view its full trace, causal chain, and similar decisions.";

  // Show decisions list when no decision is selected
  if (!decision) {
    return (
      <Box p={4}>
        <VStack gap={4} align="stretch">
          <Heading size="sm">{listTitle}</Heading>
          <Text fontSize="sm" color="gray.500">
            {listDescription}
          </Text>

          {graphDecisions.length > 0 && (
            <Badge colorPalette="blue" size="sm" alignSelf="flex-start">
              {graphDecisions.length} decision
              {graphDecisions.length !== 1 ? "s" : ""} in graph
            </Badge>
          )}

          {loadingRecent ? (
            <Flex justify="center" py={8}>
              <Spinner size="md" />
            </Flex>
          ) : decisionsToShow.length > 0 ? (
            <VStack gap={2} align="stretch">
              {decisionsToShow.map((d) => (
                <RecentDecisionCard
                  key={d.id}
                  decision={d}
                  onClick={() => onDecisionSelect(d)}
                />
              ))}
            </VStack>
          ) : (
            <Text color="gray.500" textAlign="center" py={4}>
              No decisions found.
            </Text>
          )}
        </VStack>
      </Box>
    );
  }

  const typeColor = DECISION_TYPE_COLORS[decision.decision_type] || "gray";
  const categoryColor = CATEGORY_COLORS[decision.category] || "gray";

  return (
    <Box p={4}>
      <VStack gap={4} align="stretch">
        {/* Back button */}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onDecisionSelect(null as unknown as Decision)}
        >
          Back to list
        </Button>

        {/* Decision Header */}
        <Box>
          <HStack gap={2} mb={2} flexWrap="wrap">
            <Badge colorPalette={typeColor} size="lg">
              {decision.decision_type.replace(/_/g, " ")}
            </Badge>
            <Badge colorPalette={categoryColor} variant="outline">
              {decision.category}
            </Badge>
            <Badge
              colorPalette={
                decision.status === "approved"
                  ? "green"
                  : decision.status === "rejected"
                    ? "red"
                    : "yellow"
              }
            >
              {decision.status}
            </Badge>
          </HStack>
          <Text fontSize="sm" color="gray.500">
            {decision.timestamp
              ? new Date(decision.timestamp).toLocaleString()
              : "Unknown date"}
          </Text>
          <Text fontSize="xs" color="gray.400" mt={1}>
            ID: {decision.id.slice(0, 8)}...
          </Text>
        </Box>

        <Separator />

        {/* Reasoning */}
        <Box>
          <Heading size="sm" mb={2}>
            Reasoning
          </Heading>
          <Box
            bg="bg.subtle"
            p={3}
            borderRadius="md"
            fontSize="sm"
            whiteSpace="pre-wrap"
          >
            {decision.reasoning || "No reasoning provided."}
          </Box>
        </Box>

        {/* Confidence */}
        <HStack gap={4}>
          <Box>
            <Text fontSize="xs" color="gray.500" mb={1}>
              Confidence Score
            </Text>
            <Text fontWeight="medium">
              {(decision.confidence ?? decision.confidence_score)
                ? `${((decision.confidence ?? decision.confidence_score ?? 0) * 100).toFixed(0)}%`
                : "N/A"}
            </Text>
          </Box>
        </HStack>

        {/* Risk Factors */}
        {Array.isArray(decision.risk_factors) &&
          decision.risk_factors.length > 0 && (
            <Box>
              <Heading size="sm" mb={2}>
                Risk Factors
              </Heading>
              <Flex gap={2} flexWrap="wrap">
                {decision.risk_factors.map((factor, idx) => (
                  <Badge key={idx} colorPalette="orange" variant="subtle">
                    {String(factor).replace(/_/g, " ")}
                  </Badge>
                ))}
              </Flex>
            </Box>
          )}

        <Separator />

        {/* Causal Chain */}
        <Box>
          <Heading size="sm" mb={2}>
            Causal Chain
          </Heading>
          {loading ? (
            <Flex justify="center" py={4}>
              <Spinner size="sm" />
            </Flex>
          ) : causalChain ? (
            <VStack gap={2} align="stretch">
              {/* Causes */}
              {causalChain.causes && causalChain.causes.length > 0 && (
                <Box>
                  <Text fontSize="xs" color="gray.500" mb={1}>
                    Caused by ({causalChain.causes.length})
                  </Text>
                  {causalChain.causes.map((cause) => (
                    <DecisionCard
                      key={cause.id}
                      decision={cause}
                      onClick={() => onDecisionSelect(cause)}
                      direction="cause"
                    />
                  ))}
                </Box>
              )}

              {/* Effects */}
              {causalChain.effects && causalChain.effects.length > 0 && (
                <Box>
                  <Text fontSize="xs" color="gray.500" mb={1}>
                    Led to ({causalChain.effects.length})
                  </Text>
                  {causalChain.effects.map((effect) => (
                    <DecisionCard
                      key={effect.id}
                      decision={effect}
                      onClick={() => onDecisionSelect(effect)}
                      direction="effect"
                    />
                  ))}
                </Box>
              )}

              {(!causalChain.causes || causalChain.causes.length === 0) &&
                (!causalChain.effects || causalChain.effects.length === 0) && (
                  <Text fontSize="sm" color="gray.500">
                    No causal relationships found.
                  </Text>
                )}
            </VStack>
          ) : (
            <Text fontSize="sm" color="gray.500">
              No causal chain data.
            </Text>
          )}
        </Box>

        <Separator />

        {/* Similar Decisions */}
        <Box>
          <Heading size="sm" mb={2}>
            Similar Decisions
          </Heading>
          {loading ? (
            <Flex justify="center" py={4}>
              <Spinner size="sm" />
            </Flex>
          ) : similarDecisions.length > 0 ? (
            <VStack gap={2} align="stretch">
              {similarDecisions.map((similar) => (
                <SimilarDecisionCard
                  key={similar.decision.id}
                  similarDecision={similar}
                  onClick={() => onDecisionSelect(similar.decision)}
                />
              ))}
            </VStack>
          ) : (
            <Text fontSize="sm" color="gray.500">
              No similar decisions found.
            </Text>
          )}
        </Box>
      </VStack>
    </Box>
  );
}

// Recent decision card for the list view
function RecentDecisionCard({
  decision,
  onClick,
}: {
  decision: Decision;
  onClick: () => void;
}) {
  const typeColor = DECISION_TYPE_COLORS[decision.decision_type] || "gray";
  const categoryColor = CATEGORY_COLORS[decision.category] || "gray";

  return (
    <Box
      bg="bg.subtle"
      p={3}
      borderRadius="md"
      cursor="pointer"
      _hover={{ bg: "bg.emphasized" }}
      onClick={onClick}
      borderLeftWidth="3px"
      borderLeftColor={`${typeColor}.500`}
    >
      <HStack justify="space-between" mb={1} flexWrap="wrap" gap={1}>
        <HStack gap={1}>
          <Badge size="sm" colorPalette={typeColor}>
            {decision.decision_type}
          </Badge>
          <Badge size="sm" colorPalette={categoryColor} variant="outline">
            {decision.category}
          </Badge>
        </HStack>
        {(decision.confidence ?? decision.confidence_score) && (
          <Text fontSize="xs" color="gray.500">
            {(
              (decision.confidence ?? decision.confidence_score ?? 0) * 100
            ).toFixed(0)}
            % conf
          </Text>
        )}
      </HStack>
      <Text fontSize="sm" color="gray.600" lineClamp={2}>
        {decision.reasoning?.slice(0, 120) || "No reasoning"}
        {decision.reasoning && decision.reasoning.length > 120 ? "..." : ""}
      </Text>
      <HStack justify="space-between" mt={2}>
        <Text fontSize="xs" color="gray.400">
          {decision.timestamp
            ? new Date(decision.timestamp).toLocaleDateString()
            : ""}
        </Text>
        {Array.isArray(decision.risk_factors) &&
          decision.risk_factors.length > 0 && (
            <Badge size="sm" colorPalette="orange" variant="subtle">
              {decision.risk_factors.length} risk factors
            </Badge>
          )}
      </HStack>
    </Box>
  );
}

// Decision card for causal chain
function DecisionCard({
  decision,
  onClick,
  direction,
}: {
  decision: Decision;
  onClick: () => void;
  direction: "cause" | "effect";
}) {
  const typeColor = DECISION_TYPE_COLORS[decision.decision_type] || "gray";
  const arrow = direction === "cause" ? "^" : "v";

  return (
    <Box
      bg="bg.subtle"
      p={2}
      borderRadius="md"
      cursor="pointer"
      _hover={{ bg: "bg.emphasized" }}
      onClick={onClick}
      mb={1}
    >
      <HStack gap={2}>
        <Text color={direction === "cause" ? "blue.500" : "green.500"}>
          {arrow}
        </Text>
        <Badge size="sm" colorPalette={typeColor}>
          {decision.decision_type.replace(/_/g, " ")}
        </Badge>
        <Text fontSize="xs" color="gray.500" flex={1} truncate>
          {decision.category}
        </Text>
      </HStack>
    </Box>
  );
}

// Similar decision card with similarity score
function SimilarDecisionCard({
  similarDecision,
  onClick,
}: {
  similarDecision: SimilarDecision;
  onClick: () => void;
}) {
  const { decision, similarity_score, similarity_type } = similarDecision;
  const typeColor = DECISION_TYPE_COLORS[decision.decision_type] || "gray";

  return (
    <Box
      bg="bg.subtle"
      p={3}
      borderRadius="md"
      cursor="pointer"
      _hover={{ bg: "bg.emphasized" }}
      onClick={onClick}
    >
      <HStack justify="space-between" mb={1}>
        <Badge size="sm" colorPalette={typeColor}>
          {decision.decision_type.replace(/_/g, " ")}
        </Badge>
        <HStack gap={1}>
          <Badge size="sm" variant="outline">
            {similarity_type}
          </Badge>
          <Text fontSize="xs" fontWeight="bold" color="brand.500">
            {(similarity_score * 100).toFixed(0)}%
          </Text>
        </HStack>
      </HStack>
      <Text fontSize="sm" color="gray.600" lineClamp={2}>
        {decision.reasoning?.slice(0, 150) || "No reasoning"}...
      </Text>
      <Text fontSize="xs" color="gray.400" mt={1}>
        {(decision.timestamp ?? decision.decision_timestamp)
          ? new Date(
              decision.timestamp ?? decision.decision_timestamp ?? "",
            ).toLocaleDateString()
          : ""}
      </Text>
    </Box>
  );
}
