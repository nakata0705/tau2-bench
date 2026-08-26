"""Explicit offline execution of business-interview research diagnostics.

Production evaluation deliberately excludes usage and joint structural alignment.
This module is the one-way adapter used by offline tooling:

    production result -> usage experiment -> joint experiment

Nothing under ``tau2.domains.business_interview`` imports this module or the
experiment implementations.
"""

from __future__ import annotations

from pydantic import BaseModel

from experiments.business_interview import (
    joint_structural_alignment,
    usage_alignment,
)
from experiments.business_interview.joint_structural_alignment import (
    JointStructuralAlignmentDiagnostics,
)
from experiments.business_interview.usage_alignment import (
    UsageAlignmentDiagnostics,
)
from tau2.domains.business_interview.evaluation_diagnostics import (
    EvaluationDiagnostics,
)
from tau2.domains.business_interview.graph import (
    AgentGraph,
    BusinessProcessGraph,
    business_graph_projection,
)


class OfflineExperimentDiagnostics(BaseModel):
    """Versioned results produced only by the offline experiment lane."""

    schema_version: str = "business_interview.offline_experiments.v1"
    usage_alignment: UsageAlignmentDiagnostics
    joint_structural_alignment: JointStructuralAlignmentDiagnostics


def build_offline_experiment_diagnostics(
    agent: AgentGraph,
    truth: BusinessProcessGraph,
    diagnostics: EvaluationDiagnostics,
) -> OfflineExperimentDiagnostics:
    """Run usage and joint experiments from an ordinary production result.

    The production diagnostics contain the selected Agent-to-Truth mappings.
    They are comparison inputs for these experiments, never constraints or
    outputs fed back into production scoring.
    """

    target = business_graph_projection(truth)
    node_mapping = {
        item.agent_node_id: item.truth_node_id
        for item in diagnostics.node_diagnostics
        if item.agent_node_id is not None
    }
    edge_mapping = {
        item.agent_edge_id: item.truth_edge_id
        for item in diagnostics.edge_diagnostics
        if item.agent_edge_id is not None
    }
    concept_mapping = dict(diagnostics.concepts.agent_to_truth)

    usage = usage_alignment.build_usage_alignment_diagnostics(
        agent,
        target,
        node_mapping=node_mapping,
        edge_mapping=edge_mapping,
        current_agent_to_truth=concept_mapping,
        lexical_pairs=diagnostics.concepts.candidate_pairs,
    )
    joint = joint_structural_alignment.build_joint_structural_alignment_diagnostics(
        agent,
        target,
        production_node_to_truth=node_mapping,
        production_concept_to_truth=concept_mapping,
        usage_concept_to_truth=usage.usage_agent_to_truth,
    )
    return OfflineExperimentDiagnostics(
        usage_alignment=usage,
        joint_structural_alignment=joint,
    )
