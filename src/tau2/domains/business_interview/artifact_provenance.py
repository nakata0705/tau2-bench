"""Primary-input provenance for business_interview evaluation artifacts.

The benchmark's reference score is derived data.  The durable primary inputs
are the complete canonical TruthGraph, the concrete StakeholderKnowledge
objects used by the stakeholder simulator, and seed provenance describing how
those objects were obtained.  This module keeps those inputs together in a
human-readable JSON envelope and provides deterministic fingerprints and a
small offline reference re-evaluation helper.

The evaluator-private mappings in ``StakeholderKnowledge`` are intentionally
serialized here.  They are never rendered by the stakeholder prompt or
exposed through business-interview tools; they exist in offline artifacts so a
future evaluator can reconstruct the same Truth correspondence without
trusting a historical score or regenerating knowledge from a seed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from tau2.domains.business_interview.graph import (
    BusinessProcessGraph,
    canonical_structure_errors,
    validate_canonical_graph,
)
from tau2.domains.business_interview.knowledge import StakeholderKnowledge

ARTIFACT_SCHEMA_VERSION = "business_interview.evaluation_inputs.v1"
FINGERPRINT_ALGORITHM = "sha256"


class SeedProvenance(BaseModel):
    """Explicit meaning of every seed relevant to one artifact.

    ``simulation_seed`` is the run seed passed to the orchestrator.  The
    stakeholder projection may have been materialized before the run (as in
    the current zero-forgetting scenarios), in which case the generation and
    forgetting seeds are ``None`` and that fact is recorded rather than
    implying that the simulation seed controlled the projection.
    """

    simulation_seed: Optional[int] = None
    stakeholder_generation_seed: Optional[int] = None
    forgetting_seed: Optional[int] = None
    stakeholder_generation_rng: str = "not_recorded"
    forgetting_rng: str = "not_recorded"
    simulation_seed_drives_stakeholder_knowledge: bool = False
    forgetting_seed_is_simulation_seed: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _set_seed_relation(self) -> "SeedProvenance":
        self.forgetting_seed_is_simulation_seed = bool(
            self.simulation_seed is not None
            and self.forgetting_seed is not None
            and self.simulation_seed == self.forgetting_seed
        )
        return self


class StakeholderEvaluationInput(BaseModel):
    """One stable stakeholder identity and the exact knowledge object used."""

    stakeholder_id: str
    stakeholder_name: str = ""
    stakeholder_role: Optional[str] = None
    forgetting_configuration: dict[str, Any] = Field(default_factory=dict)
    seed_provenance: SeedProvenance = Field(default_factory=SeedProvenance)
    knowledge: StakeholderKnowledge
    stakeholder_knowledge_fingerprint: str = ""

    @model_validator(mode="after")
    def _complete_fingerprint(self) -> "StakeholderEvaluationInput":
        if not self.stakeholder_id.strip():
            raise ValueError("stakeholder_id must be non-empty")
        if not self.stakeholder_knowledge_fingerprint:
            self.stakeholder_knowledge_fingerprint = fingerprint(
                serialize_stakeholder_knowledge(self.knowledge)
            )
        return self


class BusinessInterviewEvaluationInputs(BaseModel):
    """The complete primary-input envelope stored in an offline artifact."""

    schema_version: str = ARTIFACT_SCHEMA_VERSION
    seed: Optional[int] = Field(
        default=None,
        description="Simulation seed, if the run was seeded.",
    )
    seed_provenance: SeedProvenance = Field(default_factory=SeedProvenance)
    truth_graph: BusinessProcessGraph
    truth_graph_fingerprint: str = ""
    stakeholders: list[StakeholderEvaluationInput] = Field(default_factory=list)
    fingerprint_policy: dict[str, str] = Field(
        default_factory=lambda: {
            "algorithm": FINGERPRINT_ALGORITHM,
            "truth_graph": (
                "full canonical TruthGraph; dict keys and unordered semantic "
                "collections are canonicalized"
            ),
            "stakeholder_knowledge": (
                "full StakeholderKnowledge including opaque local ids and "
                "evaluator-private Truth mappings/shortcut provenance"
            ),
        }
    )

    @model_validator(mode="after")
    def _validate_inputs(self) -> "BusinessInterviewEvaluationInputs":
        errors = canonical_structure_errors(self.truth_graph)
        if errors:
            raise ValueError(
                "evaluation_inputs.truth_graph is not canonical:\n- "
                + "\n- ".join(errors)
            )
        ids = [item.stakeholder_id for item in self.stakeholders]
        if len(ids) != len(set(ids)):
            raise ValueError("stakeholder_id values must be unique")
        if not self.truth_graph_fingerprint:
            self.truth_graph_fingerprint = fingerprint(
                serialize_truth_graph(self.truth_graph)
            )
        if self.seed_provenance.simulation_seed is None:
            self.seed_provenance.simulation_seed = self.seed
        elif self.seed is None:
            self.seed = self.seed_provenance.simulation_seed
        same_seed = bool(
            self.seed_provenance.simulation_seed is not None
            and self.seed_provenance.forgetting_seed is not None
            and self.seed_provenance.simulation_seed
            == self.seed_provenance.forgetting_seed
        )
        self.seed_provenance.forgetting_seed_is_simulation_seed = same_seed
        self.seed_provenance.simulation_seed_drives_stakeholder_knowledge = same_seed
        return self

    def stakeholder_by_id(self) -> dict[str, StakeholderEvaluationInput]:
        """Return identifier-indexed inputs, independent of list ordering."""
        return {item.stakeholder_id: item for item in self.stakeholders}


def serialize_truth_graph(graph: BusinessProcessGraph) -> dict[str, Any]:
    """Serialize the complete canonical TruthGraph, including boundaries."""
    validate_canonical_graph(graph)
    return graph.model_dump(mode="json")


def deserialize_truth_graph(value: Mapping[str, Any]) -> BusinessProcessGraph:
    """Restore and validate a complete canonical TruthGraph from JSON data."""
    graph = BusinessProcessGraph.model_validate(value)
    validate_canonical_graph(graph)
    return graph


def serialize_stakeholder_knowledge(
    knowledge: StakeholderKnowledge,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Serialize the exact post-forgetting StakeholderKnowledge object.

    The evaluator-private ``node_truth_ids``, ``edge_truth_ids`` and
    ``shortcut_provenance`` fields are deliberately included.  ``validate``
    can be disabled only for reading historical pre-canonical artifacts; new
    artifacts should always use the default.
    """
    if validate:
        errors = knowledge.graph.structure_errors()
        if errors:
            raise ValueError(
                "stakeholder knowledge is not canonical:\n- " + "\n- ".join(errors)
            )
    return knowledge.model_dump(mode="json")


def deserialize_stakeholder_knowledge(
    value: Mapping[str, Any],
    *,
    validate: bool = True,
) -> StakeholderKnowledge:
    """Restore StakeholderKnowledge, preserving epistemic markers/provenance."""
    knowledge = StakeholderKnowledge.model_validate(value)
    if validate:
        errors = knowledge.graph.structure_errors()
        if errors:
            raise ValueError(
                "stakeholder knowledge is not canonical:\n- " + "\n- ".join(errors)
            )
    return knowledge


_UNORDERED_LIST_KEYS = frozenset(
    {
        "canonical_terms",
        "end_node_ids",
        "start_node_ids",
        "reads",
        "writes",
        "mentions",
        "evidence",
    }
)


def _canonical_value(value: Any, *, key: Optional[str] = None) -> Any:
    """Normalize JSON values for order-independent diagnostic hashing."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {
            str(name): _canonical_value(child, key=str(name))
            for name, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        normalized = [_canonical_value(child, key=key) for child in value]
        if key in _UNORDERED_LIST_KEYS:
            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        return normalized
    if isinstance(value, tuple):
        return _canonical_value(list(value), key=key)
    return value


def canonical_json(value: Any) -> str:
    """Return stable human-independent JSON for fingerprint generation."""
    normalized = _canonical_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    """Hash canonical JSON, independent of mapping/list insertion order."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def truth_graph_fingerprint(graph: BusinessProcessGraph) -> str:
    """Return the canonical fingerprint of a complete TruthGraph."""
    return fingerprint(serialize_truth_graph(graph))


def stakeholder_knowledge_fingerprint(knowledge: StakeholderKnowledge) -> str:
    """Return the full diagnostic fingerprint of StakeholderKnowledge."""
    return fingerprint(serialize_stakeholder_knowledge(knowledge))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _profile_values(item: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Extract profile metadata without importing scenario/evaluation modules."""
    if isinstance(item, Mapping):
        knowledge = item.get("knowledge")
        filter_ = item.get("stakeholder")
        stakeholder_id = item.get("stakeholder_id")
        name = item.get("stakeholder_name") or item.get("name")
        role = item.get("stakeholder_role") or item.get("role")
        forgetting = item.get("forgetting_configuration")
        seeds = item.get("seed_provenance")
        return knowledge, filter_, stakeholder_id, name, role, (forgetting, seeds)

    knowledge = getattr(item, "knowledge", None)
    filter_ = getattr(item, "stakeholder", None)
    stakeholder_id = getattr(item, "stakeholder_id", None)
    name = getattr(item, "stakeholder_name", None) or getattr(item, "name", None)
    role = getattr(item, "stakeholder_role", None) or getattr(item, "role", None)
    forgetting = getattr(item, "forgetting_configuration", None)
    seeds = getattr(item, "seed_provenance", None)
    return knowledge, filter_, stakeholder_id, name, role, (forgetting, seeds)


def _coerce_profile(
    item: Any,
    *,
    simulation_seed: Optional[int],
) -> StakeholderEvaluationInput:
    knowledge, filter_, stakeholder_id, name, role, extras = _profile_values(item)
    forgetting, explicit_seeds = extras
    if not isinstance(knowledge, StakeholderKnowledge):
        knowledge = deserialize_stakeholder_knowledge(knowledge)

    filter_name = getattr(filter_, "name", None) or ""
    name = str(name or filter_name or knowledge.graph.name or "")
    stakeholder_id = (
        stakeholder_id
        or getattr(filter_, "stakeholder_id", None)
        or _slug(name)
        or _slug(knowledge.graph.id)
        or "stakeholder"
    )
    stakeholder_id = str(stakeholder_id)
    role = role or getattr(filter_, "role", None)

    if forgetting is None and filter_ is not None:
        forgetting_model = getattr(filter_, "forgetting", None)
        forgetting = (
            forgetting_model.model_dump(mode="json")
            if isinstance(forgetting_model, BaseModel)
            else forgetting_model
        )
    if not isinstance(forgetting, Mapping):
        forgetting = {}

    if explicit_seeds is None:
        explicit_seeds = {}
    if isinstance(explicit_seeds, SeedProvenance):
        seed_provenance = explicit_seeds.model_copy(deep=True)
    else:
        seed_provenance = SeedProvenance.model_validate(explicit_seeds or {})
    generation_seed = getattr(knowledge, "generation_seed", None)
    rng_source = getattr(knowledge, "generation_rng_source", "not_recorded")
    if seed_provenance.stakeholder_generation_seed is None:
        seed_provenance.stakeholder_generation_seed = generation_seed
    if seed_provenance.forgetting_seed is None:
        seed_provenance.forgetting_seed = generation_seed
    if seed_provenance.stakeholder_generation_rng == "not_recorded":
        seed_provenance.stakeholder_generation_rng = rng_source
    if seed_provenance.forgetting_rng == "not_recorded":
        seed_provenance.forgetting_rng = rng_source
    seed_provenance.simulation_seed = simulation_seed
    if (
        simulation_seed is not None
        and seed_provenance.forgetting_seed is not None
        and simulation_seed == seed_provenance.forgetting_seed
    ):
        seed_provenance.simulation_seed_drives_stakeholder_knowledge = True
    seed_provenance.forgetting_seed_is_simulation_seed = bool(
        simulation_seed is not None
        and seed_provenance.forgetting_seed is not None
        and simulation_seed == seed_provenance.forgetting_seed
    )

    return StakeholderEvaluationInput(
        stakeholder_id=stakeholder_id,
        stakeholder_name=name,
        stakeholder_role=str(role) if role is not None else None,
        forgetting_configuration=dict(forgetting),
        seed_provenance=seed_provenance,
        knowledge=knowledge,
    )


def build_evaluation_inputs(
    truth_graph: BusinessProcessGraph,
    stakeholders: list[Any] | tuple[Any, ...],
    *,
    simulation_seed: Optional[int] = None,
    seed: Optional[int] = None,
    seed_provenance: Optional[SeedProvenance | Mapping[str, Any]] = None,
) -> BusinessInterviewEvaluationInputs:
    """Build an artifact envelope from concrete run objects.

    ``stakeholders`` may contain ``ScenarioStakeholder`` objects,
    ``StakeholderReferenceInput``-shaped mappings, or objects with
    ``stakeholder_id``/``knowledge`` attributes.  The Knowledge instances are
    serialized directly; no projection or inference is performed here.
    """
    if simulation_seed is not None and seed is not None and simulation_seed != seed:
        raise ValueError("simulation_seed and seed must match when both are provided")
    if simulation_seed is None:
        simulation_seed = seed
    if not isinstance(truth_graph, BusinessProcessGraph):
        truth_graph = deserialize_truth_graph(truth_graph)
    serialized_truth = serialize_truth_graph(truth_graph)
    profiles = [
        _coerce_profile(item, simulation_seed=simulation_seed) for item in stakeholders
    ]
    top = (
        seed_provenance.model_copy(deep=True)
        if isinstance(seed_provenance, SeedProvenance)
        else SeedProvenance.model_validate(seed_provenance or {})
    )
    top.simulation_seed = simulation_seed
    if len(profiles) == 1:
        profile_seed = profiles[0].seed_provenance
        if top.stakeholder_generation_seed is None:
            top.stakeholder_generation_seed = profile_seed.stakeholder_generation_seed
        if top.forgetting_seed is None:
            top.forgetting_seed = profile_seed.forgetting_seed
        if top.stakeholder_generation_rng == "not_recorded":
            top.stakeholder_generation_rng = profile_seed.stakeholder_generation_rng
        if top.forgetting_rng == "not_recorded":
            top.forgetting_rng = profile_seed.forgetting_rng
    top.simulation_seed_drives_stakeholder_knowledge = bool(
        top.simulation_seed is not None
        and top.forgetting_seed is not None
        and top.simulation_seed == top.forgetting_seed
    )
    top.forgetting_seed_is_simulation_seed = (
        top.simulation_seed_drives_stakeholder_knowledge
    )
    if top.forgetting_seed is None:
        top.notes.append(
            "StakeholderKnowledge was materialized independently of the "
            "simulation seed; the concrete Knowledge object is the authority."
        )
    return BusinessInterviewEvaluationInputs(
        seed=simulation_seed,
        seed_provenance=top,
        truth_graph=deserialize_truth_graph(serialized_truth),
        truth_graph_fingerprint=fingerprint(serialized_truth),
        stakeholders=profiles,
    )


def serialize_evaluation_inputs(
    inputs: BusinessInterviewEvaluationInputs,
) -> dict[str, Any]:
    """Serialize an input envelope as JSON-ready human-readable data."""
    return inputs.model_dump(mode="json")


def deserialize_evaluation_inputs(
    value: Mapping[str, Any],
) -> BusinessInterviewEvaluationInputs:
    """Restore an input envelope and validate its canonical contract."""
    return BusinessInterviewEvaluationInputs.model_validate(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read artifact JSON {path}: {exc}") from exc


def load_evaluation_inputs(
    artifact: BusinessInterviewEvaluationInputs | Mapping[str, Any] | str | Path,
    private_artifact: str | Path | None = None,
) -> BusinessInterviewEvaluationInputs:
    """Load inputs from a diagnostic artifact or a public/private pair.

    New artifacts contain ``evaluation_inputs`` directly.  For a simulation
    public artifact, ``private_artifact`` may point to the evaluator-private
    sidecar.  Historical artifacts without the new envelope fail closed rather
    than pretending a stored score is sufficient provenance.
    """
    if isinstance(artifact, BusinessInterviewEvaluationInputs):
        return artifact
    if isinstance(artifact, (str, Path)):
        public = _read_json(Path(artifact))
        if private_artifact is not None:
            private = _read_json(Path(private_artifact))
            if private.get("evaluation_inputs") is not None:
                return deserialize_evaluation_inputs(private["evaluation_inputs"])
        if public.get("evaluation_inputs") is not None:
            return deserialize_evaluation_inputs(public["evaluation_inputs"])
        info = public.get("info") or {}
        if info.get("evaluation_inputs") is not None:
            return deserialize_evaluation_inputs(info["evaluation_inputs"])
        if info.get("business_interview_evaluation_inputs") is not None:
            return deserialize_evaluation_inputs(
                info["business_interview_evaluation_inputs"]
            )
        raise ValueError(
            f"{artifact} does not contain evaluation_inputs; use a new artifact "
            "or provide its evaluator-private sidecar"
        )

    if artifact.get("evaluation_inputs") is not None:
        return deserialize_evaluation_inputs(artifact["evaluation_inputs"])
    info = artifact.get("info") or {}
    if info.get("evaluation_inputs") is not None:
        return deserialize_evaluation_inputs(info["evaluation_inputs"])
    if info.get("business_interview_evaluation_inputs") is not None:
        return deserialize_evaluation_inputs(
            info["business_interview_evaluation_inputs"]
        )
    if (
        artifact.get("truth_graph") is not None
        and artifact.get("stakeholders") is not None
    ):
        return deserialize_evaluation_inputs(artifact)
    raise ValueError("artifact does not contain evaluation_inputs")


def recompute_stakeholder_truth_reference(
    artifact: BusinessInterviewEvaluationInputs | Mapping[str, Any] | str | Path,
    private_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Re-run reference evaluation from saved Truth + actual Knowledge.

    The returned score is never read from ``evaluator_metrics`` or a prior
    diagnostic trace.  The evaluator is imported lazily to keep this module's
    serialization helpers independent of the large primary evaluator module.
    """
    inputs = load_evaluation_inputs(artifact, private_artifact)
    from tau2.domains.business_interview.reference_evaluation import (
        StakeholderReferenceInput,
        aggregate_stakeholder_truth_references,
        evaluate_stakeholder_truth_reference,
    )

    evaluations = []
    for profile in sorted(inputs.stakeholders, key=lambda item: item.stakeholder_id):
        reference = StakeholderReferenceInput(
            stakeholder_id=profile.stakeholder_id,
            stakeholder_name=profile.stakeholder_name,
            stakeholder_role=profile.stakeholder_role,
            forgetting_configuration=dict(profile.forgetting_configuration),
            knowledge=profile.knowledge,
        )
        evaluations.append(
            evaluate_stakeholder_truth_reference(inputs.truth_graph, reference)
        )
    aggregate = aggregate_stakeholder_truth_references(evaluations)
    return {
        "schema_version": "business_interview.reference_revaluation.v1",
        "seed": inputs.seed,
        "truth_graph_fingerprint": inputs.truth_graph_fingerprint,
        "stakeholders": [
            item.stakeholder_id
            for item in sorted(
                inputs.stakeholders, key=lambda item: item.stakeholder_id
            )
        ],
        "stakeholder_truth_reference": [
            item.model_dump(mode="json") for item in evaluations
        ],
        "stakeholder_truth_reference_aggregate": aggregate.model_dump(mode="json"),
    }


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BusinessInterviewEvaluationInputs",
    "FINGERPRINT_ALGORITHM",
    "SeedProvenance",
    "StakeholderEvaluationInput",
    "build_evaluation_inputs",
    "canonical_json",
    "deserialize_evaluation_inputs",
    "deserialize_stakeholder_knowledge",
    "deserialize_truth_graph",
    "fingerprint",
    "load_evaluation_inputs",
    "recompute_stakeholder_truth_reference",
    "serialize_evaluation_inputs",
    "serialize_stakeholder_knowledge",
    "serialize_truth_graph",
    "stakeholder_knowledge_fingerprint",
    "truth_graph_fingerprint",
]
