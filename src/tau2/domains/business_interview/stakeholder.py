"""Stakeholder as a filter over the single Truth process graph.

There is exactly one Truth ``BusinessProcessGraph``; a stakeholder is a
``StakeholderFilter`` describing which parts of it they know. The filter
drives ``project_knowledge`` (knowledge.py):

- ``visible_node_ids`` / ``visible_edge_ids``: elements the stakeholder
  knows exist. Unknown nodes/edges are REMOVED from the stakeholder's world
  model (never shortcut edges).
- ``visible_node_attributes`` / ``visible_edge_attributes``: per-element
  property knowledge. A known property keeps its value; a property the
  stakeholder knows to be absent is None; an unknown property of a known
  element is ``DONT_KNOW``.
- ``concept_overrides``: per-concept knobs — whether the stakeholder's local
  description and terminology are known, each independently (``DONT_KNOW``
  otherwise), and optional local (possibly wrong) terms.
"""

from typing import Optional

from pydantic import BaseModel, Field

from tau2.domains.business_interview.graph import DONT_KNOW

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")
_EDGE_PROPS = ("condition",)


class ConceptKnowledgeOverride(BaseModel):
    """Per-concept stakeholder knowledge knobs.

    ``description_known`` / ``terms_known`` default to True; set False to
    make the stakeholder's description/terminology ``DONT_KNOW`` (e.g. "I
    know this thing exists and how it is called, but not what it is").
    ``local_terms`` overrides the canonical terms with the stakeholder's own
    wording (a wrong/foreign local term is allowed).
    """

    description_known: bool = True
    terms_known: bool = True
    local_terms: Optional[list[str]] = None


class StakeholderFilter(BaseModel):
    """What a stakeholder knows of the Truth.

    - ``visible_node_ids`` / ``visible_edge_ids``: which nodes / edges exist
      in the stakeholder's world model.
    - ``visible_node_attributes``: per-node property knowledge
      (activity/actor/system/reads/writes/rationale); a node not listed
      knows nothing about any property (all DONT_KNOW).
    - ``visible_edge_attributes``: per-edge condition knowledge.
    - ``concept_overrides``: per-concept description/term knobs.

    Hidden information must remain unknown, not invented: unknown properties
    are ``DONT_KNOW`` and unknown elements are removed.
    """

    name: str
    visible_node_ids: list[str] = Field(default_factory=list)
    visible_edge_ids: list[str] = Field(default_factory=list)
    visible_node_attributes: dict[str, list[str]] = Field(default_factory=dict)
    visible_edge_attributes: dict[str, list[str]] = Field(default_factory=dict)
    concept_overrides: dict[str, ConceptKnowledgeOverride] = Field(default_factory=dict)

    def node_properties_for(self, node_id: str) -> set[str]:
        """The set of node properties the stakeholder knows for ``node_id``."""
        return set(self.visible_node_attributes.get(node_id, []))

    def edge_properties_for(self, edge_id: str) -> set[str]:
        """The set of edge properties (condition) known for ``edge_id``."""
        return set(self.visible_edge_attributes.get(edge_id, []))

    def concept_description_for(self, concept_id: str, truth_concept):
        """The stakeholder's local description of a Truth concept, or
        DONT_KNOW."""
        override = self.concept_overrides.get(concept_id)
        if override is not None and not override.description_known:
            return DONT_KNOW
        return truth_concept.description

    def concept_terms_for(self, concept_id: str, truth_concept):
        """The stakeholder's local terms for a Truth concept (may be a local
        or wrong wording), or DONT_KNOW."""
        override = self.concept_overrides.get(concept_id)
        if override is not None and not override.terms_known:
            return DONT_KNOW
        if override is not None and override.local_terms is not None:
            return list(override.local_terms)
        return list(truth_concept.canonical_terms)
