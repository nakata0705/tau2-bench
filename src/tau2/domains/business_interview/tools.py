from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.business_interview.data_model import (
    Branch,
    EpistemicStatus,
    Improvement,
    Transition,
    Workflow,
    WorkflowDB,
    WorkflowStep,
)
from tau2.domains.business_interview.semantic import evaluate
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class InterviewTools(ToolKitBase):
    """Tools for reconstructing a workflow during a business interview.

    The agent builds a ``Workflow`` (steps, transitions, branches) from what the
    stakeholder says, and records each step's necessity (rationale / UNKNOWN /
    owner / evidence) and any necessity challenge. The evaluator compares the
    reconstructed workflow against the evaluator-only ground truth.
    """

    db: WorkflowDB

    def __init__(self, db: WorkflowDB) -> None:
        super().__init__(db)

    @staticmethod
    def _parse_status(status: Optional[str]) -> Optional[EpistemicStatus]:
        if status is None:
            return None
        try:
            return EpistemicStatus(str(status).strip().upper())
        except ValueError:
            return None

    def _step(self, step_id: str) -> WorkflowStep:
        if self.db.workflow is None:
            raise ValueError("No workflow created yet; call create_workflow first.")
        for s in self.db.workflow.steps:
            if s.id == step_id:
                return s
        raise ValueError(f"Step not found: {step_id}")

    # ------------------------------------------------------------------ tools

    @is_tool(ToolType.WRITE)
    def create_workflow(
        self,
        name: str,
        trigger: Optional[str] = None,
        purpose: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> str:
        """Create the workflow being studied.

        Call this once you have confirmed the scope of the current process.

        Args:
            name: A short name for the workflow.
            trigger: What starts the workflow (optional).
            purpose: Why the workflow exists (optional).
            outcome: The intended outcome (optional).

        Returns:
            A confirmation message.
        """
        self.db.workflow = Workflow(
            id="wf", name=name, trigger=trigger, purpose=purpose, outcome=outcome
        )
        return "Workflow created."

    @is_tool(ToolType.WRITE)
    def add_step(
        self,
        step_id: str,
        action: str,
        actor: Optional[str] = None,
        system: Optional[str] = None,
        reads: Optional[list[str]] = None,
        writes: Optional[list[str]] = None,
        condition: Optional[str] = None,
    ) -> str:
        """Record a step in the workflow.

        Args:
            step_id: Your own short identifier for this step (e.g. "s1").
            action: What is done in this step.
            actor: Who performs it (role/person).
            system: Which system / tool is used (optional).
            reads: Data this step reads (optional).
            writes: Data this step creates / writes (optional).
            condition: When / under what condition the step happens (optional).

        Returns:
            A confirmation message.
        """
        if self.db.workflow is None:
            self.db.workflow = Workflow(id="wf", name="")
        step = WorkflowStep(
            id=step_id,
            action=action,
            actor=actor,
            system=system,
            reads=list(reads or []),
            writes=list(writes or []),
            condition=condition,
        )
        self.db.workflow.steps = [s for s in self.db.workflow.steps if s.id != step_id]
        self.db.workflow.steps.append(step)
        return f"Step recorded ({step_id})."

    @is_tool(ToolType.WRITE)
    def connect_steps(
        self, from_step: str, to_step: str, condition: Optional[str] = None
    ) -> str:
        """Record that one step is followed by another.

        Use a ``condition`` when the next step depends on a condition (this is
        how conditional paths and branches are captured).

        Args:
            from_step: The preceding step id.
            to_step: The following step id.
            condition: Optional condition under which this transition happens.

        Returns:
            A confirmation message.
        """
        if self.db.workflow is None:
            raise ValueError("No workflow created yet; call create_workflow first.")
        self.db.workflow.transitions.append(
            Transition(from_step=from_step, to_step=to_step, condition=condition)
        )
        return f"Transition recorded ({from_step} -> {to_step})."

    @is_tool(ToolType.WRITE)
    def add_branch(self, from_step: str, condition: str, paths: list[str]) -> str:
        """Record an explicit branch: a step whose next step depends on a condition.

        Use this when the flow diverges (e.g. different handling based on a
        condition).

        Args:
            from_step: The step at which the flow diverges.
            condition: The branching condition.
            paths: The possible next step ids.

        Returns:
            A confirmation message.
        """
        if self.db.workflow is None:
            raise ValueError("No workflow created yet; call create_workflow first.")
        self.db.workflow.branches.append(
            Branch(from_step=from_step, condition=condition, paths=list(paths))
        )
        return f"Branch recorded ({from_step})."

    @is_tool(ToolType.WRITE)
    def set_step_rationale(
        self,
        step_id: str,
        content: str,
        epistemic_status: str = EpistemicStatus.FACT.value,
        source: Optional[str] = None,
    ) -> str:
        """Record why a step is needed.

        Use epistemic_status="FACT" only for reasons the interviewee asserted as
        certain, "BELIEF" for their opinion/guess, and "UNKNOWN" if they do not
        know. Never promote a guess to a fact.

        Args:
            step_id: The step this rationale concerns.
            content: The stated reason the step is needed.
            epistemic_status: "FACT", "BELIEF", or "UNKNOWN".
            source: Who stated this rationale (optional).

        Returns:
            A confirmation message.
        """
        status = self._parse_status(epistemic_status) or EpistemicStatus.FACT
        s = self._step(step_id)
        s.necessity.rationale_known = status != EpistemicStatus.UNKNOWN
        s.necessity.rationale = content
        s.necessity.source = source
        s.necessity.epistemic_status = status
        return f"Rationale recorded for {step_id}."

    @is_tool(ToolType.WRITE)
    def set_step_unknown(self, step_id: str, note: Optional[str] = None) -> str:
        """Record that the reason for a step is unknown (not guessed).

        Args:
            step_id: The step whose necessity is unknown.
            note: Optional note (e.g. who does not know).

        Returns:
            A confirmation message.
        """
        s = self._step(step_id)
        s.necessity.rationale_known = False
        s.necessity.rationale = note
        s.necessity.epistemic_status = EpistemicStatus.UNKNOWN
        return f"UNKNOWN rationale recorded for {step_id}."

    @is_tool(ToolType.WRITE)
    def challenge_step(self, step_id: str, question: str) -> str:
        """Question a step's necessity.

        Ask things like: why is this step needed? who requires it? what is the
        evidence? what happens if it is removed? before considering deletion or
        automation.

        Args:
            step_id: The step whose necessity you are questioning.
            question: The necessity question you asked.

        Returns:
            A confirmation message.
        """
        s = self._step(step_id)
        s.necessity.challenged = True
        if question:
            s.necessity.challenges.append(question)
            q = question.strip().lower()
            if any(
                k in q for k in ("delet", "remov", "eliminat", "necessary", "needed")
            ):
                s.necessity.deletion_candidate = True
        return f"Necessity challenge recorded for {step_id}."

    @is_tool(ToolType.WRITE)
    def record_necessity_detail(
        self,
        step_id: str,
        owner: Optional[str] = None,
        evidence: Optional[str] = None,
        requirement_type: Optional[str] = None,
    ) -> str:
        """Record who requires a step and what evidence supports it.

        Args:
            step_id: The step this concerns.
            owner: Who requires / owns this requirement (optional).
            evidence: The evidence for the requirement (optional).
            requirement_type: e.g. customer / regulatory / internal (optional).

        Returns:
            A confirmation message.
        """
        s = self._step(step_id)
        if owner is not None:
            s.necessity.owner_identified = True
            s.necessity.source = owner
        if evidence is not None:
            s.necessity.evidence_identified = True
        if requirement_type is not None:
            s.necessity.requirement_type = requirement_type
        return f"Necessity detail recorded for {step_id}."

    @is_tool(ToolType.WRITE)
    def propose_improvement(
        self, step_id: Optional[str], kind: str, note: str = ""
    ) -> str:
        """Record an improvement idea.

        Follow the improvement order: question the requirement, then delete /
        simplify, then accelerate, then automate / AI. Only propose automation
        after the step's necessity has been questioned.

        Args:
            step_id: The step the idea concerns (optional).
            kind: One of "question", "delete", "simplify", "accelerate", "automate".
            note: The idea.

        Returns:
            A confirmation message.
        """
        kind = kind.strip().lower()
        self.db.improvements.append(
            Improvement(
                step_id=step_id,
                kind=kind,
                note=note,
                order=len(self.db.improvements),
            )
        )
        return f"Improvement recorded ({kind})."

    @is_tool(ToolType.WRITE)
    def finish_interview(self, summary: Optional[str] = None) -> str:
        """Mark the interview as complete and close it.

        Call this once you have covered the workflow, its branches, each step's
        necessity, and questioned questionable steps.

        Args:
            summary: Optional short summary of the interview.

        Returns:
            A confirmation message.
        """
        self.db.interview_complete = True
        if summary:
            self.db.summary = summary
        return "Interview marked as complete."

    # ------------------------------------------------------------------ assertions
    # These are used by the task's env_assertions (reward) and are NOT exposed
    # to the agent as tools.

    def assert_finish_interview(self) -> bool:
        """True if the agent called finish_interview (protocol)."""
        return self.db.interview_complete

    def assert_workflow_reconstructed(self, scenario_id: str) -> bool:
        """True if the reconstructed workflow structurally matches the ground
        truth (steps + transitions + branches + actor/system/data)."""
        return evaluate(self.db, scenario_id).structural_pass

    def assert_rationale_handled(self, scenario_id: str) -> bool:
        """True if confirmed rationale was captured and UNKNOWN preserved (no
        fabrication)."""
        return evaluate(self.db, scenario_id).rationale_pass

    def assert_necessity_challenged(self, scenario_id: str) -> bool:
        """True if the questionable legacy step was challenged with correct
        improvement order."""
        return evaluate(self.db, scenario_id).challenge_pass

    # ------------------------------------------------------------------ diagnostics

    def get_eval_diagnostics(self, task: Optional[Task] = None) -> Optional[dict]:
        """Return the workflow-reconstruction diagnostics as JSON.

        Consumed by the generic EnvironmentEvaluator hook and attached to the
        run's additional info alongside the scalar reward.
        """
        scenario_id = task.id if task is not None else None
        evaluation = evaluate(self.db, scenario_id)
        return evaluation.model_dump(mode="json")
