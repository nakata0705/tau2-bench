"""
Layer 1: Simulation execution.

Runs a pre-built orchestrator and evaluates the result.
No registry dependency, no config parsing, no side effects.
"""

from typing import Optional, Union

from loguru import logger

from tau2.data_model.simulation import SimulationRun
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Orchestrator


def run_simulation(
    orchestrator: Union[Orchestrator, FullDuplexOrchestrator],
    *,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    env_kwargs: Optional[dict] = None,
) -> SimulationRun:
    """Run a simulation and evaluate the result.

    Takes a fully constructed orchestrator (with agent, user, environment, and task
    already wired in), runs the simulation, evaluates it, and returns the result
    with reward_info attached.

    This is the lowest-level entry point. It has no dependency on the registry
    or RunConfig -- everything is already encapsulated in the orchestrator.

    Args:
        orchestrator: A fully constructed Orchestrator (half-duplex) or
            FullDuplexOrchestrator (full-duplex/voice). Must have agent, user,
            environment, and task set.
        evaluation_type: The type of evaluation to perform. Defaults to ALL.
        env_kwargs: Additional kwargs passed to the evaluator's environment
            constructor (e.g., retrieval_variant for banking_knowledge).

    Returns:
        SimulationRun with reward_info attached.

    Example:
        # Build your own instances (no registry needed):
        env = MyEnvironment()
        agent = MyAgent(tools=env.get_tools(), domain_policy=env.get_policy())
        user = UserSimulator(llm="gpt-4.1", instructions=task.user_scenario,
                             tools=env.get_user_tools())
        orchestrator = Orchestrator(
            domain="airline", agent=agent, user=user,
            environment=env, task=task, max_steps=100,
        )
        result = run_simulation(orchestrator)
        print(result.reward_info.reward)
    """
    # Run the orchestrator
    simulation = orchestrator.run()

    # Business-interview exposes its concrete post-forgetting evaluator inputs
    # only after the run has been built.  Store them in the offline simulation
    # artifact; this never enters an Agent/User prompt or tool surface.
    input_builder = getattr(orchestrator.environment, "get_evaluation_inputs", None)
    if callable(input_builder):
        try:
            evaluation_inputs = input_builder(
                simulation_seed=simulation.seed,
                user=orchestrator.user,
            )
            dump_inputs = getattr(evaluation_inputs, "model_dump", None)
            if not callable(dump_inputs):
                raise TypeError(
                    "business_interview evaluation inputs must be a Pydantic model"
                )
            info = dict(simulation.info or {})
            info["evaluation_inputs"] = dump_inputs(mode="json")
            simulation.info = info
        except Exception as exc:  # noqa: BLE001 - fail closed on missing inputs
            raise RuntimeError(
                "Could not capture business_interview evaluation inputs"
            ) from exc

    # Save the actual policy used for this simulation
    simulation.policy = orchestrator.environment.get_policy()

    # Extract context from the orchestrator -- no external params needed
    domain = orchestrator.environment.get_domain_name()
    task = orchestrator.task
    is_full_duplex = isinstance(orchestrator, FullDuplexOrchestrator)
    mode = (
        CommunicationMode.FULL_DUPLEX
        if is_full_duplex
        else CommunicationMode.HALF_DUPLEX
    )
    solo_mode = getattr(orchestrator, "solo_mode", False)

    # Evaluate
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=evaluation_type,
        solo_mode=solo_mode,
        domain=domain,
        mode=mode,
        env_kwargs=env_kwargs or {},
    )
    simulation.reward_info = reward_info

    logger.info(
        f"Simulation complete: domain={domain}, task={task.id}, "
        f"reward={reward_info.reward}"
    )

    return simulation
