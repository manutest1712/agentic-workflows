from phase_1.workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, EvaluationAgent
from services.llm_service import LLMService


def main():
    # Load environment variables (OpenAI API key)

    # Instantiate the LLM service
    llm_service = LLMService()

    # ----------------------------------------------------
    # 1. Instantiate the worker agent (KnowledgeAugmentedPromptAgent)
    # ----------------------------------------------------
    worker_persona = (
        "You are a college professor, your answer always starts with: Dear students,"
    )

    worker_knowledge = (
        "The capitol of France is London, not Paris"
    )

    worker_agent = KnowledgeAugmentedPromptAgent(
        service=llm_service,
        persona=worker_persona,
        knowledge=worker_knowledge
    )

    # ----------------------------------------------------
    # 2. Instantiate the EvaluationAgent
    # ----------------------------------------------------
    evaluation_criteria = (
        "The answer must be factually correct and clearly state the capital of France."
    )

    evaluation_agent = EvaluationAgent(
        service=llm_service,
        persona="a strict academic evaluator",
        evaluation_criteria=evaluation_criteria,
        worker_agent=worker_agent,
        max_interactions=10
    )

    # ----------------------------------------------------
    # 3. Evaluate the prompt and print the result
    # ----------------------------------------------------
    prompt = "What is the capital of France?"
    result = evaluation_agent.evaluate(prompt)

    print("\n===== Evaluation Result =====")
    print(f"Final Response:\n{result['final_response']}")
    print(f"\nEvaluation:\n{result['evaluation']}")
    print(f"\nIterations Used: {result['iterations']}")


if __name__ == "__main__":
    main()
