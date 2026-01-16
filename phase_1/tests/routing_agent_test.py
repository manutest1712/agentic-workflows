# routing_agent_test.py
from phase_1.workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent
from services.llm_service import LLMService


def main():
    # ---------------------------------------------------------
    # 1. Instantiate shared LLM service
    # ---------------------------------------------------------
    service = LLMService()

    # ---------------------------------------------------------
    # 2. Instantiate KnowledgeAugmentedPromptAgents
    # ---------------------------------------------------------

    texas_agent = KnowledgeAugmentedPromptAgent(
        service=service,
        persona="Texas history expert.",
        knowledge="You know everything about Texas."
    )

    europe_agent = KnowledgeAugmentedPromptAgent(
        service=service,
        persona="European history professor.",
        knowledge="You know everything about Europe."
    )

    math_agent = KnowledgeAugmentedPromptAgent(
        service=service,
        persona="Mathematics Teacher.",
        knowledge=(
            "You know everything about math, you take prompts with numbers, "
            "extract math formulas, and show the answer without explanation"
        )
    )

    # ---------------------------------------------------------
    # 3. Define agent functions / lambdas
    # ---------------------------------------------------------

    texas_func = lambda prompt: texas_agent.respond(prompt)
    europe_func = lambda prompt: europe_agent.respond(prompt)
    math_func = lambda prompt: math_agent.respond(prompt)

    # ---------------------------------------------------------
    # 4. Assign agents to router
    # ---------------------------------------------------------

    agents = [
        {
            "name": "texas_agent",
            "description": "Questions about Texas cities, history, and places in Texas",
            "func": texas_func
        },
        {
            "name": "europe_agent",
            "description": "Questions about European history, cities, and countries",
            "func": europe_func
        },
        {
            "name": "math_agent",
            "description": "Math word problems, arithmetic, and calculations",
            "func": math_func
        }
    ]

    router = RoutingAgent(service=service, agents=agents)

    # ---------------------------------------------------------
    # 5. Test routing with prompts
    # ---------------------------------------------------------

    test_prompts = [
        "Tell me about the history of Rome, Texas",
        "Tell me about the history of Rome, Italy",
        "One story takes 2 days, and there are 20 stories"
        "f=2+3/5*38. What is the value of f?"
    ]

    for prompt in test_prompts:
        print("\n" + "=" * 60)
        print(f"User Prompt: {prompt}")
        result = router.route(prompt)
        print("Final Answer:")
        print(result)


if __name__ == "__main__":
    main()
