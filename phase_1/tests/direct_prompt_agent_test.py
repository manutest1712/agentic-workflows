from config.env import load_env
from phase_1.workflow_agents.base_agents import DirectPromptAgent
from services.llm_service import LLMService


def main():
    # 1. Instantiate the LLM service
    llm_service = LLMService()

    # 2. Instantiate the DirectPromptAgent
    direct_agent = DirectPromptAgent(service=llm_service)

    # 3. Prompt the agent
    prompt = "What is the Capital of France?"
    response = direct_agent.respond(prompt)

    # 4. Print the response
    print("Agent Response:")
    print(response)

    # 5. Explain the knowledge source
    print("\nKnowledge Source Explanation:")
    print(
        "The agent answered this question using general world knowledge "
        "embedded in the underlying Large Language Model (LLM). "
        "This information was learned during the model’s training on a large "
        "corpus of publicly available, licensed, and human-created text, "
        "rather than from any external database or real-time lookup."
    )


if __name__ == "__main__":
    main()
