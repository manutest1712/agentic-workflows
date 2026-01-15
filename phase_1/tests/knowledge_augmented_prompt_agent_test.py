from config.env import load_env
from phase_1.workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
from services.llm_service import LLMService


def main():

    # 1. Instantiate the LLM service
    llm_service = LLMService()

    # 2. Define persona and knowledge
    persona = "a factual geography assistant"

    knowledge = (
        "France is a country in Western Europe. "
        "The capital city of France is Paris."
    )

    # 3. Instantiate the KnowledgeAugmentedPromptAgent
    knowledge_agent = KnowledgeAugmentedPromptAgent(
        service=llm_service,
        persona=persona,
        knowledge=knowledge
    )

    # 4. Prompt the agent
    prompt = "What is the capital of France?"
    knowledge_augmented_agent_response = knowledge_agent.respond(prompt)

    # 5. Print the response
    print("KnowledgeAugmentedPromptAgent Response:")
    print(knowledge_augmented_agent_response)

    # 6. Explain the knowledge source and persona impact
    print("\nKnowledge Source Explanation:")
    print(
        "The agent generated this response strictly using the explicitly "
        "provided knowledge snippet rather than relying on its general "
        "training data. The system prompt constrained the model to ignore "
        "external or prior knowledge."
    )

    print("\nPersona Influence Explanation:")
    print(
        "The specified persona influenced the tone and style of the response, "
        "ensuring it remained factual and concise, consistent with a "
        "knowledge-based geography assistant."
    )


if __name__ == "__main__":
    main()
