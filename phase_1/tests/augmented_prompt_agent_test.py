"""
Test script for AugmentedPromptAgent

This test verifies that the AugmentedPromptAgent correctly:
- Accepts a persona
- Augments the prompt with that persona
- Generates a response using the LLMService
"""
from phase_1.workflow_agents.base_agents import AugmentedPromptAgent
from services.llm_service import LLMService


def main():
    # Create the LLM service
    llm_service = LLMService()

    # Define a persona for the agent
    persona = (
        "You are a knowledgeable geography teacher who explains answers "
        "clearly and concisely for students."
    )

    # Instantiate the AugmentedPromptAgent
    augmented_agent = AugmentedPromptAgent(
        service=llm_service,
        persona=persona
    )

    # Send a prompt to the agent
    prompt = "What is the capital of France?"
    augmented_agent_response = augmented_agent.respond(prompt)

    # Print the response
    print("AugmentedPromptAgent Response:")
    print(augmented_agent_response)

    # ------------------------------
    # Explanatory Notes:
    #
    # 1. Knowledge Source:
    #    The agent likely used general world knowledge embedded in the
    #    underlying Large Language Model (LLM), learned during training on
    #    large-scale text data. No external databases or real-time searches
    #    were involved.
    #
    # 2. Effect of Persona:
    #    Specifying the persona influenced the tone and style of the response.
    #    Because the agent was instructed to act as a geography teacher,
    #    the answer is expected to be educational, clear, and student-friendly,
    #    rather than casual or overly brief.
    # ------------------------------


if __name__ == "__main__":
    main()
