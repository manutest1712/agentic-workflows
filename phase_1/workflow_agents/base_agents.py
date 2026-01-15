# DirectPromptAgent class definition
from services.llm_service import LLMService

# DirectPromptAgent class definition
class DirectPromptAgent:

    def __init__(self, service: LLMService):
        self.service = service


    def respond(self, prompt):
        # Generate a response using the OpenAI API

        response = self.service.run(prompt, "")
        return response


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:

    def __init__(self, service: LLMService, persona: str):
        """
        Initialize the agent with an LLM service and a persona.
        """
        self.service = service
        self.persona = persona

    def respond(self, prompt: str):
        """
        Generate a response by augmenting the user prompt
        with a system persona instruction.
        """

        system_prompt = (
            f"You are acting as the following persona:\n"
            f"{self.persona}\n\n"
            f"Forget any previous context and respond only based on this persona."
        )

        response = self.service.run(prompt, system_prompt)
        return response


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:

    def __init__(self, service: LLMService, persona: str, knowledge: str):
        """
        Initialize the agent with an LLM service, persona, and knowledge base.
        """
        self.service = service
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, prompt: str) -> str:
        """
        Generate a response using only the provided knowledge
        and the defined persona.
        """

        system_message = (
            f"You are {self.persona} knowledge-based assistant. "
            f"Forget all previous context.\n\n"
            f"Use only the following knowledge to answer, do not use your own knowledge:\n"
            f"{self.knowledge}\n\n"
            f"Answer the prompt based on this knowledge, not your own."
        )

        response = self.service.run(prompt, system_message)
        return response
