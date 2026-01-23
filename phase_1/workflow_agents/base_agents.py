import os
from datetime import datetime
from enum import Enum

import pandas as pd
from langchain_openai import OpenAI

from services.llm_service import LLMService
import csv
import numpy as np
from typing import List, Dict
from openai import OpenAI

# DirectPromptAgent class definition
class DirectPromptAgent:

    def __init__(self, service: LLMService):
        self.service = service


    def respond(self, prompt):
        # Generate a response using the OpenAI API
        print(f"The Prompt: {prompt}")
        response = self.service.run(prompt, "")
        return response


class WorkflowStepClassifierAgent:
    def __init__(self, service: LLMService):
        self.service = service

    def is_user_story(self, promt: str):
        SYSTEM_PROMPT = """
        You are a strict workflow step classification agent.

        Your task is to determine whether the given text indicates the USER STORIES step
        in a product development workflow.

        Rules:
        1. Respond YES only if the text explicitly refers to the USER STORIES phase.
        2. Do NOT infer intent.
        3. This is NOT about user story content.

        Output:
        Respond with ONLY one word: YES or NO.
        """
        return self.respond(promt, SYSTEM_PROMPT)

    def respond(self, prompt: str, system_message: str):
        response = self.service.run(prompt, system_message)
        result = response.strip().upper()

        if result == "YES":
            return True

        return False


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

        print(f"Prompt: {prompt}")
        print(f"System Prompt: {system_prompt}")
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
        print(f"Promt : {prompt}");
        system_message = (
            f"{self.persona}\n\n"
            f"Forget all previous context.\n\n"
            f"Use only the following knowledge to answer, do not use your own knowledge:\n"
            f"{self.knowledge}\n\n"
            f"Answer the prompt based on this knowledge, not your own."
        )

        print(f"System message -- KnowledgeAagent - {system_message}")

        response = self.service.run(prompt, system_message)
        return response


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, service: LLMService,
                 persona: str,
                 chunk_size: int =2000,
                 chunk_overlap: int =100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.service = service
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        response = self.service.get_embedding(text)
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        system_message = (
            f"You are {self.persona}, a knowledge-based assistant. "
            f"Forget previous context.\n\n"
            f"Use ONLY the following information to answer:\n"
            f"{best_chunk}"
        )

        # 6. Ask the LLM using the centralized service
        response = self.service.run(
            prompt=f"Answer the following question based only on the provided information:\n{prompt}",
            system_message=system_message
        )

        return response


class EvaluationAgent:
    """
    EvaluationAgent manages an iterative loop between a worker agent and
    an evaluator (LLM) to ensure responses meet predefined criteria.
    """

    def __init__(
        self,
        service: LLMService,
        persona: str,
        evaluation_criteria: str,
        worker_agent,
        max_interactions: int = 3
    ):
        """
        Initialize the EvaluationAgent with required attributes.
        """
        # 1. Declare class attributes
        self.service = service
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        self.client = OpenAI(
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://openai.vocareum.com/v1"
            ),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def evaluate(self, initial_prompt: str) -> dict:
        """
        Manages interactions between the worker agent and the evaluator
        until the response meets criteria or max_interactions is reached.
        """
        print(f"Initial Prompt {initial_prompt}")
        prompt_to_evaluate = initial_prompt
        final_response = None
        final_evaluation = None

        # 2. Interaction loop limited by max_interactions
        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i + 1} ---")

            # 3. Retrieve worker response
            print(f"Step 1: Worker agent generates a response. Prompt {prompt_to_evaluate}")
            final_response = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{final_response}")

            # 4. Construct evaluation prompt
            evaluation_prompt = (
                f"Does the following answer:\n{final_response}\n\n"
                f"Meet this criteria:\n{self.evaluation_criteria}\n\n"
                f"Respond with '--Yes--' or '--No--' and explain why."
            )

            # 5. Evaluation message structure (temperature=0 via LLMService)
            final_evaluation = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": evaluation_prompt},
                ],
                temperature=0
            ).choices[0].message.content

            # final_evaluation = self.client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #         {"role": "system", "content": self.persona},
            #         {"role": "user", "content": evaluation_prompt},
            #     ],
            #     temperature=0
            # ).choices[0].message.content

            print(f"Evaluator Agent Evaluation:\n{final_evaluation}")

            # 6. Check if evaluation is positive
            if final_evaluation.lower().startswith("--yes--"):
                print("✅ Final solution accepted.")
                break

            # 7. Generate correction instructions
            correction_prompt = (
                f"The following answer failed evaluation:\n{final_response}\n\n"
                f"Reasons:\n{final_evaluation}\n\n"
                f"Provide clear instructions to correct the answer."
            )

            instructions = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are an expert providing correction instructions."},
                    {"role": "user", "content": correction_prompt},
                ],
                temperature=0
            ).choices[0].message.content

            # self.service.run(
            #     prompt=correction_prompt,
            #     system_message="You are an expert providing correction instructions."
            # )

            print(f"Instructions to fix:\n{instructions}")

            # Prepare refined prompt for next iteration
            prompt_to_evaluate = (
                f"Original prompt:\n{initial_prompt}\n\n"
                f"Previous answer:\n{final_response}\n\n"
                f"Apply ONLY the following corrections:\n{instructions}"
            )

        # 8. Return final results
        return {
            "final_response": final_response,
            "evaluation": final_evaluation,
            "iterations": i + 1
        }




class RoutingAgent:
    """
    Routes user prompts to the most relevant agent based on
    embedding similarity between the prompt and agent descriptions.
    """

    def __init__(self, service: LLMService, agents: List[Dict]):
        """
        :param service: Shared LLMService instance (for embeddings)
        :param agents: List of agent definitions:
                       {
                           "name": str,
                           "description": str,
                           "func": callable
                       }
        """
        self.service = service
        self.agents = agents  # TODO 1 completed

        # Pre-compute embeddings for agent descriptions
        for agent in self.agents:
            print(f"Agent Description: {agent["description"]}")
            agent["embedding"] = self.get_embedding(agent["description"])

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding using text-embedding-3-large
        """
        embedding = self.service.get_embedding(text)  # TODO 2 completed
        return np.array(embedding)

    def route(self, user_input: str) -> str:
        """
        Route the user prompt to the most appropriate agent.
        """

        # TODO 4 - Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)

        best_agent = None
        best_score = -1.0

        for agent in self.agents:
            # TODO 5 - Compute the embedding of the agent description
            agent_emb = agent.get("embedding")
            if agent_emb is None:
                continue

            # Cosine similarity
            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )

            print(f"[Router] Similarity with {agent['name']}: {similarity:.3f}")

            # TODO 6 - Select best agent
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)

    def get_best_agent(self, user_input: str):
        """
        Return the best matching agent (dict) and similarity score.
        """

        input_emb = self.get_embedding(user_input)

        best_agent = None
        best_score = -1.0

        for agent in self.agents:
            agent_emb = agent.get("embedding")
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (
                    np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )

            print(f"[Router] Similarity with {agent['name']}: {similarity:.3f}")

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        return best_agent, best_score

    def get_best_agent_name(self, user_input: str) -> str:
        """
        Return the name of the best matching agent.
        """

        best_agent, best_score = self.get_best_agent(user_input)

        if best_agent is None:
            return "No suitable agent found"

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["name"]


class ActionPlanningAgent:
    """
    ActionPlanningAgent extracts a list of action steps from a user prompt
    using ONLY the provided knowledge.
    """

    def __init__(self, service: LLMService, knowledge: str):
        # 1. Initialize agent attributes
        self.service = service
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt: str) -> list[str]:
        """
        Extracts action steps from the user prompt based on provided knowledge.

        Args:
            prompt (str): User request describing an action to be performed.

        Returns:
            list[str]: Ordered list of steps.
        """

        system_message = (
            "You are an action planning agent. "
            "Using ONLY the provided knowledge, identify steps that are "
            "directly relevant to the user's request. "
            "Include related planning steps if required. "
            "Do NOT include unrelated planning steps. "
            "Do NOT expand, split, or rewrite steps. "
            "Do NOT invent new steps. "
            "Return the result strictly as a numbered list. "
            "If only one step is relevant, return only that step. "
            "Forget all previous context.\n\n"
            f"This is your knowledge:\n{self.knowledge}"
        )

        # 3. Call LLM via LLMService
        response_text = self.service.run(
            prompt=prompt,
            system_message=system_message
        )

        # 5. Clean and format the extracted steps
        steps = [
            step.strip("- ").strip()
            for step in response_text.split("\n")
            if step.strip()
        ]

        return steps


'''
class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize the agent attributes here

    def extract_steps_from_prompt(self, prompt):

        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        # Provide the following system prompt along with the user's prompt:
        # "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {pass the knowledge here}"

        response_text = ""  # TODO: 4 - Extract the response text from the OpenAI API response

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")

        return steps
'''
