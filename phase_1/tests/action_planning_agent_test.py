# action_planning_agent_test.py

# ---------------------------------------------------------
# TODO: 1 - Import all required libraries
# ---------------------------------------------------------
from config.env import load_env
from services.llm_service import LLMService
from phase_1.workflow_agents.base_agents import ActionPlanningAgent


def main():
    # ---------------------------------------------------------
    # Shared LLM service
    # ---------------------------------------------------------
    service = LLMService()

    knowledge = """
    # Fried Egg
    1. Heat pan with oil or butter
    2. Crack egg into pan
    3. Cook until white is set (2-3 minutes)
    4. Season with salt and pepper
    5. Serve

    # Scrambled Eggs
    1. Crack eggs into a bowl
    2. Beat eggs with a fork until mixed
    3. Heat pan with butter or oil over medium heat
    4. Pour egg mixture into pan
    5. Stir gently as eggs cook
    6. Remove from heat when eggs are just set but still moist
    7. Season with salt and pepper
    8. Serve immediately

    # Boiled Eggs
    1. Place eggs in a pot
    2. Cover with cold water (about 1 inch above eggs)
    3. Bring water to a boil
    4. Remove from heat and cover pot
    5. Let sit: 4-6 minutes for soft-boiled or 10-12 minutes for hard-boiled
    6. Transfer eggs to ice water to stop cooking
    7. Peel and serve
    """

    # ---------------------------------------------------------
    # TODO: 3 - Instantiate the ActionPlanningAgent
    # ---------------------------------------------------------
    action_agent = ActionPlanningAgent(
        service=service,
        knowledge=knowledge
    )

    # ---------------------------------------------------------
    # TODO: 4 - Test the agent
    # ---------------------------------------------------------
    prompt = "One morning I wanted to have scrambled eggs"

    print("\nUser Prompt:")
    print(prompt)

    steps = action_agent.extract_steps_from_prompt(prompt)

    print("\nExtracted Action Steps:")
    for step in steps:
        print(step)


if __name__ == "__main__":
    main()
