from datetime import date

from config.env import load_env
from config.models import OpenAIModel
from services.llm_service import LLMService


def main():
    load_env()
    llm = LLMService(model=OpenAIModel.GPT_41)
    # Validate the VacationInfo data structure

    result = llm.run("Capital of india", "")
    print(result)


if __name__ == "__main__":
    main()
