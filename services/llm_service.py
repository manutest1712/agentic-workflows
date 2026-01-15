from config.models import OpenAIModel
from config.openai_client import get_openai_client
from config.settings import DEFAULT_MODEL


class LLMService:
    def __init__(self, model: OpenAIModel = DEFAULT_MODEL):
        self.client = get_openai_client()
        self.model = model

    def run(
            self,
            prompt: str,
            system_message: str = "You are a helpful assistant"
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return response.choices[0].message.content

