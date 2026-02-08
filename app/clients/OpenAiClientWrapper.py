


from openai import OpenAI

class OpenAIClientWrapper:
    def __init__(self, api_key: str, llm_model: str):
        self.api_key = api_key
        print(api_key, " blabla")
        self.llm_model = llm_model
        self.client = OpenAI(api_key=api_key)


    def llm_generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
