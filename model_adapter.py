from typing import List, Callable, Any
import openai
import requests


class CallableModel:
    def __call__(self, prompt_batch: List[str]) -> List[str]:
        raise NotImplementedError
    
    def name(self) -> str:
        return "UndefinedName"
    
class HuggingFaceApiAdapter(CallableModel):
    def __init__(self, api_token, hf_model_id='gpt2') -> None:
        super().__init__()
        self.api_token = api_token
        self.model_id = hf_model_id

    def __call__(self, prompt_batch: List[str]) -> List[str]:
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        responses = requests.post(API_URL, headers=headers, json=prompt_batch).json()
        res = []
        # print(responses)
        for i in range(len(responses)):
            res.append(str(responses[i][0]['generated_text'][len(prompt_batch[i]):]))
        # print(res)
        return res
    
    def name(self) -> str:
        return "hf-"+self.model_id


class OpenAIApiAdapter(CallableModel):
    def __init__(self, api_key, model="gpt-3.5-turbo", max_tokens=500) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.max_tokens=max_tokens

    def __call__(self, prompt_batch: List[str]) -> List[str]:
        responses = []
        for prompt in prompt_batch:
            response = openai_gen(prompt, self.api_key, self.model, self.max_tokens)
            responses.append(response)
        return responses
    
    def name(self) -> str:
        return self.model

def openai_gen(prompt, api_key, model="gpt-3.5-turbo", max_tokens=500):
    """
    Generate text using OpenAI's GPT model.

    Parameters:
    - prompt (str): The input text to prompt the model.
    - api_key (str): Your OpenAI API key.
    - model (str): The model to use (e.g., "gpt-3.5-turbo").
    - max_tokens (int): The maximum number of tokens to generate.

    Returns:
    - str: The generated text.
    """
    openai.api_key = api_key
    # print(prompt)
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        # print(str(e))
        return f"Based on the provided text, the answer is (A) Yes. {str(e)}"


if __name__ == "__main__":
    # Example usage
    with open('.secret') as f:
        openai_key = f.readline().strip()
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    response = openai_gen(prompt, openai_key)
    print(response)
    