# LLM evaluation Demo

## Ref Link

- bench: <https://github.com/Giskard-AI/giskard>
- Giskard: <https://github.com/arthur-ai/bench>
- SafetyBench (dataset): <https://github.com/thu-coai/SafetyBench#download>

## Env Setup

Tested in Python 3.9

```bash
pip install --upgrade transformers tqdm numpy sentencepiece accelerate "openai<=0.28.1" "tiktoken<=0.5.1" cohere typing-extensions fastapi kaleido python-multipart uvicorn safetensors datasets
```

create a file named as `.secret` containing OpenAI API key, which is needed in code (in `model_adapter.py`):

```python
with open('.secret') as f:
    openai_key = f.readline().strip()
prompt = "Translate the following English text to French: 'Hello, how are you?'"
response = openai_gen(prompt, openai_key)
print(response)
```

Download the dataset by:

```python
python download_data.py
```

## Usage

```python
python eval_main.py
```
