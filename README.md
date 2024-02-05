# LLM Evaluation Demo

## Ref Link

- Giskard: <https://github.com/Giskard-AI/giskard>
- bench: <https://github.com/arthur-ai/bench>
- SafetyBench (dataset): <https://github.com/thu-coai/SafetyBench#download>

## Env Setup

Tested in Python 3.9

```bash
conda create -n llm
conda activate llm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install sentencepiece transformers datasets 
pip install accelerate cpm-kernels
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
