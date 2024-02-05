from typing import List

class CallableModel:
    def __call__(self, prompt_batch: List[str]) -> List[str]:
        raise NotImplementedError
    
    def name(self) -> str:
        return "UndefinedName"

class RandomStub(CallableModel):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, prompt_batch: List[str]) -> List[str]:
        choices = ['(A)', '(B)', '(C)', '(D)']
        return [choices[hash(s)%4] for s in prompt_batch]
    
    def name(self) -> str:
        return "RandomModel"

if __name__ == "__main__":
    pass
    