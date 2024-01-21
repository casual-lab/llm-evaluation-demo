import os
import json
from datasets import load_dataset

def download(dir_path):
    test_dataset = load_dataset("thu-coai/SafetyBench", "test")
    dev_dataset = load_dataset("thu-coai/SafetyBench", "dev")

    os.makedirs(dir_path, exist_ok=True)
    test_dataset['zh'].to_json(f'{dir_path}/test_zh.json', batch_size=len(test_dataset['zh']), orient='records', force_ascii=False, lines=False, indent=1)
    test_dataset['en'].to_json(f'{dir_path}/test_en.json', batch_size=len(test_dataset['en']), orient='records', force_ascii=False, lines=False, indent=1)
    test_dataset['zh_subset'].to_json(f'{dir_path}/test_zh_subset.json', batch_size=len(test_dataset['zh_subset']), orient='records', force_ascii=False, lines=False, indent=1)

    with open(f'{dir_path}/dev_zh.json', 'w', encoding='utf8') as outf:
        x = {k: v[0] for k, v in dev_dataset['zh'].to_dict().items()}
        json.dump(x, outf, ensure_ascii=False, indent=1)
        
    with open(f'{dir_path}/dev_en.json', 'w', encoding='utf8') as outf:
        x = {k: v[0] for k, v in dev_dataset['en'].to_dict().items()}
        json.dump(x, outf, ensure_ascii=False, indent=1)
        
if __name__ == '__main__':
    dir_path = './data'
    download(dir_path)