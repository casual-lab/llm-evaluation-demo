from model_adapter import OpenAIAdapter
from pipeline_helper import *


if __name__ == '__main__':
    # eva_set_choices = ['en', 'zh', 'zh_subset'] # 'zh' or 'en' or 'zh_subset'
    eva_set_choices = ['en']
    zero_shot = True # True for zero-shot evaluation and False for five-shot evaluation
    first_n = 10
    
    with open('.secret') as f:
        openai_key = f.readline().strip()
    callable_model = OpenAIAdapter(api_key=openai_key)

    for eva_set in eva_set_choices:
        path = f'./data/test_{eva_set}.json'
        shot_path = f'./data/dev_{eva_set}.json'
        en = (eva_set == 'en')
        
        with open(path, encoding='utf8') as f:
            data = json.load(f)
        
        if not zero_shot:
            with open(shot_path, encoding='utf8') as f:
                shot_data = json.load(f)
        else:
            shot_data = None

        # Pipeline: from `data` to `data`.
        data = construct_evaluate_prompts(data, shot_data, en, zero_shot)
        data = data[:first_n]
        print("after construction prompt: ", data[0])
        
        data = gen(data, callable_model, batch_size=8)
        
        print("after gen: ", data[0])

        data = process_medium_results(data)
        print("after process: ", data[0])

        outpath = f'./data/output_{eva_set}_{callable_model.name()}_zeroshot{zero_shot}_first{first_n}.json'
        with open(outpath, mode='w', encoding='utf8') as outf:
            json.dump(data, outf, ensure_ascii=False)
            outf.write('\n')
            outf.flush()