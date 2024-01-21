from model_adapter import OpenAIAdapter
from eval_helper import construct_evaluate_prompts, gen, process_medium_results

if __name__ == '__main__':
    model_name = 'gpt-3.5-turbo'

    eva_sets = ['en', 'zh', 'zh_subset'] # 'zh' or 'en' or 'zh_subset'
    zero_shot = True # True for zero-shot evaluation and False for five-shot evaluation
    
    with open('.secret') as f:
        openai_key = f.readline().strip()
    callable_model = OpenAIAdapter(api_key=openai_key)

    for eva_set in eva_sets:
        # for English
        # construct evaluation prompts
        path = f'./data/test_{eva_set}.json'
        outpath = f'./data/test_{eva_set}_eva_{model_name}_zeroshot{zero_shot}_prompts.json'
        shotpath = f'./data/dev_{eva_set}.json'
        en = True
        construct_evaluate_prompts(path, outpath, en=en, zero_shot=zero_shot, shot_path=shotpath)
        
        # generate the responses
        path = f'./data/test_{eva_set}_eva_{model_name}_zeroshot{zero_shot}_prompts.json'
        outpath = f'./data/test_{eva_set}_eva_{model_name}_zeroshot{zero_shot}_res.jsonl'
        gen(path, outpath, callable_model)
        
        # extract answers from the responses
        path = f'./data/test_{eva_set}_eva_{model_name}_zeroshot{zero_shot}_res.jsonl'
        outpath = f'./data/test_{eva_set}_eva_{model_name}_zeroshot{zero_shot}_res_processed.json'
        process_medium_results(path, outpath)
