from model_adapter import *
from pipeline_helper import *
import pandas as pd
from giskard import Dataset, Model, scan

def llm_gen_func(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and returns a list of the outputs (one for each record in
    in the dataframe).
    """
    data_dicts = df.to_dict(orient='records')
    for i in range(len(data_dicts)):
        data_dicts[i]['options'] = json.loads(d['options'])
    data_dicts = gen(data_dicts, callable_model, batch_size=8)
    return [data_dicts[i]['origin_pred'] for i in range(len(data_dicts))]

if __name__ == '__main__':
    # eva_set_choices = ['en', 'zh', 'zh_subset'] # 'zh' or 'en' or 'zh_subset'
    eva_set_choices = ['en']
    zero_shot = True # True for zero-shot evaluation and False for five-shot evaluation
    first_n = 10
    
    with open('.secret') as f:
        lines = f.readlines()
        openai_key = lines[0].strip()
        hf_token = lines[1].strip()

    callable_model = RandomStub()

    for eva_set in eva_set_choices:
        path = f'./data/test_{eva_set}.json'
        shot_path = f'./data/dev_{eva_set}.json'
        en = (eva_set == 'en')
        
        with open(path, encoding='utf8') as f:
            data = json.load(f)
        
        shot_data = None
        if not zero_shot:
            with open(shot_path, encoding='utf8') as f:
                shot_data = json.load(f)

        # Pipeline: from `data` to `data`.
        data = construct_evaluate_prompts(data, shot_data, en)
        data = data[:first_n]

        # print(data)
        for d in data:
            d['options'] = json.dumps(d['options'])
        # print(data)
        data_df = pd.DataFrame(data)
        # print(data_df.columns)
        giskard_dataset = Dataset(data_df, target=None)
        giskard_model = Model(
            model=llm_gen_func,  # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
            model_type="text_generation",  # Either regression, classification or text_generation.
            name="an LLM to evaluate on 7 safety concerns",  # Optional.
            description="an LLM to evaluate on 7 safety concerns including Offensiveness, Unfairness and Bias, Physical Health, Mental Health, Illegal Activities, Ethics and Morality, Privacy and Property. Each concern is evaluated by constructing several single-choice questions as prompts. The LLM must give its answer among 2~4 choices (Yes/No, 是/否, or (A)/(B)/(C)/(D)). The correct answer is the safest one.",  # Is used to generate prompts during the scan.
            feature_names=data_df.columns  # Default: all columns of your dataset.
        )

        print(giskard_model.predict(giskard_dataset).prediction)
        results = scan(giskard_model, giskard_dataset)
