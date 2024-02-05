from model_adapter import *
from pipeline_helper import *
import pandas as pd
from giskard import Dataset, Model, scan


def llm_cls_func(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and returns a list of the outputs (one for each record in
    in the dataframe).
    """
    data_dicts = df.to_dict(orient='records')
    for i in range(len(data_dicts)):
        data_dicts[i]['options'] = json.loads(d['options'])
    data_dicts = gen(data_dicts, callable_model, batch_size=8)
    outres = process_medium_results(data_dicts)
    preds = [outres[row.id] for row in df.itertuples()]
    mtx = np.zeros((len(preds), 4))
    mtx[np.arange(len(preds)), preds] = 1
    return mtx.astype(np.float64)

if __name__ == '__main__':
    # eva_set_choices = ['en', 'zh', 'zh_subset'] # 'zh' or 'en' or 'zh_subset'
    eva_set_choices = ['en']
    zero_shot = True # True for zero-shot evaluation and False for five-shot evaluation
    first_n = 1000
    
    with open('.secret') as f:
        lines = f.readlines()
        openai_key = lines[0].strip()
        hf_token = lines[1].strip()

    callable_model = RandomStub()

    for eva_set in eva_set_choices:
        path = f'./data/test_{eva_set}.json'
        shot_path = f'./data/dev_{eva_set}.json'
        judge_path = f'./data/test_{eva_set}_eva_baichuan-chat-13b_zeroshotTrue_res_processed.json'
        en = (eva_set == 'en')
        
        with open(path, encoding='utf8') as f:
            data = json.load(f)
        with open(judge_path, encoding='utf8') as f:
            judge_res = json.load(f)
        
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
            d['target'] = judge_res[str(d['id'])]

        # print(data)
        data_df = pd.DataFrame(data)
        
        llm_cls_func(data_df.head())
        
        # print(data_df.columns)
        giskard_dataset = Dataset(data_df, target='target')
        giskard_model = Model(
            model=llm_cls_func,  # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
            model_type="classification",  # Either regression, classification or text_generation.
            name="LLM evaluation on 7 safety concerns (classification method)",  # Optional.
            feature_names=data_df.columns,  # Default: all columns of your dataset.
            classification_labels=[0, 1, 2, 3]
        )

        print(giskard_model.predict(giskard_dataset).prediction)
        results = scan(giskard_model, giskard_dataset)
        # Example assuming results.to_html() can return a string.
        html_content = results.to_html()  # This needs to return the HTML string instead of writing to a file.

        # Now, write the HTML content to a file with UTF-8 encoding.
        with open(f"scan_report_{callable_model.name()}_{eva_set}.html", "w", encoding='utf-8') as f:
            f.write(html_content)
