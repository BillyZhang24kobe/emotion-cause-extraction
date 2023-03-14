import openai
import os
from prompts import TRS_2, TRS_8, RSE_5, RSE_10, TRS_4
import pandas as pd

# OpenAI GPT-3 Config
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

# prompt responses for train, val and test data
# shots = ['TRS-2', 'TRS-8', 'RSE-5', 'RSE-10']
shots = ['TRS-4']
shots_map = {
    'TRS-2': TRS_2,
    'TRS-4': TRS_4,
    'TRS-8': TRS_8,
    'RSE-5': RSE_5,
    'RSE-10': RSE_10
}
splits = ['train', 'dev']
# splits = ['train']

# for ghazi dataset
folds = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']

for shot in shots:
    for fold in folds:
        for split in splits:
            df = pd.read_csv('../../data/ghazi/{}/{}.tsv'.format(fold, split), delimiter="\t")
            exp_lst = []
            examples = shots_map[shot]
            for idx, row in df.iterrows():
                doc = row['document']
                print(doc)
                gpt_prompt = examples + "\n\n Q: '{}' Why does the cause result in the emotion mentioned in the text? \n\n A:".format(doc.replace('\"', ''))
                try:
                    response = openai.Completion.create(
                        engine="text-davinci-002",
                        prompt=gpt_prompt,
                        temperature=0.7,
                        max_tokens=256,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    exp_lst.append(response['choices'][0]['text'])
                except Exception as e:
                    print(e)
                    print('Bad Input Item for OpenAI...')
                    print('THIS FAILS THE API: ' + doc)
                    exp_lst.append('NULL')

            print('finished {}!!!'.format(shot))
            df['explanation'] = exp_lst
            df.to_csv('./data/ghazi/{}/eca-{}-cleaned-exp-{}.tsv'.format(fold, split, shot), sep='\t', index=False)
