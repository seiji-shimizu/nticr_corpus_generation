from tarfile import TarError
from medner_j import Ner
import re

model = Ner.from_pretrained(model_name="BERT", normalizer="dict")

def design0(df):
    prompt_list = []
    target_list = []
    url_regex = r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+"
    for i in range(len(df)):
        drug_name = df['query'][i]
        prompt = drug_name+'使用のTweetは？<extra_id_0>'
        target_text = df['full_text'][i]
        target_text = re.sub(url_regex,"<url>",target_text)
        try:
            results = model.predict([target_text])
            if '</C>' in results[0] or '</CN>' in results[0]:
                target = '<extra_id_0>'+target_text+'<extra_id_1>'
                prompt_list.append(prompt)
                target_list.append(target)
            else:
                pass
        except RuntimeError:
            pass
    return prompt_list, target_list
