from medner_j import Ner
import re
import os
import pickle

model = Ner.from_pretrained(model_name="BERT", normalizer="dict")
save_dir1 = 'generated_text1_filtered/'
save_dir2 = 'generated_text2_filtered/'

dir_name1 = 'generated_text1/'
dir_name2 = 'generated_text2/'

for file in os.listdir(dir_name1):
    file_name = dir_name1+file
    with open(file_name,'r') as f:
        text = f.read()
    text_lines = text.split('\n')
    text_set = set(text_lines)
    text_list = list(text_set)
    results = model.predict(text_list)
    for i,result in enumerate(results):
        if '</C>' in result or '</CN>' in result:
            with open(save_dir1+file,'a') as f:
                f.write(text_list[i])
                f.write('\n')

for file in os.listdir(dir_name2):
    file_name = dir_name2+file
    with open(file_name,'rb') as f:
        text_lines = pickle.load(f)
    text_set = set(text_lines)
    text_list = list(text_set)
    results = model.predict(text_list)
    for i,result in enumerate(results):
        if '</C>' in result or '</CN>' in result:
            with open(save_dir2+file,'a') as f:
                f.write(text_list[i])
                f.write('\n')

