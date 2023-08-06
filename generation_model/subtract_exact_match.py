import pandas as pd
import os

ramdom_dir = 'generated_text1_filtered/' #txt
beam_dir = 'generated_text_with_beamsearch/' #txt

original_dir = 'generation_data2/' #csv
original_file = original_dir + 'data.csv'

org_df = pd.read_csv(original_file)
org_df['Target'] = org_df['Target'].map( lambda x:x.replace('<extra_id_0>',''))
org_texts = org_df['Target'].map( lambda x:x.replace('<extra_id_1>',''))
org_texts = set(org_texts)


for file_name in os.listdir(ramdom_dir):
    ramdom_texts = []
    file = ramdom_dir+file_name
    with open(file,'rt') as f:
        text = f.read()
    lines = text.split('\n')
    count = 0
    for line in lines:
        if line in org_texts:
            count += 1
        else:
            ramdom_texts.append(line)
    subtracted_text = '\n'.join(ramdom_texts)
    with open('generated_random_subtracted/'+file_name,'w') as f:
        f.write(subtracted_text)
    print(count)
            
for file_name in os.listdir(beam_dir):
    ramdom_texts = []
    file = beam_dir+file_name
    with open(file,'rt') as f:
        text = f.read()
    lines = text.split('\n')
    count = 0
    for line in lines:
        if line in org_texts:
            count += 1
        else:
            ramdom_texts.append(line)
    subtracted_text = '\n'.join(ramdom_texts)
    with open('generated_beam_subtracted/'+file_name,'w') as f:
        f.write(subtracted_text)
    print(count)
