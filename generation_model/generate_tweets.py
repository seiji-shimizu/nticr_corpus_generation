from __future__ import unicode_literals
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from generation_dataset import Tweet_SympDataset
import textwrap
import argparse
#from prompt_design import design0
import pandas as pd
from os import listdir
# https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja から引用・一部改変
import re
import unicodedata
import pickle

check_point_dir = 'generation_checkpoint2/'
PATH =  check_point_dir + listdir(check_point_dir)[0]
generation_result = 'generated_text/generation_result1.pickle'

USE_GPU = True
args_dict = dict(
    data_dir='',  # データセットのディレクトリ
    model_name_or_path="sonoisa/t5-base-japanese",
    tokenizer_name_or_path="sonoisa/t5-base-japanese",

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    max_input_length=512,
    max_target_length=512,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=24,
    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0
)

args = argparse.Namespace(**args_dict)

PRETRAINED_MODEL_DIR = "sonoisa/t5-base-japanese"
from generation_experiment import T5FineTuner
model_tuner = T5FineTuner.load_from_checkpoint(checkpoint_path=PATH,name_space =args)
trained_model = model_tuner.model
tokenizer = model_tuner.tokenizer

MAX_SOURCE_LENGTH = 50  # 入力される記事本文の最大トークン数
MAX_TARGET_LENGTH = 70 # 生成されるタイトルの最大トークン数


def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s
    
import math
def normalize_text(text):
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text

def preprocess_answer(text):
    return normalize_text(text.replace("\n", ""))

def generate_bodies1(num,prompt):
    inputs = [preprocess_answer(prompt)]
    batch = tokenizer.batch_encode_plus(
    inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
    padding="longest", return_tensors="pt").to('cuda')
    input_ids = batch['input_ids']
    input_mask = batch['attention_mask']
    outputs = trained_model.generate(
        input_ids=input_ids, attention_mask=input_mask, 
        max_length=MAX_TARGET_LENGTH,
        temperature=2.0,  # 生成にランダム性を入れる温度パラメータ
        #num_beams= num,  # ビームサーチの探索幅
        #diversity_penalty=0.5,  # 生成結果の多様性を生み出すためのペナルティパラメータ
        #This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time. Note that diversity_penalty is only effective if group beam search is enabled.
        #num_beam_groups= num,  # ビームサーチのグループ
        do_sample=True, 
        top_k=30,
        num_return_sequences= num,  # 生成する文の数
        repetition_penalty=8.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
    )

    # 生成されたトークン列を文字列に変換する
    generated_texts = [tokenizer.decode(ids, skip_special_tokens=True, 
                                     clean_up_tokenization_spaces=False) 
                    for ids in outputs]
    return generated_texts

def generate_bodies2(num,prompt):
    inputs = [preprocess_answer(prompt)]
    batch = tokenizer.batch_encode_plus(
    inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
    padding="longest", return_tensors="pt").to('cuda')
    input_ids = batch['input_ids']
    input_mask = batch['attention_mask']
    outputs = trained_model.generate(
        input_ids=input_ids, attention_mask=input_mask, 
        max_length=MAX_TARGET_LENGTH,
        temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
        num_beams= num,  # ビームサーチの探索幅
        diversity_penalty=0.5,  # 生成結果の多様性を生み出すためのペナルティパラメータ
        #This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time. Note that diversity_penalty is only effective if group beam search is enabled.
        num_beam_groups= num,  # ビームサーチのグループ
        #do_sample=True, 
        # top_k=30,
        num_return_sequences= num,  # 生成する文の数
        repetition_penalty=8.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
    )

    # 生成されたトークン列を文字列に変換する
    generated_texts = [tokenizer.decode(ids, skip_special_tokens=True, 
                                     clean_up_tokenization_spaces=False) 
                    for ids in outputs]
    return generated_texts



# 推論モード設定
trained_model.eval()
trained_model.to('cuda')
conditions = ['ステロイド', 'インスリン', '抗生剤', 'プレドニゾロン', '抗菌薬', 'ヘパリン', '利尿剤', 'プレドニン',
       '補液', 'ステロイドパルス', 'ワーファリン', 'アルドステロン', 'コルチゾール', 'シクロスポリン',
       '免疫抑制剤', 'アルブミン', 'アミオダロン', 'ＳＴ合剤', '利尿薬', 'カテコラミン', '抗結核薬',
       '抗生物質', 'メチルプレドニゾロン', 'アスピリン', '造影剤', 'フロセミド', 'デキサメサゾン', 'ワルファリン',
       'アシクロビル', 'タクロリムス', 'リツキシマブ', '降圧薬', 'ヒドロコルチゾン', 'メサラジン', '免疫グロブリン', 'チアマゾール', '抗血小板薬', 'イマチニブ', 'グルカゴン', 'γグロブリン', 'トルバプタン',
       '強化インスリン', '降圧剤', 'インターフェロン', 'インフリキシマブ', 'シクロホスファミド', 'メトトレキサート',
       'メロペネム', 'メトホルミン', 'バンコマイシン', 'ミノサイクリン', 'アザチオプリン', 'シスプラチン',
       'ウロキナーゼ', 'グリメピリド', 'デキサメタゾン', 'セフトリアキソン', 'メトロニダゾール', '抗菌剤', '輸液',
       'リバビリン', 'アルガトロバン', 'シクロフォスファミド', 'ガンシクロビル', 'ステロイド剤', 'ノルアドレナリン',
       '抗凝固薬', 'コルヒチン']

prompts = [(i+'使用のTweetは？<extra_id_0>',i) for i in conditions]

#data_dir1 = 'generated_text1/'
# 生成処理を行う
#for p in prompts:
#    file_name = data_dir1+p[1]+'generated_text1.txt'
#    for i in range(100):
#        bodies = generate_bodies1(100,p[0])
#        with open(file_name,"a",newline="",encoding="utf-8") as f:
#            f.writelines('\n'.join(bodies))


data_dir2 = 'generated_text2/'
# 生成処理を行う
for p in prompts:
    bodies = generate_bodies2(1000,p[0])
    file_name = data_dir2+p[1]+'generated_text2.pickle'
    with open(file_name,'wb') as f:
        pickle.dump(bodies,f)