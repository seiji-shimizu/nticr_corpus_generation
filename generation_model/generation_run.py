import random
import torch
import argparse
import numpy as np 

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl


from generation_dataset import Tweet_SympDataset
from generation_experiment import T5FineTuner

# 事前学習済みモデル
PRETRAINED_MODEL_DIR = "sonoisa/t5-base-japanese"

#data_dir = 'generation_data1'
data_dir = 'generation_data2'

DATADIR = 'generation_checkpoint2'

# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# GPU利用有無
USE_GPU = torch.cuda.is_available()
# 各種ハイパーパラメータ
args_dict = dict(
    data_dir=data_dir,  # データセットのディレクトリ
    model_name_or_path=PRETRAINED_MODEL_DIR,
    tokenizer_name_or_path=PRETRAINED_MODEL_DIR,

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    max_input_length=512,
    max_target_length=512,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=18,
    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42,
)

args = argparse.Namespace(**args_dict)
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    amp_backend='apex',
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    accelerator="gpu", 
    devices=[1]
)
#転移学習の実行（GPUを利用すれば1エポック10分程度）
model = T5FineTuner(args)

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=False,
    dirpath=DATADIR,
)

trainer = pl.Trainer(**train_params,callbacks = [checkpoint])

print(trainer.device_ids)
#trainer.fit(model)
trainer.fit(model,ckpt_path='generation_checkpoint1/epoch=3-step=198516-v1.ckpt')

del model

