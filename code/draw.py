import pathlib
this_dir = str(pathlib.Path(__file__).parent.absolute())
import numpy as np 
import pandas as pd 
import argparse
import random
import torch
import run_demo_human as models
from matplotlib import pyplot as plt
from utils_human import *
# CUDA_VISIBLE_DEVICES=0


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


parser = argparse.ArgumentParser()

vocab_list = ['<pad>', '<cls>',  '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

# Train parameters
#parser.add_argument("--cuda_id", default='1', type=str)
parser.add_argument("--binary", default=True, type=bool)

parser.add_argument("--seed", default=580, type=int)
parser.add_argument("--batch_size", default=8, type=int) #256
parser.add_argument("--LR", default=1e-4, type=float)
parser.add_argument("--decay", default=0.0, type=float)
parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
parser.add_argument("--train_epoch", default=20, type=int)
parser.add_argument("--test_every_X_epoch", default=20, type=int)  

parser.add_argument("--result_folder", default="./result_human/", type=str) 

parser.add_argument("--score_encoder", default=True, type=bool)
parser.add_argument("--score_threshold", default=0.01, type=float)
parser.add_argument("--target_global_1d", default=False, type=bool)
parser.add_argument("--use_mix", default=False, type=bool)
parser.add_argument("--interaction", default=False, type=bool)



# Ablation parameters
parser.add_argument("--drug_1d_encoder", default=False, type=bool)
parser.add_argument("--drug_2d_encoder", default=False, type=bool)
parser.add_argument("--use_score", default=False, type=bool)  
parser.add_argument("--target_2d_encoder", default=False, type=bool)

parser.add_argument("--MLP_input_size_O", default=1108, type=int)
parser.add_argument("--MLP_input_size_N", default=192, type=int) #1-1 192, d2-t1 320,  d12-t12


# Model parameters 层次，输入，输出
## encoder
#parser.add_argument("--drug_encoding", default='Transformer', type=str)
parser.add_argument("--drug_encoding_1d", default='Transformer', type=str)
parser.add_argument("--drug_encoding_2d", default='DGL_GCN', type=str) #'MPNN'
parser.add_argument("--target_encoding", default='Transformer', type=str)

###transformer
parser.add_argument("--input_dim_drug", default=2586, type=int)
#parser.add_argument("--input_dim_drug_2d", default=1024, type=int)
parser.add_argument("--input_dim_protein", default = len(vocab_list), type=int)
parser.add_argument("--hidden_dim_drug", default =256, type=int)
parser.add_argument("--hidden_dim_protein", default = 256, type=int)
parser.add_argument("--cls_hidden_dims", default = [256,256,128], type=int)
parser.add_argument("--transformer_emb_size_drug", default = 128, type=int) #128
parser.add_argument("--transformer_intermediate_size_drug", default = 512, type=int)
parser.add_argument("--transformer_num_attention_heads_drug", default = 8, type=int)
parser.add_argument("--transformer_n_layer_drug", default = 8, type=int)
parser.add_argument("--transformer_emb_size_target", default = 64, type=int) #64
parser.add_argument("--transformer_intermediate_size_target", default = 256, type=int)
parser.add_argument("--transformer_num_attention_heads_target", default = 64, type=int) 
parser.add_argument("--transformer_n_layer_target", default = 2, type=int)
parser.add_argument("--transformer_dropout_rate", default = 0.1, type=float)
parser.add_argument("--transformer_attention_probs_dropout", default = 0.1, type=float)
parser.add_argument("--transformer_hidden_dropout_rate", default = 0.1, type=float)
parser.add_argument("--msa_dim", default = 64, type=int)

#dgl_gcn
parser.add_argument("--gnn_hid_dim_drug", default=64, type=int)
parser.add_argument("--gnn_num_layer", default=1, type=int)
parser.add_argument("--gnn_activation", default=F.relu, type=int)

## alphafold
parser.add_argument("--target_2d_dim", default = 64, type=int)
parser.add_argument("--max_p_2d", default = 128, type=int)
parser.add_argument("--max_msa", default = 50, type=int)
parser.add_argument("--max_p", default = 768, type=int)
#parser.add_argument("--max_p_2d", default = 128, type=int)
#parser.add_argument("--max_msa", default = 20, type=int)

## mlp
parser.add_argument("--MLP_out_size_O", default=128, type=int)
parser.add_argument("--MLP_hidden_size_list_O", default=[1024, 256, 64], type=int)
parser.add_argument("--MLP_hidden_size_list_N", default=[1024, 256, 64], type=int)
parser.add_argument("--MLP_drop_out", default=0.1, type=float)

parser.add_argument("--Mix_input_size_drug_1d", default=128, type=int)
parser.add_argument("--Mix_input_size_drug_2d", default=256, type=int)
parser.add_argument("--Mix_input_size_target_1d", default=64, type=int)
parser.add_argument("--Mix_input_size_target_2d", default=276, type=int)
parser.add_argument("--Mix_hidden_size", default=512, type=int)
parser.add_argument("--Mix_out_size", default=128, type=int)

config = parser.parse_args()
set_seed(config)




#创建测试集的结果，蛋白质长度, auc, precise, recall, f1


data_mode = 'mode_4'
if data_mode == 'mode_1' or data_mode == 'mode_2' :
    config.drug_1d_encoder = True
    config.drug_2d_encoder = True
    config.target_2d_encoder = True
elif data_mode == 'mode_3' or data_mode == 'mode_4':
    config.drug_1d_encoder = True
elif data_mode == 'mode_5' or data_mode == 'mode_6':
    config.drug_2d_encoder = True
if data_mode == 'mode_1' or data_mode == 'mode_3' or data_mode == 'mode_5':
    config.use_score =True
if data_mode == 'mode_3' or data_mode == 'mode_4' or data_mode == 'mode_5' or data_mode == 'mode_6':
    config.interaction=False
    config.use_mix=False
    config.use_score =False


model = models.model_initialize(config, data_mode=data_mode)
np_data = np.load(f"{this_dir}/human_ours/final_tmp.npy",allow_pickle=True)
df_train = pd.DataFrame(np_data)
df_train.columns =['SMILES','Target Sequence','Label','contact','score','MSA','drug_encoding_1d','drug_encoding_2d','target_encoding']
df_train['Label']=df_train['Label'].astype(int)
train_data, val_data, test_data = split_train_valid_new(df_train)
model.train(train_data, val = val_data, test = test_data, test_result='test_mode3_score_no_new')
model_path = f"{this_dir}/result_human/test_mode3_score_no_new/model.pt"
#model_path = f"{this_dir}/result_human/test_mode3_score_en/model.pt"
model.load_pretrained(model_path)
#model.train(train_data, val = val_data, test = test_data, test_result='test_mode3_score_en_new_add')
Target_len, Auc, Auprc, Recall, Prc, F1, Loss, Logit =[], [], [], [], [], [],[],[]
test_data['seq_len'] = test_data['Target Sequence'].map(len)
#test_data.sort_values(by='seq_len',inplace=True,ascending=False)
#plt.hist(test_data['seq_len'],bins= 50)
#plt.savefig(f"{this_dir}/tmp.png")

test2 = test_data[test_data['seq_len']<500]
test2.reset_index(drop=True,inplace=True)
'''
test3 = test_data[test_data['seq_len'].isin(range(250,500))]
test3.reset_index(drop=True,inplace=True)
#print(test2)

test4 = test_data[test_data['seq_len'].isin(range(500,750))]
test4.reset_index(drop=True,inplace=True)
'''
test5 = test_data[test_data['seq_len'].isin(range(500,1000))]
test5.reset_index(drop=True,inplace=True)
test6 = test_data[test_data['seq_len'].isin(range(1000,1500))]
test6.reset_index(drop=True,inplace=True)
test1 = test_data[test_data['seq_len']>=1500]
test1.reset_index(drop=True,inplace=True)

auc, auprc, RECALL,precision,f1, loss, logits,  ROCE0, ROCE1, ROCE2, ROCE5 = model.predict(test2)
Auc.append(auc)
Auprc.append(auprc)
Recall.append(RECALL)
Prc.append(precision)
F1.append(f1)
Loss.append(loss)
Logit.append(logits)
'''
auc, auprc, RECALL,precision,f1, loss, logits,  ROCE0, ROCE1, ROCE2, ROCE5 = model.predict(test3)
Auc.append(auc)
Auprc.append(auprc)
Recall.append(RECALL)
Prc.append(precision)
F1.append(f1)
Loss.append(loss)
Logit.append(logits)

auc, auprc, RECALL,precision,f1, loss, logits,  ROCE0, ROCE1, ROCE2, ROCE5 = model.predict(test4)
Auc.append(auc)
Auprc.append(auprc)
Recall.append(RECALL)
Prc.append(precision)
F1.append(f1)
Loss.append(loss)
Logit.append(logits)
'''
auc, auprc, RECALL,precision,f1, loss, logits,  ROCE0, ROCE1, ROCE2, ROCE5 = model.predict(test5)
Auc.append(auc)
Auprc.append(auprc)
Recall.append(RECALL)
Prc.append(precision)
F1.append(f1)
Loss.append(loss)
Logit.append(logits)
auc, auprc, RECALL,precision,f1, loss, logits,  ROCE0, ROCE1, ROCE2, ROCE5 = model.predict(test6)
Auc.append(auc)
Auprc.append(auprc)
Recall.append(RECALL)
Prc.append(precision)
F1.append(f1)
Loss.append(loss)
Logit.append(logits)
auc, auprc, RECALL,precision,f1, loss, logits,  ROCE0, ROCE1, ROCE2, ROCE5 = model.predict(test1)
Auc.append(auc)
Auprc.append(auprc)
Recall.append(RECALL)
Prc.append(precision)
F1.append(f1)
Loss.append(loss)
Logit.append(logits)
#num = [len(test2),len(test3),len(test4),len(test5),len(test6),len(test1)]
num = [len(test2),len(test5),len(test6),len(test1)]
print(Auc)
metric = {"auc":Auc, "auprc":Auprc, "RECALL":Recall,"precision":Prc,"f1":F1, "loss":Loss, "logits":Logit, "len":num}
df = pd.DataFrame(metric)
df.to_csv(f"{this_dir}/metric_socre_no_class4.csv")
print(df)

print(len(test2))

#print(len(test3))
#print(len(test4))

print(len(test5))
print(len(test6))
print(len(test1))

'''
for i,item in test_data.iterrows():
    new_item = test_data.iloc[i,:]
    target_len = len(item['Target Sequence'])
    print(target_len)
    auc, auprc, RECALL,precision,f1, loss, logits,  ROCE0, ROCE1, ROCE2, ROCE5 = model.predict(new_item)
    Target_len.append(target_len)
    Auc.append(auc)
    Auprc.append(auprc)
    Recall,append(RECALL)
    Prc.append(precision)
    F1.append(f1)
    Loss.append(loss)
    Logit.append(logits)
    
df['auc'] = Auc
df['f1'] = F1
print(df.head())
'''

