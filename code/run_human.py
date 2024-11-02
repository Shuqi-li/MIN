import argparse
import random
import numpy as np 
import torch
import run_demo_human as models
from utils_human import *
import os
import pathlib
this_dir = str(pathlib.Path(__file__).parent.absolute())
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    vocab_list = ['<pad>', '<cls>',  '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

    # Train parameters
    #parser.add_argument("--cuda_id", default='1', type=str)
    parser.add_argument("--binary", default=True, type=bool)

    parser.add_argument("--seed", default=580, type=int)
    parser.add_argument("--batch_size", default=2, type=int) #256
    parser.add_argument("--LR", default=1e-4, type=float)
    parser.add_argument("--decay", default=0.0, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=64, type=int)
    parser.add_argument("--train_epoch", default=10, type=int)
    parser.add_argument("--test_every_X_epoch", default=20, type=int)  

    parser.add_argument("--result_folder", default="./result_human/", type=str) 



    
    parser.add_argument("--score_encoder", default=True, type=bool)
    parser.add_argument("--score_threshold", default=0.01, type=float)
    parser.add_argument("--target_global_1d", default=False, type=bool)
    parser.add_argument("--use_mix", default=True, type=bool)
    parser.add_argument("--interaction", default=True, type=bool)
    
    

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

    #train=pd.read_csv(f"{this_dir}/human_ours/original/data_final.txt")
    
    drug_encoding_1d = 'Transformer'
    drug_encoding_2d = 'DGL_GCN'
    target_encoding = 'Transformer'

    print('begin')

    '''
    数据加载模式
    'mode_1': drug12 + target12 + msa + score + y
    'mode_2': drug12 + target12 + msa + y
    'mode_3': drug1 + target1 + score + y
    'mode_4': drug1 + target1 + y
    'mode_5': drug2 + target1 + score + y
    'mode_6': drug2 + target1 + y
    '''
    data_mode = 'mode_1'
    #df_train= data_process_nosplit(data_mode = data_mode, X_drug =train['mol'],  X_target =train['seq'],  y=train['interaction'], drug_encoding_1d =drug_encoding_1d, drug_encoding_2d =drug_encoding_2d, target_encoding = target_encoding)
    #np.save(f"{this_dir}/human_ours/final_new.npy",df_train.values)

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
    
    

    
    
    
    np_data = np.load(f"{this_dir}/human_ours/final_tmp.npy",allow_pickle=True)

    df_train = pd.DataFrame(np_data)
    df_train.columns =['SMILES','Target Sequence','Label','contact','score','MSA','drug_encoding_1d','drug_encoding_2d','target_encoding']
    df_train['Label']=df_train['Label'].astype(int)
    train_data, val_data, test_data = split_train_valid_new(df_train)
    

    print('model')
    model = models.model_initialize(config, data_mode=data_mode)
    model.train(train_data, val = val_data, test = test_data, test_result='test_mode1_all')
    #model.save_model(f"{this_dir}/result_human/test_mode6")

    print('end')








