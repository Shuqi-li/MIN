

import argparse
import random
import numpy as np 
import torch
import run_demo as models
from run_demo import dgl_collate_func
from utils1 import *
import os
import pathlib
from time import time
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

    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--batch_size", default=4, type=int) #256
    parser.add_argument("--LR", default=1e-4, type=float)
    parser.add_argument("--decay", default=0.0, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=64, type=int)
    parser.add_argument("--train_epoch", default=5, type=int)
    parser.add_argument("--test_every_X_epoch", default=20, type=int)  

    parser.add_argument("--result_folder", default="./result/", type=str) 



    
    parser.add_argument("--score_encoder", default=True, type=bool)
    parser.add_argument("--score_threshold", default=0.01, type=float)
    parser.add_argument("--target_global_1d", default=True, type=bool)
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
    parser.add_argument("--max_p", default=768, type=int)
    parser.add_argument("--input_dim_drug", default=2586, type=int)
    #parser.add_argument("--input_dim_drug_2d", default=1024, type=int)
    parser.add_argument("--input_dim_protein", default = len(vocab_list), type=int)
    parser.add_argument("--hidden_dim_drug", default =128, type=int)
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
        

    ###MPNN 
    
    parser.add_argument("--gnn_hid_dim_drug", default=64, type=int)
    parser.add_argument("--gnn_num_layer", default=1, type=int)
    parser.add_argument("--gnn_activation", default=F.relu, type=int)

    

    ## alphafold
    
    parser.add_argument("--max_p_2d", default = 128, type=int)
    parser.add_argument("--max_msa", default = 50, type=int)
    '''
    parser.add_argument("--AlphaFold_seq_len", default=128, type=int)
    parser.add_argument("--AlphaFold_depth", default=1, type=int)
    parser.add_argument("--AlphaFold_dim", default=128, type=int)
    parser.add_argument("--AlphaFold_heads", default=4, type=int)    
    parser.add_argument("--AlphaFold_dim_head", default=32, type=int)
    parser.add_argument("--AlphaFold_attn_dropout", default=0., type=float)
    parser.add_argument("--AlphaFold_ff_dropout", default=0., type=float)
    '''

    ## mlp
    parser.add_argument("--MLP_out_size_O", default=128, type=int)
    parser.add_argument("--MLP_hidden_size_list_O", default=[256, 256, 128], type=int)
    parser.add_argument("--MLP_hidden_size_list_N", default=[512, 256, 64], type=int)
    parser.add_argument("--MLP_drop_out", default=0.1, type=float)
    ## mix
    
    parser.add_argument("--Mix_input_size_drug_1d", default=128, type=int)
    parser.add_argument("--Mix_input_size_drug_2d", default=256, type=int)
    parser.add_argument("--Mix_input_size_target_1d", default=64, type=int)
    parser.add_argument("--Mix_input_size_target_2d", default=276, type=int)
    parser.add_argument("--Mix_hidden_size", default=512, type=int)
    parser.add_argument("--Mix_out_size", default=128, type=int)
    
    
    config = parser.parse_args()
    set_seed(config)



    '''
    
    数据加载模式
    'mode_1': drug12 + target12 + msa + score + y
    'mode_2': drug12 + target12 + msa + y
    'mode_3': drug1 + target1 + score + y
    'mode_4': drug1 + target1 + y
    'mode_5': drug2 + target1 + score + y
    'mode_6': drug2 + target1 + y
    '''
    
    testFoldPath = f"{this_dir}/DUDE/dataPre/DUDE-foldTest1"
    testProteinList = getTestProteinList(testFoldPath)
    decoy_path = f"{this_dir}/DUDE/decoy_smile"
    active_path = f"{this_dir}/DUDE/active_smile"
    contactPath = f"{this_dir}/DUDE/contactMap"
    data_dict = getDataDict(testProteinList, active_path, decoy_path, contactPath)

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

    drug_encoding_1d = 'Transformer'
    drug_encoding_2d = 'DGL_GCN'
    target_encoding = 'Transformer'


    print('model')
    model = models.model_initialize(config, data_mode=data_mode)
    '''
    model_path = f"{this_dir}/result/test_mode4_fold2/model.pt"
    model.load_pretrained(model_path)
    result={}
    mean_auc, mean_auprc, mean_f1,  mean_precision, mean_ROCE0, mean_ROCE1, mean_ROCE2, mean_ROCE5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    '''
    t_start = time()
    total_data = pd.DataFrame(columns = ['drug', 'target', 'label'])
    for item in testProteinList:
        
        test_data = data_dict[item]
        test = pd.DataFrame(test_data, columns = ['drug', 'target', 'label'])
        #df_test= data_process_nosplit(data_mode = data_mode, X_drug =test['drug'],  X_target =test['target'],  y=test['label'], drug_encoding_1d =drug_encoding_1d, drug_encoding_2d =drug_encoding_2d, target_encoding = target_encoding)
        total_data = total_data.append(test, ignore_index=True)
        '''
        auc, auprc, f1, loss, logits, precision, ROCE0, ROCE1, ROCE2, ROCE5 = model.predict(df_test)
        
        result[item] = [auc, auprc, f1, loss, logits,  precision, ROCE0, ROCE1, ROCE2, ROCE5]
        print([auc, auprc, f1,  precision, ROCE0, ROCE1, ROCE2, ROCE5])
        
        mean_auc += auc * len(df_test)
        mean_auprc += auprc * len(df_test)
        mean_f1 += f1 * len(df_test)
        mean_precision += precision *len(df_test)
        mean_ROCE0 += ROCE0 * len(df_test)
        mean_ROCE1 += ROCE1 * len(df_test)
        mean_ROCE2 += ROCE2 * len(df_test)
        mean_ROCE5 += ROCE5 * len(df_test)
        '''
        t_now = time()
        print(str(int(t_now - t_start)/3600)[:7] + " hours")
    #np.save(f"{this_dir}/result/test_mode4_fold3/test_result.npy", result)

    print(len(total_data))
    print(len(total_data['drug'].unique()))
    #print([mean_auc/total_data, mean_auprc/total_data, mean_f1/total_data,  mean_precision/total_data, mean_ROCE0/total_data, mean_ROCE1/total_data, mean_ROCE2/total_data, mean_ROCE5/total_data])

    
    
    
    
    
