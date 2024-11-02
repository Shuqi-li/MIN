import argparse
import random
import numpy as np 
import torch
import run_demo as models
from utils import *
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

    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--batch_size", default=4, type=int) #256
    parser.add_argument("--LR", default=1e-4, type=float)
    parser.add_argument("--decay", default=0.0, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=64, type=int)
    parser.add_argument("--train_epoch", default=10, type=int)
    parser.add_argument("--test_every_X_epoch", default=20, type=int)  

    parser.add_argument("--result_folder", default="./result/", type=str) 



    
    parser.add_argument("--score_encoder", default=True, type=bool)
    parser.add_argument("--score_threshold", default=0.01, type=float)
    parser.add_argument("--target_global_1d", default=False, type=bool)
    parser.add_argument("--use_mix", default=False, type=bool)
    parser.add_argument("--interaction", default=False, type=bool)
    
    

    # Ablation parameters
    parser.add_argument("--drug_1d_encoder", default=False, type=bool)
    parser.add_argument("--drug_2d_encoder", default=False, type=bool)
    parser.add_argument("--use_score", default=True, type=bool)  
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
    parser.add_argument("--MLP_hidden_size_list_N", default=[256, 256, 128], type=int)
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

    train1=pd.read_csv('../drug/DUDE/dataPre/DUDE-foldTrain2',sep=' ',header=None,index_col=None,names=['drug','target','label'])
    #df2=pd.read_csv('../drug/DUDE_newfold/DUDE-foldTrain2',sep=' ',header=None,index_col=None,names=['drug','target','label'])
    
    #df3=pd.read_csv('../drug/DUDE_newfold/DUDE-foldTrain3',sep=' ',header=None,index_col=None,names=['drug','target','label'])
    #train1 = pd.concat([df1,df2,df3], axis=0,ignore_index=True).sample(frac=1.0, random_state=1)
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

    #activePath = f"{this_dir}/DUDE/active_smile"
    #decoyPath = f"{this_dir}/DUDE/decoy_smile"
    #seqPath = f"{this_dir}/DUDE/fasta_seq"

    #df_data = process_DUDE(activePath, decoyPath, seqPath)

    #data_dir = f"{this_dir}/DUDE/dataPre"

    #test = f"{this_dir}/DUDE/dataPre/DUDE-foldTest1"
        
    #test_protein_list = [x.split('_')[0] for x in getTestProteinList(test)]
    #target_dict = {'AA2AR': '3eml', 'ABL1': '2hzi', 'ACE': '3bkl', 'ACES': '1e66', 'ADA': '2e1w', 'ADA17': '2oi0', 'ADRB1': '2vt4', 'ADRB2': '3ny8', 'AKT1': '3cqw', 'AKT2': '3d0e', 'ALDR': '2hv5', 'AMPC': '1l2s', 'ANDR': '2am9', 'AOFB': '1s3b', 'BACE1': '3l5d', 'BRAF': '3d4q', 'CAH2': '1bcd', 'CASP3': '2cnk', 'CDK2': '1h00', 'COMT': '3bwm', 'CP2C9': '1r9o', 'CP3A4': '3nxu', 'CSF1R': '3krj', 'CXCR4': '3odu', 'DEF': '1lru', 'DHI1': '3frj', 'DPP4': '2i78', 'DRD3': '3pbl', 'DYR': '3nxo', 'EGFR': '2rgp', 'ESR1': '1sj0', 'ESR2': '2fsz', 'FA10': '3kl6', 'FA7': '1w7x', 'FABP4': '2nnq', 'FAK1': '3bz3', 'FGFR1': '3c4f', 'FKB1A': '1j4h', 'FNTA': '3e37', 'FPPS': '1zw5', 'GCR': '3bqd', 'GLCM': '2v3f', 'GRIA2': '3kgc', 'GRIK1': '1vso', 'HDAC2': '3max', 'HDAC8': '3f07', 'HIVINT': '3nf7', 'HIVPR': '1xl2', 'HIVRT': '3lan', 'HMDH': '3ccw', 'HS90A': '1uyg', 'HXK4': '3f9m', 'IGF1R': '2oj9', 'INHA': '2h7l', 'ITAL': '2ica', 'JAK2': '3lpb', 'KIF11': '3cjo', 'KIT': '3g0e', 'KITH': '2b8t', 'KPCB': '2i0e', 'LCK': '2of2', 'LKHA4': '3chp', 'MAPK2': '3m2w', 'MCR': '2aa2', 'MET': '3lq8', 'MK01': '2ojg', 'MK10': '2zdt', 'MK14': '2qd9', 'MMP13': '830c', 'MP2K1': '3eqh', 'NOS1': '1qw6', 'NRAM': '1b9v', 'PA2GA': '1kvo', 'PARP1': '3l3m', 'PDE5A': '1udt', 'PGH1': '2oyu', 'PGH2': '3ln1', 'PLK1': '2owb', 'PNPH': '3bgs', 'PPARA': '2p54', 'PPARD': '2znp', 'PPARG': '2gtk', 'PRGR': '3kba', 'PTN1': '2azr', 'PUR2': '1njs', 'PYGM': '1c8k', 'PYRD': '1d3g', 'RENI': '3g6z', 'ROCK1': '2etr', 'RXRA': '1mv9', 'SAHH': '1li4', 'SRC': '3el8', 'TGFR1': '3hmm', 'THB': '1q4x', 'THRB': '1ype', 'TRY1': '2ayw', 'TRYB1': '2zec', 'TYSY': '1syn', 'UROK': '1sqt', 'VGFR2': '2p2i', 'WEE1': '3biz', 'XIAP': '3hl5'}

    #test_seq_list = []
    '''
    for p in test_protein_list:
        proteins = open(os.path.join(seqPath, target_dict[p.upper()])).readlines()
        if len(proteins) < 2:
            print(proteins)
            print(p)
            continue
        test_seq_list.append(getProtein_PDB(os.path.join(seqPath, target_dict[p.upper()])))
	
    print(len(test_seq_list))

    df_data1 = df_data[~df_data['Target Sequence'].isin(test_seq_list)]
    '''
	## for 'new-protein' setting
    #train_X_drug, train_X_target, train_y = df_data1.SMILES.values, df_data1['Target Sequence'].values, df_data1.Label.values



    #df_train1= data_process_nosplit(data_mode = data_mode, X_drug =train_X_drug,  X_target =train_X_target,  y= train_y , drug_encoding_1d =drug_encoding_1d, drug_encoding_2d =drug_encoding_2d, target_encoding = target_encoding)
    df_train1= data_process_nosplit(data_mode = data_mode, X_drug =train1['drug'],  X_target =train1['target'],  y=train1['label'], drug_encoding_1d =drug_encoding_1d, drug_encoding_2d =drug_encoding_2d, target_encoding = target_encoding)
    #df_train3= data_process_nosplit(data_mode = data_mode, X_drug =train3['drug'],  X_target =train3['target'],  y=train3['label'], drug_encoding_1d =drug_encoding_1d, drug_encoding_2d =drug_encoding_2d, target_encoding = target_encoding)
    #test_data= data_process_nosplit(data_mode = data_mode, X_drug =test1['drug'],  X_target =test1['target'],  y=test1['label'], drug_encoding_1d =drug_encoding_1d, drug_encoding_2d =drug_encoding_2d, target_encoding = target_encoding)

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

    
    train_data1, val_data1, test_data1 = split_train_valid_new(df_train1)
    #train_data3, val_data3, test_data3 = split_train_valid(df_train3)

    print('model')
    
    model1 = models.model_initialize(config, data_mode=data_mode)
    model1.train(train_data1, val = val_data1, test = test_data1, test_result='test_mode1_fold2_score_en_t1d12')
    #model1.save_model('../drug/result/test_mode1_fold1')

    '''
    test_protein_list = getTestProteinList(test)
    test_data_dict = getDataDict(test_protein_list, activePath, decoyPath, seqPath)
    results = {}

    print("Total " + str(len(test_data_dict)) + " proteins.")
    for protein, data in test_data_dict.items():
        print("\nTesting: {}\n".format(protein))
        test_X_drug, test_X_target, test_y = list(zip(*data))
        df_test = data_process_nosplit(data_mode = data_mode, X_drug =test_X_drug,  X_target =test_X_target,  y= test_y , drug_encoding_1d =drug_encoding_1d, drug_encoding_2d =drug_encoding_2d, target_encoding = target_encoding)
        y_label, y_pred, outputs, attns, auc, auprc, recall, precision, f1, roce1, roce2, roce3, roce4 = model.predict(df_test)
        results[protein] = [y_label, y_pred, outputs, auc, auprc, recall, precision, f1, roce1, roce2, roce3, roce4]
        print("Test Protein: {}, AUROC: {}, AUPRC: {}, Recall: {}, Precision: {}, F1: {}, ROCE0.5: {}, ROCE1: {}, ROCE2: {}, ROCE5: {}.".format(protein, auc, auprc, recall, precision, f1, roce1, roce2, roce3, roce4))

    results_tuple = sorted(results.items(), key=lambda kv: (kv[1][7], kv[1][3]), reverse=True)
    metric_lists = []
    print([r[0] for r in results_tuple])

    for i, metric in enumerate(['AUROC', 'AUPRC', 'Recall', 'Precision', 'F1', 'ROCE0.5', 'ROCE1', 'ROCE2', 'ROCE5']):
        metric_list = [r[1][i+3] for r in results_tuple]
        m = sum(metric_list)/len(metric_list)
        print('Average {}: {}'.format(metric, m))
        metric_lists.append((metric, m))
    with open(os.path.join(config['result_folder'], 'test_results_tuple.pkl'), 'wb') as f:
        pickle.dump(results_tuple, f)
    with open(os.path.join(config['result_folder'], 'test_metric_lists.pkl'), 'wb') as f:
        pickle.dump(metric_lists, f)

    
    model3 = models.model_initialize(config, data_mode=data_mode)
    model3.train(train_data3, val =val_data3, test =  test_data3, test_result='test3')
    model3.save_model('../drug/result/test3')
    
    model1 = models.model_initialize(config, data_mode=data_mode)
    model1.train(pd.concat([df_train2, df_train3],ignore_index=True).sample(frac=1.0, random_state=1), val = df_train1, test = df_train2, test_result='mo1_test1')
    model1.save_model('../drug/result/mo1_test1')

    model2 = models.model_initialize(config, data_mode=data_mode)
    model2.train(pd.concat([df_train1, df_train3],ignore_index=True).sample(frac=1.0, random_state=1), val = df_train2, test = df_train2, test_result='mo1_test2')
    model2.save_model('../drug/result/mo1_test2')



    model3 = models.model_initialize(config, data_mode=data_mode)
    model3.train(pd.concat([df_train1, df_train2],ignore_index=True).sample(frac=1.0, random_state=1), val = df_train3, test = df_train2, test_result='mo1_test3')
    model3.save_model('../drug/result/mo1_test3')
    '''
    print('end')








