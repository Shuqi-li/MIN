import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from DeepPurpose.chemutils import get_mol, atom_features, bond_features, MAX_NB, ATOM_FDIM, BOND_FDIM
from DeepPurpose.utils import *
from subword_nmt.apply_bpe import BPE
import codecs
import pickle
import wget
from zipfile import ZipFile 
import os
import sys


import pathlib
import pathlib
from subword_nmt.apply_bpe import BPE
import codecs
this_dir = str(pathlib.Path(__file__).parent.absolute())



def msa2emb(x_msa, max_p, max_msa):
    vocab_list = ['<pad>', '<cls>',  '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
    #t1 = pbpe.process_line(x).split()  # split
    words2idx = dict(zip(vocab_list, range(len(vocab_list))))
    msa_init = np.asarray([ [words2idx['<cls>']]+[words2idx[i] if (i in vocab_list) else words2idx['<unk>'] for i in list(j)] + [words2idx['<eos>']] for j in x_msa])
    mask_x  = np.ones_like(msa_init)
    if msa_init.shape[1] < max_p :
        x = np.array([np.pad(i,(0, max_p -  msa_init.shape[1]), 'constant', constant_values = 0) for i in msa_init])
        mask_x = np.array([np.pad(i,(0, max_p -  msa_init.shape[1]), 'constant', constant_values = 0) for i in mask_x])
    else:
        x = msa_init[:, :max_p]
        mask_x = mask_x[:, :max_p]
    
    
    if x.shape[0] < max_msa:
        pad = np.zeros(shape = [max_msa - x.shape[0], max_p], dtype=int)
        x = np.concatenate([x, pad])
        mask_x = np.concatenate([mask_x, pad])
    else:
        x = x[:max_msa, :]
        mask_x = mask_x[:max_msa, :]
    return x, mask_x

#msa_mask缺失部分的mask


def contact_pad(x_pre, max_p):
    mask_x  = np.ones_like(x_pre)
    if x_pre.shape[1] < max_p :
        x = np.array([np.pad(i,(0, max_p -  x_pre.shape[1]), 'constant', constant_values = 0) for i in x_pre])
        mask_x = np.array([np.pad(i,(0, max_p -  x_pre.shape[1]), 'constant', constant_values = 0) for i in mask_x])
    else:
        x = x_pre[:, :max_p]
        mask_x = mask_x[:, :max_p]
    if x_pre.shape[0] < max_p:
        pad = np.zeros(shape = [max_p - x_pre.shape[0], max_p], dtype=np.float32)
        x = np.concatenate([x, pad])
        mask_x = np.concatenate([mask_x, pad])
    else:
        x = x[:max_p, :]
        mask_x = mask_x[:max_p, :]
    return x, mask_x





#path = f"{this_dir}/human_ours/original/data_final.txt"

def load_protein_2d(train):
    AA = pd.Series(train['Target Sequence'].unique()).apply(read_contact)
    AA_dict = dict(zip(train['Target Sequence'].unique(), AA))
    train['contact'] = [AA_dict[i][0] for i in train['Target Sequence']]
    train['score'] = [AA_dict[i][1] for i in train['Target Sequence']]
    train['MSA'] = [AA_dict[i][2] for i in train['Target Sequence']]
    return train

def read_contact(target):
    max_p = 768
    max_p_2d =  768
    max_msa = 50
    
    contact_map=pd.read_csv(f"{this_dir}/human_ours/original/data_final.txt")
    contact_root = f"{this_dir}//human_ours/contactMap/"
    msa_root = f"{this_dir}//human_ours/msa_uniref_matrix/"
    contact_map=dict(zip(contact_map['seq'], contact_map['pdb_id']))
    
    contact_name = contact_root + contact_map[target]  +'.txt'
    msa_matrix_name = msa_root + contact_map[target] +'.fasta'
    contact_file = open(contact_name, 'r')
    content = contact_file.read() 
    contact_file.close()
    rowlist = content.splitlines()
    contact_pre = np.array([row.split() for row in rowlist[2:]]).astype(np.float32)
    msa_file =open(msa_matrix_name, 'r')
    msa_content = msa_file.read()
    msa_file.close()
    msa_rowlist = msa_content.splitlines()
    score = np.array(msa_rowlist[0].split()).astype(np.float32)
    pre_msa = np.array([list(row) for row in msa_rowlist[1:]])
    if score.shape[0] != pre_msa.shape[1]:
        print('wrong')

    if len(score)< max_p :
        score = np.pad(score, (0, max_p-len(score)), 'constant', constant_values =0.0)
    else:
        score = score[:max_p]

    contact, contact_mask = contact_pad(contact_pre, max_p_2d)
    msa, msa_mask = msa2emb(pre_msa, max_p_2d, max_msa)
    return (contact, contact_mask), score , (msa, msa_mask)




def data_process_nosplit(X_drug = None,  X_target = None, y=None, data_mode=None, drug_encoding_1d=None, drug_encoding_2d=None, target_encoding=None):

    df_data = pd.DataFrame(zip(X_drug,  X_target, y))
    df_data.rename(columns={0:'SMILES',
                                1:'Target Sequence',
                                2: 'Label'}, 
                                inplace=True)
    print('in total: ' + str(len(df_data)) + ' drug-target pairs')
    if data_mode == 'mode_1':
        print('Drug Target Interaction Prediction Mode_1...') 
    elif data_mode == 'mode_2':
        print('Drug Target Interaction Prediction Mode_2...')
    elif data_mode == 'mode_3':
        print('Drug Target Interaction Prediction Mode_3...')
    elif data_mode == 'mode_4':
        print('Drug Target Interaction Prediction Mode_4...')
    elif data_mode == 'mode_5':
        print('Drug Target Interaction Prediction Mode_5...')
    elif data_mode == 'mode_6':
        print('Drug Target Interaction Prediction Mode_6...')



    if data_mode == 'mode_1' or data_mode == 'mode_2':    
        print('start')    
        df_data = load_protein_2d(df_data)
        df_data = encode_drug(df_data, drug_encoding_1d, save_column_name = 'drug_encoding_1d')
        df_data = encode_drug(df_data, drug_encoding_2d, save_column_name = 'drug_encoding_2d')
        df_data = encode_protein(df_data, target_encoding)
    elif data_mode == 'mode_4' or data_mode == 'mode_3':
        df_data = encode_drug(df_data, drug_encoding_1d, save_column_name = 'drug_encoding_1d')
        df_data = encode_protein(df_data, target_encoding)
    elif data_mode == 'mode_6' or data_mode == 'mode_5':
        df_data = encode_drug(df_data, drug_encoding_2d, save_column_name = 'drug_encoding_2d')
        df_data = encode_protein(df_data, target_encoding)
    if data_mode == 'mode_3' or data_mode == 'mode_5':
        df_data = load_protein_2d(df_data)
    '''
    elif DDI_flag:
        df_data = encode_drug(df_data, drug_encoding, 'SMILES 1', 'drug_encoding_1')
        df_data = encode_drug(df_data, drug_encoding, 'SMILES 2', 'drug_encoding_2')
    elif PPI_flag:
        df_data = encode_protein(df_data, target_encoding, 'Target Sequence 1', 'target_encoding_1')
        df_data = encode_protein(df_data, target_encoding, 'Target Sequence 2', 'target_encoding_2')
    elif property_prediction_flag:
        df_data = encode_drug(df_data, drug_encoding)
    elif function_prediction_flag:
        df_data = encode_protein(df_data, target_encoding)
    '''

    return df_data.reset_index(drop=True)


class data_process_loader_new(data.Dataset):

    def __init__(self, list_IDs, labels, df, data_mode, **config):
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df
        self.config = config
        self.mode = data_mode
        if self.config['drug_encoding_2d'] in ['DGL_GCN', 'DGL_NeuralFP']:
            from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
            self.node_featurizer = CanonicalAtomFeaturizer()
            self.edge_featurizer = CanonicalBondFeaturizer(self_loop = True)
            from functools import partial
            self.fc = partial(smiles_to_bigraph, add_self_loop=True)


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        if self.mode == 'mode_1':
            index = self.list_IDs[index]
            v_d_1d = self.df.iloc[index]['drug_encoding_1d']   
            if self.config['drug_encoding_1d'] == 'CNN' or self.config['drug_encoding_1d'] == 'CNN_RNN':
                v_d_1d = drug_2_embed(v_d_1d)
            v_d_2d = self.df.iloc[index]['drug_encoding_2d']  
            if self.config['drug_encoding_2d'] in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
                v_d_2d = self.fc(smiles = v_d_2d, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer)     
            v_p_1d = self.df.iloc[index]['target_encoding']
            if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
                v_p_1d = protein_2_embed(v_p_1d)
            v_p_2d = self.df.iloc[index]['contact']
            MSA = self.df.iloc[index]['MSA']
            score = self.df.iloc[index]['score'] 
            y = self.labels[index]
            return v_d_2d, v_d_1d, v_p_1d, v_p_2d, MSA, score, y
        elif self.mode == 'mode_2':
            index = self.list_IDs[index]
            v_d_1d = self.df.iloc[index]['drug_encoding_1d']   
            if self.config['drug_encoding_1d'] == 'CNN' or self.config['drug_encoding_1d'] == 'CNN_RNN':
               v_d_1d = drug_2_embed(v_d_1d)
            v_d_2d = self.df.iloc[index]['drug_encoding_2d']  
            if self.config['drug_encoding_2d'] in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
                v_d_2d = self.fc(smiles = v_d_2d, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer)     
            v_p_1d = self.df.iloc[index]['target_encoding']
            if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
                v_p_1d = protein_2_embed(v_p_1d)
            v_p_2d = self.df.iloc[index]['contact']
            MSA = self.df.iloc[index]['MSA'] 
            y = self.labels[index]
            return v_d_2d, v_d_1d, v_p_1d, v_p_2d, MSA, y
        elif self.mode == 'mode_3':
            index = self.list_IDs[index]
            v_d = self.df.iloc[index]['drug_encoding_1d']   
            if self.config['drug_encoding_1d'] == 'CNN' or self.config['drug_encoding_1d'] == 'CNN_RNN':
                v_d = drug_2_embed(v_d)     
            v_p = self.df.iloc[index]['target_encoding']
            if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
                v_p = protein_2_embed(v_p)
            score = self.df.iloc[index]['score'] 
            y = self.labels[index]
            MSA = self.df.iloc[index]['MSA'] 
            return v_d, v_p, MSA, score, y
        elif self.mode == 'mode_4':
            index = self.list_IDs[index]
            v_d = self.df.iloc[index]['drug_encoding_1d']   
            if self.config['drug_encoding_1d'] == 'CNN' or self.config['drug_encoding_1d'] == 'CNN_RNN':
                v_d = drug_2_embed(v_d)     
            v_p = self.df.iloc[index]['target_encoding']
            if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
                v_p = protein_2_embed(v_p)
            y = self.labels[index]
            return v_d, v_p, y
        elif self.mode == 'mode_5':
            index = self.list_IDs[index]
            v_d = self.df.iloc[index]['drug_encoding_2d'] 
            if self.config['drug_encoding_2d'] == 'CNN' or self.config['drug_encoding_2d'] == 'CNN_RNN':
                v_d = drug_2_embed(v_d)
            elif self.config['drug_encoding_2d'] in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
                v_d = self.fc(smiles = v_d, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer)
       
            v_p = self.df.iloc[index]['target_encoding']
            if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
                v_p = protein_2_embed(v_p)
            score = self.df.iloc[index]['score'] 
            y = self.labels[index]
            MSA = self.df.iloc[index]['MSA'] 
            return v_d, v_p, MSA, score, y
        elif self.mode == 'mode_6':
            index = self.list_IDs[index]
            v_d = self.df.iloc[index]['drug_encoding_2d'] 
            if self.config['drug_encoding_2d'] == 'CNN' or self.config['drug_encoding_2d'] == 'CNN_RNN':
                v_d = drug_2_embed(v_d)
            elif self.config['drug_encoding_2d'] in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
                v_d = self.fc(smiles = v_d, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer)
       
            v_p = self.df.iloc[index]['target_encoding']
            if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
                v_p = protein_2_embed(v_p)
            y = self.labels[index]
            return v_d, v_p, y

def split_train_valid_new(df_data, split_method = 'random', r = 0.2, random_seed = 1234):
    if split_method == 'random':
        random_list = list(range(len(df_data)))
        np.random.seed(random_seed)
        np.random.shuffle(random_list)
        n = int((1-r) * len(df_data))
        train, val_test = df_data.iloc[random_list[:n],], df_data.iloc[random_list[n:],]
        n = int(0.5 * len(val_test))
        random_list = list(range(len(val_test)))
        np.random.shuffle(random_list)
        val, test = val_test.iloc[random_list[:n],], val_test.iloc[random_list[n:],]
        # val_test = df_data.sample(frac = r, replace = False, random_state = random_seed)
        # train = df_data[~df_data.index.isin(val_test.index)]

        # test = val_test.sample(frac = 0.5, replace = False, random_state = random_seed)
        # val =  val_test[~ val_test.index.isin(test.index)]

        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    else:
        return 
'''
def split_train_valid_new(df_data, split_method = 'random', r = 0.2, random_seed = 1):
    if split_method == 'random':
        val_test = df_data.sample(frac = r, replace = False, random_state = random_seed)
        train = df_data[~df_data.index.isin(val_test.index)]

        test = val_test.sample(frac = 0.5, replace = False, random_state = random_seed)
        val =  val_test[~ val_test.index.isin(test.index)]

        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    else:
        return 
'''
def split_train_valid(df_data, split_method = 'random', r = 0.2, random_seed = 1):
    if split_method == 'random':
        val = df_data.sample(frac = r, replace = False, random_state = random_seed)
        train = df_data[~df_data.index.isin(val.index)]


        #val_test = df_data.sample(frac = r, replace = False, random_state = random_seed)
        #train = df_data[~df_data.index.isin(val_test.index)]

        #test = val_test.sample(frac = 0.5, replace = False, random_state = random_seed)
        #val =  val_test[~ val_test.index.isin(test.index)]
        return train.reset_index(drop=True), val.reset_index(drop=True)
        #return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    else:
        return 



def getROCE(predList,targetList,roceRate):
	p = sum(targetList)
	n = len(targetList) - p
	predList = [[index,x] for index,x in enumerate(predList)]
	predList = sorted(predList,key = lambda x:x[1],reverse = True)
	tp1 = 0
	fp1 = 0
	maxIndexs = []
	for x in predList:
		if(targetList[x[0]] == 1):
			tp1 += 1
		else:
			fp1 += 1
			if(fp1>((roceRate*n)/100)):
				break
	roce = (tp1*n)/(p*fp1)
	return roce






'''
train = pd.read_csv(path)
train = data_process_nosplit(X_drug=train['mol'],X_target=train['seq'],y=train['interaction'],data_mode='mode_1',drug_encoding_1d='Transformer',drug_encoding_2d='MPNN',target_encoding='Transformer')
print(train['score'].head())

contact_name = f"{this_dir}//human_ours/contactMap/3shqA.txt"
contact_file = open(contact_name, 'r')
content = contact_file.read() 
contact_file.close()
rowlist = content.splitlines()
print(len(rowlist))
contact_pre = np.array([row.split() for row in rowlist[2:]]).astype(np.float32)
'''