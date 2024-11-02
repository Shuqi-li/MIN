import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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





#path = f"{this_dir}/DUDE/dataPre/DUDE-foldTrain1"

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
    contact_map=pd.read_csv(f"{this_dir}/DUDE/dataPre/DUDE-contactDict",sep=':',header=None,index_col=None,names=['target','file_name'])
    contact_root = f"{this_dir}/DUDE/contactMap/"
    msa_root = f"{this_dir}/DUDE/msa_matrix/"
    contact_map=dict(zip(contact_map['target'], contact_map['file_name']))
    contact_name = contact_root + contact_map[target]
    msa_matrix_name = msa_root + contact_map[target].rstrip('_full') +'.fasta'
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
        df_data = encode_drug(df_data, drug_encoding_1d, save_column_name = 'drug_encoding_1d')
        df_data = encode_drug(df_data, drug_encoding_2d, save_column_name = 'drug_encoding_2d')
        df_data = encode_protein(df_data, target_encoding)
        df_data = load_protein_2d(df_data)
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
            if v_d_2d==None:
                print(self.df.iloc[index]['drug_encoding_2d']  )
            return v_d_2d, v_d_1d, v_p_1d, v_p_2d, MSA, score, y
        elif self.mode == 'mode_2':
            index = self.list_IDs[index]
            v_d_1d = self.df.iloc[index]['drug_encoding_1d']   
            if self.config['drug_encoding_1d'] == 'CNN' or self.config['drug_encoding_1d'] == 'CNN_RNN':
                v_d = drug_2_embed(v_d)
            v_d_2d = self.df.iloc[index]['drug_encoding_2d']  
            if self.config['drug_encoding_2d'] in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
                v_d_2d = self.fc(smiles = v_d, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer)     
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
            MSA = self.df.iloc[index]['MSA']
            y = self.labels[index]
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
            MSA = self.df.iloc[index]['MSA']
            y = self.labels[index]
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



def split_train_valid_new(df_data, split_method = 'random', r = 0.2, random_seed = 1):
    if split_method == 'random':
        #val = df_data.sample(frac = r, replace = False, random_state = random_seed)
        #train = df_data[~df_data.index.isin(val.index)]


        val_test = df_data.sample(frac = r, replace = False, random_state = random_seed)
        train = df_data[~df_data.index.isin(val_test.index)]

        test = val_test.sample(frac = 0.5, replace = False, random_state = random_seed)
        val =  val_test[~ val_test.index.isin(test.index)]
        #return train.reset_index(drop=True), val.reset_index(drop=True)
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    else:
        return 


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





 
def readLinesStrip(lines):
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    return lines
def getProteinSeq(path,contactMapName):
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    return seq
def getProtein(path,contactMapName,contactMap = True):
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    if(contactMap):
        contactMap = []
        for i in range(2,len(proteins)):
            contactMap.append(proteins[i])
        return seq,contactMap
    else:
        return seq

def getTrainDataSet(trainFoldPath):
    with open(trainFoldPath, 'r') as f:
        trainCpi_list = f.read().strip().split('\n')
    trainDataSet = [cpi.strip().split() for cpi in trainCpi_list]
    return trainDataSet#[[smiles, sequence, interaction],.....]
def getTestProteinList(testFoldPath):
    testProteinList = readLinesStrip(open(testFoldPath).readlines())[0].split()
    return testProteinList#['kpcb_2i0eA_full','fabp4_2nnqA_full',....]
def getSeqContactDict(contactPath,contactDictPath):# make a seq-contactMap dict 
    contactDict = open(contactDictPath).readlines()
    seqContactDict = {}
    for data in contactDict:
        _,contactMapName = data.strip().split(':')
        seq,contactMap = getProtein(contactPath,contactMapName)
        contactmap_np = [list(map(float, x.strip(' ').split(' '))) for x in contactMap]
        feature2D = np.expand_dims(contactmap_np, axis=0)
        feature2D = torch.FloatTensor(feature2D)    
        seqContactDict[seq] = feature2D
    return seqContactDict
def getLetters(path):
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars
def getDataDict(testProteinList,activePath,decoyPath,contactPath):
    dataDict = {}
    for x in testProteinList:#'xiap_2jk7A_full'
        xData = []
        protein = x.split('_')[0]
        proteinActPath = activePath+"/"+protein+"_actives_final.ism"
        proteinDecPath = decoyPath+"/"+protein+"_decoys_final.ism"
        act = open(proteinActPath,'r').readlines()
        dec = open(proteinDecPath,'r').readlines()
        actives = [[x.split(' ')[0],1] for x in act] ######
        decoys = [[x.split(' ')[0],0] for x in dec]# test
        seq = getProtein(contactPath,x,contactMap = False)
        for i in range(len(actives)):
            xData.append([actives[i][0],seq,actives[i][1]])
        for i in range(len(decoys)):
            xData.append([decoys[i][0],seq,decoys[i][1]])
        dataDict[x] = xData
    return dataDict