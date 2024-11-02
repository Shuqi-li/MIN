import argparse
import numpy as np
import pandas as pd
'''
parser = argparse.ArgumentParser()
parser.add_argument("--MPNN_hidden_size", default=50, type=int)
parser.add_argument("--MPNN_depth", default=3, type=int)
###transformer
parser.add_argument("--transformer_emb_size_drug", default = 128, type=int)
parser.add_argument("--transformer_intermediate_size_drug", default = 512, type=int)
parser.add_argument("--transformer_num_attention_heads_drug", default = 8, type=int)
parser.add_argument("--transformer_n_layer_drug", default = 8, type=int)
parser.add_argument("--transformer_emb_size_target", default = [64, 14], type=int)
parser.add_argument("--transformer_intermediate_size_target", default = 256, type=int)
configs = parser.parse_args()




contact_map=pd.read_csv('../drug/DUDE/dataPre/DUDE-contactDict',sep=':',header=None,index_col=None,names=['seq','file_name'])
train1=pd.read_csv('../drug/DUDE/dataPre/DUDE-foldTrain1',sep=' ',header=None,index_col=None,names=['mol','seq','interaction'])
contact_root = '../drug/DUDE/contactMap/'
msa_root = '../drug/DUDE/msa_matrix/'
for i in range(len(train1)):
    target_name = contact_map[contact_map['seq']==train1.loc[0,'seq']]['file_name'].values[0]
    contact_name = contact_root + target_name
    msa_matrix_name = msa_root + target_name.strip('_full') +'.fasta'
    #read contact file 
    contact_file = open(contact_name, 'r')
    content = contact_file.read() 
    contact_file.close()
    rowlist = content.splitlines()

    contact = np.array([row.split() for row in rowlist[2:]])
    #read msa file
    msa_file =open(msa_matrix_name, 'r')
    msa_content = msa_file.read()
    msa_file.close()
    msa_rowlist = msa_content.splitlines()
    score = np.array(msa_rowlist[0].split())
    msa = np.array([list(row) for row in msa_rowlist[1:]])
    if score.shape[0] != msa.shape[1]:
        print('wrong:'+ i)
    train1.iloc(i, ['score']) = score

import pathlib
from subword_nmt.apply_bpe import BPE
import codecs
this_dir = str(pathlib.Path(__file__).parent.absolute())


# ESPF encoding
vocab_path = f"{this_dir}/ESPF/protein_codes_uniprot_2000.txt"
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
#sub_csv = pd.read_csv(dataFolder + '/subword_units_map_protein.csv')
sub_csv = pd.read_csv(f"{this_dir}/ESPF/subword_units_map_uniprot_2000.csv")

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))
words2idx_p.update({"-": len(idx2word_p)})

def msa2emb(x_msa):
    max_p = 545
    max_msa = 2000
    msa_init = np.array([[words2idx_p[i] for i in j] for j in x_msa])
    mask_x  = np.ones_like(msa_init)
    if msa_init.shape[1] < max_p :
        x = np.array([np.pad(i,(0, max_p -  msa_init.shape[1]), 'constant', constant_values = 0) for i in msa_init])
        mask_x = np.array([np.pad(i,(0, max_p -  msa_init.shape[1]), 'constant', constant_values = 0) for i in mask_x])
    else:
        x = msa_init[:, :max_p]
        mask_x = mask_x[:, :max_p]
    if msa_init.shape[0] < max_msa:
        pad = np.zeros(shape = [max_msa - msa_init.shape[0], max_p], dtype=int)
        x = np.concatenate([x, pad])
        mask_x = np.concatenate([mask_x, pad])
    else:
        x = msa_init[:max_msa, :]
        mask_x = mask_x[:max_msa, :]
    return x, mask_x




path = '../drug/DUDE/dataPre/DUDE-foldTrain1'

def load_2d(path):
    train=pd.read_csv(path,sep=' ',header=None,index_col=None,names=['drug','target','label'], nrows=10)
    AA = pd.Series(train['target'].unique()).apply(read_contact)
    AA_dict = dict(zip(train['target'].unique(), AA))
    train['target 2d'] = [AA_dict[i][0] for i in train['target']]
    train['score'] = [AA_dict[i][1] for i in train['target']]
    train['MSA'] = [AA_dict[i][2] for i in train['target']]
    return train

def read_contact(target):
    contact_map=pd.read_csv('../drug/DUDE/dataPre/DUDE-contactDict',sep=':',header=None,index_col=None,names=['target','file_name'])
    contact_root = '../drug/DUDE/contactMap/'
    msa_root = '../drug/DUDE/msa_matrix/'
    contact_map=dict(zip(contact_map['target'], contact_map['file_name']))
    contact_name = contact_root + contact_map[target]
    msa_matrix_name = msa_root + contact_map[target].rstrip('_full') +'.fasta'
    contact_file = open(contact_name, 'r')
    content = contact_file.read() 
    contact_file.close()
    rowlist = content.splitlines()
    contact = np.array([row.split() for row in rowlist[2:]]).astype(np.float32)
    
    msa_file =open(msa_matrix_name, 'r')
    msa_content = msa_file.read()
    msa_file.close()
    msa_rowlist = msa_content.splitlines()
    score = np.array(msa_rowlist[0].split())
    pre_msa = np.array([list(row) for row in msa_rowlist[1:]])
    if score.shape[0] != pre_msa.shape[1]:
        print('wrong:'+ i)
    msa, msa_mask = msa2emb(pre_msa)
    return contact, score, (msa, msa_mask)

train = load_2d(path)



import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange

# data


# models

from alphafold2_pytorch import Alphafold2
import alphafold2_pytorch.constants as constants

from alphafold2_pytorch.utils import *
import torch
from alphafold2_pytorch import Alphafold2

model = Alphafold2(
    dim = 256,
    depth = 3,
    heads = 4,
    dim_head = 64
).cuda()

seq = torch.randint(0, 21, (1, 256)).cuda()      # AA length of 128
msa = torch.randint(0, 21, (1, 100, 256)).cuda()   # MSA doesn't have to be the same length as primary sequence
mask = torch.ones_like(seq).bool().cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

distogram, m = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask
)
print(len(distogram))

import argparse
import random
import numpy as np 
import torch
import run_demo as models
from utils import *
import os

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()


    # Train parameters
    #parser.add_argument("--cuda_id", default='1', type=str)
    parser.add_argument("--binary", default=True, type=bool)

    parser.add_argument("--seed", default=580, type=int)
    parser.add_argument("--batch_size", default=32, type=int) #256
    parser.add_argument("--LR", default=1e-4, type=float)
    parser.add_argument("--decay", default=0.0, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--train_epoch", default=40, type=int)
    parser.add_argument("--test_every_X_epoch", default=20, type=int)  

    parser.add_argument("--result_folder", default="./result/", type=str) 



    
    parser.add_argument("--score_encoder", default=False, type=bool)
    parser.add_argument("--score_threshold", default=0.01, type=float)
    parser.add_argument("--target_global_1d", default=True, type=bool)
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
    parser.add_argument("--drug_encoding_2d", default='MPNN', type=str) #'MPNN'
    parser.add_argument("--target_encoding", default='Transformer', type=str)

    ###transformer
    parser.add_argument("--input_dim_drug", default=2586, type=int)
    #parser.add_argument("--input_dim_drug_2d", default=1024, type=int)
    parser.add_argument("--input_dim_protein", default = 4114, type=int)
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
	

    ###MPNN 
    parser.add_argument("--MPNN_hidden_size", default=128, type=int)
    parser.add_argument("--MPNN_depth", default=3, type=int)
    

    ## alphafold
    
    parser.add_argument("--max_p_2d", default = 128, type=int)
    parser.add_argument("--max_msa", default = 20, type=int)
    parser.add_argument("--AlphaFold_seq_len", default=128, type=int)
    parser.add_argument("--AlphaFold_depth", default=2, type=int)
    parser.add_argument("--AlphaFold_dim", default=128, type=int)
    parser.add_argument("--AlphaFold_heads", default=4, type=int)    
    parser.add_argument("--AlphaFold_dim_head", default=32, type=int)
    parser.add_argument("--AlphaFold_attn_dropout", default=0., type=float)
    parser.add_argument("--AlphaFold_ff_dropout", default=0., type=float)

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

    train1=pd.read_csv('../drug/DUDE/dataPre/DUDE-foldTrain1',sep=' ',header=None,index_col=None,names=['drug','target','label'])
    #train2=pd.read_csv('../drug/DUDE/dataPre/DUDE-foldTrain2',sep=' ',header=None,index_col=None,names=['drug','target','label'], nrows=1500)
    train3=pd.read_csv('../drug/DUDE/dataPre/DUDE-foldTrain3',sep=' ',header=None,index_col=None,names=['drug','target','label'])
    drug_encoding_1d = 'Transformer'
    drug_encoding_2d = 'MPNN'
    target_encoding = 'Transformer'

    print('begin')


    数据加载模式
    'mode_1': drug12 + target12 + msa + score + y
    'mode_2': drug12 + target12 + msa + y
    'mode_3': drug1 + target1 + score + y
    'mode_4': drug1 + target1 + y
    'mode_5': drug2 + target1 + score + y
    'mode_6': drug2 + target1 + y

    data_mode = 'mode_4'
    df_train1= data_process_nosplit(data_mode = data_mode, X_drug =train1['drug'],  X_target =train1['target'],  y=train1['label'], drug_encoding_1d =drug_encoding_1d, drug_encoding_2d =drug_encoding_2d, target_encoding = target_encoding)

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
contact_map=pd.read_csv(f"{this_dir}/human_ours/original/data_final.txt")
print(contact_map['pdb_id'].head())
contact_map=dict(zip(contact_map['seq'], contact_map['pdb_id']))
print(contact_map['MSPLNQSAEGLPQEASNRSLNATETSEAWDPRTLQALKISLAVVLSVITLATVLSNAFVLTTILLTRKLHTPANYLIGSLATTDLLVSILVMPISIAYTITHTWNFGQILCDIWLSSDITCCTASILHLCVIALDRYWAITDALEYSKRRTAGHAATMIAIVWAISICISIPPLFWRQAKAQEEMSDCLVNTSQISYTIYSTCGAFYIPSVLLIILYGRIYRAARNRILNPPSLYGKRFTTAHLITGSAGSSLCSLNSSLHEGHSHSAGSPLFFNHVKIKLADSALERKRISAARERKATKILGIILGAFIICWLPFFVVSLVLPICRDSCWIHPALFDFFTWLGYLNSLINPIIYTVFNEEFRQAFQKIVPFRKAS'])

print(len('VDTCSLASPASVCRTKHLHLRCSVDFTRRTLTGTAALTVQSQEDNLRSLVLDTKDLTIEKVVINGQEVKYALGERQSYKGSPMEISLPIALSKNQEIVIEISFETSPKSSALQWLTPEQTSGKEHPYLFSQCQAIHCRAILPCQDTPSVKLTYTAEVSVPKELVALMSAIRDGETPDPEDPSRKIYKFIQKVPIPCYLIALVVGALESRQIGPRTLVWSEKEQVEKSAYEFSETESMLKIAEDLGGPYVWGQYDLLVLPPSFPYGGMENPCLTFVTPTLLAGDKSLSNVIAHEISHSWTGNLVTNKTWDHFWLNEGHTVYLERHICGRLFGEKFRHFNALGGWGELQNSVKTFGETHPFTKLVVDLTDIDPDVAYSSVPYEKGFALLFYLEQLLGGPEIFLGFLKAYVEKFSYKSITTDDWKDFLYSYFKDKVDVLNQVDWNAWLYSPGLPPIKPNYDMTLTNACIALSQRWITAKEDDLNSFNATDLKDLSSHQLNEFLAQTLQRAPLPLGHIKRMQEVYNFNAINNSEIRFRWLRLCIQSKWEDAIPLALKMATEQGRMKFTRPLFKDLAAFDKSHDQAVRTYQEHKASMHPVTAMLVGKDLKVD'))

x = 'VDTCSLASPASVCRTKHLHLRCSVDFTRRTLTGTAALTVQSQEDNLRSLVLDTKDLTIEKVVINGQEVKYALGERQSYKGSPMEISLPIALSKNQEIVIEISFETSPKSSALQWLTPEQTSGKEHPYLFSQCQAIHCRAILPCQDTPSVKLTYTAEVSVPKELVALMSAIRDGETPDPEDPSRKIYKFIQKVPIPCYLIALVVGALESRQIGPRTLVWSEKEQVEKSAYEFSETESMLKIAEDLGGPYVWGQYDLLVLPPSFPYGGMENPCLTFVTPTLLAGDKSLSNVIAHEISHSWTGNLVTNKTWDHFWLNEGHTVYLERHICGRLFGEKFRHFNALGGWGELQNSVKTFGETHPFTKLVVDLTDIDPDVAYSSVPYEKGFALLFYLEQLLGGPEIFLGFLKAYVEKFSYKSITTDDWKDFLYSYFKDKVDVLNQVDWNAWLYSPGLPPIKPNYDMTLTNACIALSQRWITAKEDDLNSFNATDLKDLSSHQLNEFLAQTLQRAPLPLGHIKRMQEVYNFNAINNSEIRFRWLRLCIQSKWEDAIPLALKMATEQGRMKFTRPLFKDLAAFDKSHDQAVRTYQEHKASMHPVTAMLVGKDLKVD'
vocab_list = ['<pad>', '<cls>',  '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
words2idx = dict(zip(vocab_list, range(len(vocab_list))))

i1 = [words2idx['<cls>']]+[words2idx[i] if (i in vocab_list) else words2idx['<unk>'] for i in list(x)] + [words2idx['<eos>']]


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




import pathlib
import pandas as pd
from utils import *
this_dir = str(pathlib.Path(__file__).parent.absolute())
testFoldPath = f"{this_dir}/DUDE/dataPre/DUDE-foldTest1"
testProteinList = getTestProteinList(testFoldPath)
decoy_path = f"{this_dir}/DUDE/decoy_smile"
active_path = f"{this_dir}/DUDE/active_smile"
contactPath = f"{this_dir}/DUDE/contactMap"
data_dict = getDataDict(testProteinList, active_path, decoy_path, contactPath)
for item in testProteinList:
    test_data = data_dict[item]
    test = pd.DataFrame(test_data, columns = ['drug', 'target', 'label'])

    drug_encoding_1d = 'Transformer'
    drug_encoding_2d = 'MPNN'
    target_encoding = 'Transformer'
    data_mode = 'mode_1'
    df_train1= data_process_nosplit(data_mode = data_mode, X_drug =test['drug'],  X_target =test['target'],  y=test['label'], drug_encoding_1d =drug_encoding_1d, drug_encoding_2d =drug_encoding_2d, target_encoding = target_encoding)

import pandas as pd
import pathlib
from rdkit import Chem 
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
this_dir = str(pathlib.Path(__file__).parent.absolute())


def ring_num(x):
    m = Chem.MolFromSmiles(x)
    num_ring = Chem.GetSSSR(m)
    return num_ring

def to_img(x):
    m = Chem.MolFromSmiles(x)
    Draw.MolToImage(m, size=(150,150),kekulize=True)
    Draw.MolToFile(m,f'{this_dir}/{len(x)}.png',size=(150,150))


data_file = f"{this_dir}/human_ours/original/data_final.txt"
df = pd.read_csv(data_file)
drug = pd.DataFrame(df['mol'].unique(),columns=['mol'])
drug['ring_num'] = drug['mol'].apply(ring_num)
drug.sort_values('ring_num', ascending=False, inplace=True)
tmp = drug[:10]
tmp['mol'].apply(to_img)
'''


