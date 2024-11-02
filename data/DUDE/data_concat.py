import numpy as np
from numpy.core.records import record
import pandas as pd



contact_map=pd.read_csv('../drug/DUDE/dataPre/DUDE-contactDict',sep=':',header=None,index_col=None,names=['seq','file_name'])
train1=pd.read_csv('../drug/DUDE/dataPre/DUDE-foldTrain1',sep=' ',header=None,index_col=None,names=['drug','target','label'])
contact_root = '../drug/DUDE/contactMap/'
msa_root = '../drug/DUDE/msa_matrix/'
contact_list = []
score_list = []
msa_list = []

for i in range(len(train1)):
    target_name = contact_map[contact_map['seq']==train1.loc[i,'target']]['file_name'].values[0]
    contact_name = contact_root + target_name
    msa_matrix_name = msa_root + target_name.rstrip('_full') +'.fasta'
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
    contact_list.append(contact)
    score_list.append(score)
    msa_list.append(msa)
train1['score'] = score_list
train1['contact'] = contact_list
train1['msa'] = msa_list

train1.to_csv('../drug/DUDE/train1')



