import os
import numpy as np
import pandas as pd
import pickle
import random
from DeepPurpose.utils import *
from utils import *


def process_DUDE(activePath=None, decoyPath=None, seqPath=None, random_seed = 1):

	target_dict = {'AA2AR': '3eml', 'ABL1': '2hzi', 'ACE': '3bkl', 'ACES': '1e66', 'ADA': '2e1w', 'ADA17': '2oi0', 'ADRB1': '2vt4', 'ADRB2': '3ny8', 'AKT1': '3cqw', 'AKT2': '3d0e', 'ALDR': '2hv5', 'AMPC': '1l2s', 'ANDR': '2am9', 'AOFB': '1s3b', 'BACE1': '3l5d', 'BRAF': '3d4q', 'CAH2': '1bcd', 'CASP3': '2cnk', 'CDK2': '1h00', 'COMT': '3bwm', 'CP2C9': '1r9o', 'CP3A4': '3nxu', 'CSF1R': '3krj', 'CXCR4': '3odu', 'DEF': '1lru', 'DHI1': '3frj', 'DPP4': '2i78', 'DRD3': '3pbl', 'DYR': '3nxo', 'EGFR': '2rgp', 'ESR1': '1sj0', 'ESR2': '2fsz', 'FA10': '3kl6', 'FA7': '1w7x', 'FABP4': '2nnq', 'FAK1': '3bz3', 'FGFR1': '3c4f', 'FKB1A': '1j4h', 'FNTA': '3e37', 'FPPS': '1zw5', 'GCR': '3bqd', 'GLCM': '2v3f', 'GRIA2': '3kgc', 'GRIK1': '1vso', 'HDAC2': '3max', 'HDAC8': '3f07', 'HIVINT': '3nf7', 'HIVPR': '1xl2', 'HIVRT': '3lan', 'HMDH': '3ccw', 'HS90A': '1uyg', 'HXK4': '3f9m', 'IGF1R': '2oj9', 'INHA': '2h7l', 'ITAL': '2ica', 'JAK2': '3lpb', 'KIF11': '3cjo', 'KIT': '3g0e', 'KITH': '2b8t', 'KPCB': '2i0e', 'LCK': '2of2', 'LKHA4': '3chp', 'MAPK2': '3m2w', 'MCR': '2aa2', 'MET': '3lq8', 'MK01': '2ojg', 'MK10': '2zdt', 'MK14': '2qd9', 'MMP13': '830c', 'MP2K1': '3eqh', 'NOS1': '1qw6', 'NRAM': '1b9v', 'PA2GA': '1kvo', 'PARP1': '3l3m', 'PDE5A': '1udt', 'PGH1': '2oyu', 'PGH2': '3ln1', 'PLK1': '2owb', 'PNPH': '3bgs', 'PPARA': '2p54', 'PPARD': '2znp', 'PPARG': '2gtk', 'PRGR': '3kba', 'PTN1': '2azr', 'PUR2': '1njs', 'PYGM': '1c8k', 'PYRD': '1d3g', 'RENI': '3g6z', 'ROCK1': '2etr', 'RXRA': '1mv9', 'SAHH': '1li4', 'SRC': '3el8', 'TGFR1': '3hmm', 'THB': '1q4x', 'THRB': '1ype', 'TRY1': '2ayw', 'TRYB1': '2zec', 'TYSY': '1syn', 'UROK': '1sqt', 'VGFR2': '2p2i', 'WEE1': '3biz', 'XIAP': '3hl5'}

	SMILES = []
	Target_seq = []
	y = []
	active_len = 0
	decoy_len = 0
	decoy_len1 = 0
	
	for i, x in enumerate(target_dict.keys()):
		protein = x.lower()
		# print('\n'+protein)

		p = open(os.path.join(seqPath, target_dict[x])).readlines()

		if len(p) < 2:
			print(x)
			print(p)
			continue
		else:
			target_seq = getProtein_PDB(os.path.join(seqPath, target_dict[x]))

			proteinActPath = os.path.join(activePath, protein+"_actives_final.ism")
			proteinDecPath = os.path.join(decoyPath, protein+"_decoys_final.ism")
			act = open(proteinActPath,'r').readlines()
			dec = open(proteinDecPath,'r').readlines()
			active_smiles = [x.split(' ')[0] for x in act]
			decoy_smiles = [x.split(' ')[0] for x in dec]
			active_len += len(active_smiles)
			decoy_len += len(decoy_smiles)

			random.seed(random_seed)
			decoy_smiles = random.sample(decoy_smiles, len(active_smiles))
			decoy_len1 += len(decoy_smiles)

			for smiles in active_smiles:
				SMILES.append(smiles)
				Target_seq.append(target_seq)
				y.append(1.0)
			for smiles in decoy_smiles:
				SMILES.append(smiles)
				Target_seq.append(target_seq)
				y.append(0.0)

	X_drug, X_target, y = np.array(SMILES), np.array(Target_seq), np.array(y)

	df_data = pd.DataFrame(zip(X_drug, X_target, y))
	df_data.rename(columns={0:'SMILES',
							1: 'Target Sequence',
							2: 'Label'}, 
							inplace=True)
	# print(df_data)
	print('in total: ' + str(active_len) + ' active drugs')         
	print('in total: ' + str(decoy_len) + ' inactive drugs')        
	print('in total: ' + str(decoy_len1) + ' inactive drugs in balanced set')   
	print('in total: ' + str(len(set(Target_seq))) + ' targets')        
	print('in total: ' + str(len(df_data)) + ' drug-target pairs') 

	# in total: 22762 active drugs
	# in total: 1408914 inactive drugs
	# in total: 22762 inactive drugs in balanced set
	# in total: 101 targets
	# in total: 45524 drug-target pairs
	# print(set(Target_seq))

	return df_data.sample(frac=1).reset_index(drop=True) 


def split_train_valid_test(df_data, split_method = 'random', r = 0.1, random_seed = 1):
	if split_method == 'random':
		test = df_data.sample(frac = r, replace = False, random_state = random_seed)
		train = df_data[~df_data.index.isin(test.index)]
	elif split_method == 'new_drug':
		pass
	elif split_method == 'new_target':
		target_list = df_data['Target Sequence'].unique().tolist()
		print('unique target sequence: ' + str(len(target_list)))   # 102
		random.seed(random_seed)
		random.shuffle(target_list)
		train_X_target = target_list[:70]
		val_X_target = target_list[70:77]
		test_X_target = target_list[-25:]
		assert target_list == train_X_target + val_X_target + test_X_target

		train = df_data[df_data['Target Sequence'].isin(train_X_target)]
		valid = df_data[df_data['Target Sequence'].isin(val_X_target)]
		test = df_data[df_data['Target Sequence'].isin(test_X_target)]
		print('Train set in total: ' + str(len(train)) + ' drug-pairs for 70 targets')
		print('Valid set in total: ' + str(len(valid)) + ' drug-pairs for 7 targets')
		print('Test set in total: ' + str(len(test)) + ' drug-pairs for 25 targets')

	elif split_method == 'new_drug_target':
		pass

	return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)

def split_train_valid(df_data, split_method = 'random', r = 0.1, random_seed = 1):
	if split_method == 'random':
		val = df_data.sample(frac = r, replace = False, random_state = random_seed)
		train = df_data[~df_data.index.isin(val.index)]

		return train.reset_index(drop=True), val.reset_index(drop=True)
	else:
		return 





'''
if __name__ == '__main__':
	activePath = os.path.join('..','data','DUDE','active_smiles')
	decoyPath = os.path.join('..','data','DUDE','decoy_smiles')	
	seqPath = os.path.join('..','data','DUDE','fasta_seq')

	df_data = process_DUDE(activePath, decoyPath, seqPath)
	print(df_data)

	target_list = df_data['Target Sequence'].unique().tolist()
	print('unique target sequence: ' + str(len(target_list)))   # 101
	with open('dude_target_esm.pkl', 'wb') as f:
		pickle.dump(target_list, f)
'''