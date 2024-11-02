import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from load_dude import *
import pathlib
this_dir = str(pathlib.Path(__file__).parent.absolute())


if __name__ == '__main__':
	activePath = f"{this_dir}/DUDE/active_smile"
	decoyPath = f"{this_dir}/DUDE/decoy_smile"
	seqPath = f"{this_dir}/DUDE/fasta_seq"

	df_data = process_DUDE(activePath, decoyPath, seqPath)

	data_dir = f"{this_dir}/DUDE/dataPre"

	test = f"{this_dir}/DUDE/dataPre/DUDE-foldTest1"
	
	test_protein_list = [x.split('_')[0] for x in getTestProteinList(test)]
	target_dict = {'AA2AR': '3eml', 'ABL1': '2hzi', 'ACE': '3bkl', 'ACES': '1e66', 'ADA': '2e1w', 'ADA17': '2oi0', 'ADRB1': '2vt4', 'ADRB2': '3ny8', 'AKT1': '3cqw', 'AKT2': '3d0e', 'ALDR': '2hv5', 'AMPC': '1l2s', 'ANDR': '2am9', 'AOFB': '1s3b', 'BACE1': '3l5d', 'BRAF': '3d4q', 'CAH2': '1bcd', 'CASP3': '2cnk', 'CDK2': '1h00', 'COMT': '3bwm', 'CP2C9': '1r9o', 'CP3A4': '3nxu', 'CSF1R': '3krj', 'CXCR4': '3odu', 'DEF': '1lru', 'DHI1': '3frj', 'DPP4': '2i78', 'DRD3': '3pbl', 'DYR': '3nxo', 'EGFR': '2rgp', 'ESR1': '1sj0', 'ESR2': '2fsz', 'FA10': '3kl6', 'FA7': '1w7x', 'FABP4': '2nnq', 'FAK1': '3bz3', 'FGFR1': '3c4f', 'FKB1A': '1j4h', 'FNTA': '3e37', 'FPPS': '1zw5', 'GCR': '3bqd', 'GLCM': '2v3f', 'GRIA2': '3kgc', 'GRIK1': '1vso', 'HDAC2': '3max', 'HDAC8': '3f07', 'HIVINT': '3nf7', 'HIVPR': '1xl2', 'HIVRT': '3lan', 'HMDH': '3ccw', 'HS90A': '1uyg', 'HXK4': '3f9m', 'IGF1R': '2oj9', 'INHA': '2h7l', 'ITAL': '2ica', 'JAK2': '3lpb', 'KIF11': '3cjo', 'KIT': '3g0e', 'KITH': '2b8t', 'KPCB': '2i0e', 'LCK': '2of2', 'LKHA4': '3chp', 'MAPK2': '3m2w', 'MCR': '2aa2', 'MET': '3lq8', 'MK01': '2ojg', 'MK10': '2zdt', 'MK14': '2qd9', 'MMP13': '830c', 'MP2K1': '3eqh', 'NOS1': '1qw6', 'NRAM': '1b9v', 'PA2GA': '1kvo', 'PARP1': '3l3m', 'PDE5A': '1udt', 'PGH1': '2oyu', 'PGH2': '3ln1', 'PLK1': '2owb', 'PNPH': '3bgs', 'PPARA': '2p54', 'PPARD': '2znp', 'PPARG': '2gtk', 'PRGR': '3kba', 'PTN1': '2azr', 'PUR2': '1njs', 'PYGM': '1c8k', 'PYRD': '1d3g', 'RENI': '3g6z', 'ROCK1': '2etr', 'RXRA': '1mv9', 'SAHH': '1li4', 'SRC': '3el8', 'TGFR1': '3hmm', 'THB': '1q4x', 'THRB': '1ype', 'TRY1': '2ayw', 'TRYB1': '2zec', 'TYSY': '1syn', 'UROK': '1sqt', 'VGFR2': '2p2i', 'WEE1': '3biz', 'XIAP': '3hl5'}

	test_seq_list = []
	for p in test_protein_list:
		proteins = open(os.path.join(seqPath, target_dict[p.upper()])).readlines()
		if len(proteins) < 2:
			print(proteins)
			print(p)
			continue
		test_seq_list.append(getProtein_PDB(os.path.join(seqPath, target_dict[p.upper()])))
	
	print(len(test_seq_list))

	df_data1 = df_data[~df_data['Target Sequence'].isin(test_seq_list)]
	print(df_data1)

	## for 'new-protein' setting
	train_X_drug, train_X_target, train_y = df_data1.SMILES.values, df_data1['Target Sequence'].values, df_data1.Label.values

	## for 'new-protein-drug' setting
	# test_smiles_list = getDrugList(getTestProteinList(test), activePath, decoyPath)
	# df_data2 = df_data1[~df_data1['SMILES'].isin(test_smiles_list)]
	# print(df_data2)
	# train_X_drug, train_X_target, train_y = df_data2.SMILES.values, df_data2['Target Sequence'].values, df_data2.Label.values

	#drug_encoding = 'pubchem'
	#target_encoding = 'Transformer2'

	#df_train = data_process_nosplit(train_X_drug, train_X_target, train_y, drug_encoding, target_encoding)
	#train, val = split_train_valid(df_train, r=0.2, random_seed=1)

	#print(train)
	#print(val)

	#config = generate_config(drug_encoding, target_encoding)
	#config['train_epoch'] = 100
	#config['batch_size'] = 128
	#config['result_folder'] = '../dude_new_protein_train-fold1'
	#config['PolyEnc_flag'] = True
	#config['binary'] = True

	#print(config)

	#model = models.model_initialize(**config)
	#model.train(train, val, val)
