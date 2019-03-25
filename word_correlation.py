#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# -----------------------------------------------------------------------------
# Name:       word_correlation.py
# Purpose:    To find conditional probability of co-occurrence of specifically
#			  marked words in the text
# -----------------------------------------------------------------------------
'''
This module receives path of preprocessed text file (default ./Preprocessed.txt),
file path for output files, then parses input text and counts conditional probability
of different word classes, namely classes, marked with  gene_*_gene, chem_*_chem and 
disease_*_disease. 
The output is delivered in 6 csv files named like P_*a_*b.csv, 
which means P(A|B). For example from P_d_g.csv we can find probability of having 
disease  if specific gene is present.
COMMAND LINE ARGUMENTS:
-i : path of the input file, default './Preprocessed.txt';
-o : path of the output files, default './'. Note, that you can have custom prefixes 
	 for files, for example, if you pass '-o ./nmda_', resulting files will be named
	 nmda_P_c_d.csv, nmda_P_g_c.csv, etc.
-fd : Try to fix dataset. Removes SOME errors, saves dataset as '%initial_name%_fxd.txt'
	  
DEPENDENCIES:
Numpy, Pandas.
Tested with Python 2.7.9
'''

import numpy as np
import pandas as pd
import re,argparse

def calculate_probs(co_occ_matrix, names):
	'''
	Receives array of co-occurrences, calculates conditional probabilities,
	as described in http://www.d.umn.edu/~tpederse/Pubs/cicling2003-2.pdf
	'''

	# Probability of gene if chemical	
	# pick name lists
	n1, n2 = names['genes'], names['chems']
	# Preallocate matrix
	P_g_c_matrix = np.zeros((len(n1),len(n2)))*np.NaN
	# Calculate conditional probability like N(a and b) / N(b)
	for ng, g in enumerate(names['genes']):
		for nc, c in enumerate(names['chems']):
			P_g_c = np.sum(co_occ_matrix[ng,nc,:]) / np.sum(co_occ_matrix[:,nc,:])
			# Put result into the matrix
			P_g_c_matrix[ng,nc] = P_g_c
	# Convert array to Pandas Data Frame for saving to csv
	P_g_c_matrix = pd.DataFrame(data = P_g_c_matrix,
								index = n1,
								columns = n2)

	# Probability of chemical if gene
	P_c_g_matrix = np.zeros((len(n2),len(n1)))*np.NaN
	for ng, g in enumerate(names['genes']):
		for nc, c in enumerate(names['chems']):
			P_c_g = np.sum(co_occ_matrix[ng,nc,:]) / np.sum(co_occ_matrix[ng,:,:])
			P_c_g_matrix[nc,ng] = P_c_g
	P_c_g_matrix = pd.DataFrame(data = P_c_g_matrix,
								index = n2,
								columns = n1)

	# Probability of gene if disease
	n1, n2 = names['genes'], names['diseases']
	P_g_d_matrix = np.zeros((len(n1),len(n2)))*np.NaN
	for ng, g in enumerate(names['genes']):
		for nd, d in enumerate(names['diseases']):
			P_g_d = np.sum(co_occ_matrix[ng,:,nd]) / np.sum(co_occ_matrix[:,:,nd])
			P_g_d_matrix[ng,nd] = P_g_d
	P_g_d_matrix = pd.DataFrame(data = P_g_d_matrix,
								index = n1,
								columns = n2)

	# Probability of disease if gene
	P_d_g_matrix = np.zeros((len(names['diseases']),len(names['genes'])))*np.NaN
	for ng, g in enumerate(names['genes']):
		for nd, d in enumerate(names['diseases']):
			P_d_g = np.sum(co_occ_matrix[ng,:,nd]) / np.sum(co_occ_matrix[ng,:,:])
			P_d_g_matrix[nd,ng] = P_d_g
	P_d_g_matrix = pd.DataFrame(data = P_d_g_matrix,
						index = n2,
						columns = n1)

	# Probability of disease if chemical
	n1, n2 = names['diseases'], names['chems']
	P_d_c_matrix = np.zeros((len(n1),len(n2)))*np.NaN
	for nd, d in enumerate(names['diseases']):
		for nc, c in enumerate(names['chems']):
			P_d_c = np.sum(co_occ_matrix[:,nc,nd]) / np.sum(co_occ_matrix[:,nc,:])			
			P_d_c_matrix[nd,nc] = P_d_c
	P_d_c_matrix = pd.DataFrame(data = P_d_c_matrix,
								index = n1,
								columns = n2)

	# Probability of chemical if disease
	P_c_d_matrix = np.zeros((len(names['chems']),len(names['diseases'])))*np.NaN
	for nd, d in enumerate(names['diseases']):
		for nc, c in enumerate(names['chems']):
			P_c_d = np.sum(co_occ_matrix[:,nc,nd]) / np.sum(co_occ_matrix[:,:,nd])			
			P_c_d_matrix[nc,nd] = P_c_d
	P_c_d_matrix = pd.DataFrame(data = P_c_d_matrix,
								index = n2,
								columns = n1)

	return{'P_g_d':P_g_d_matrix,'P_d_g':P_d_g_matrix, 'P_d_c':P_d_c_matrix, 
		   'P_c_d':P_c_d_matrix, 'P_g_c':P_g_c_matrix, 'P_c_g':P_c_g_matrix}

def  co_occurences(file_path, single_occurency = True):
	''' 
	Receives file path, returns 3d - matrix of co-occurrences
	'''
	# read textfile
	with open(file_path) as Text:
		strings = Text.readlines()
	
	# construct regular expressions to match stuff
	chem_regex = r'(?<=chem_)\w*(?=\_chem\b)'
	gene_regex = r'(?<=gene_)\w*(?=\_gene\b)'
	disease_regex = r'(?<=disease_)\w*(?=\_disease\b)'
	reg_g = re.compile(gene_regex)
	reg_c = re.compile(chem_regex)
	reg_d = re.compile(disease_regex)

	# find all items in text to make Numpy array of suitable size. This approach
	# results in regexing text two times, so it is subject to fix if data gets 
	# too big
	source_text = ''.join(strings)
	t_genes = list(set(reg_g.findall(source_text))) # genes go first, so no ['None' are needed]
	t_chems = list(set(reg_c.findall(source_text))) + ['None']
	t_diseases = list(set(reg_d.findall(source_text))) + ['None']

	# preallocate co-occurences matrix
	co_occ_matrix =  np.zeros((len(t_genes), len(t_chems), len(t_diseases)))
	
	# search for genes, chemicals and diseases in string,
	# if item of some type is not present in the string, make it ['None'],
	# in order to not break next step
	for n, a in enumerate(strings):

		genes_string = list(set(reg_g.findall(a)))
		genes_string =  genes_string if len(genes_string) >0 else ['None']
		
		chems_string = list(set(reg_c.findall(a)))
		chems_string =  chems_string if len(chems_string) >0 else ['None']

		diseases_string = list(set(reg_d.findall(a)))
		diseases_string =  diseases_string if len(diseases_string) >0 else ['None']

		# increment values in co-occurrences matrix for all items involved
		for g in genes_string:
			for c in chems_string:
				for d in diseases_string:
					co_occ_matrix[t_genes.index(g), t_chems.index(c), t_diseases.index(d)] += 1
	
	return co_occ_matrix, {'genes':t_genes, 'chems':t_chems, 'diseases': t_diseases}

def fix_aliaces(file_path):
	synonims1 = ['nmdars', 'n_methyl_d_aspartate_receptor', 'nmda_receptor', 'nmda_receptor_1_subunit', 
				'n_methyl_d_aspartate_receptor', 'nr3a', 'glurepsilon1_nr2a', 'glurepsilon2_nr2b', 
				'glurepsilon', 'glurepsilon_1', 'nr2', 'glun2d']

	# open source file
	with open(file_path) as dataset:
		text = dataset.read()
	# fix mess with NMDA receptors
	for string in synonims1:
		text = text.replace('gene_%s_gene'%string, 'gene_ONE_OF_NMDA_RECEPTOR_GENES_gene')
	# label mk-801 as chem
	text = text.replace(' mk-801 ', ' chem_Dizocilpine_chem ')
	text = text.replace(' 3h]mk-801 ', ' chem_Dizocilpine_H3_chem ')
	# remove false labeling brain areas as a diseases
	text = text.replace( 'ventral_tegmental_area', 'None')
	text = text.replace('dorsolateral_prefrontal_cortex', 'none')

	# write fixed file
	with open(file_path+'_fxd.txt', 'w') as fxdfile:	
		fxdfile.write(text)



def dump_to_csv(data, output_file_path_prefix = './'):
	'''
	Save data to 6 files, base names from data.keys()
	'''
	for key in data.keys():
		path = '%s%s.csv' %(output_file_path_prefix, key)
		data[key].to_csv(path, float_format = '%.3f')
		pass


if __name__ == '__main__':

	# add command line arguments
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-i', action='store', dest='source_file_path',
						help='Input file path', default = './Preprocessed.txt')
	parser.add_argument('-o', action='store', dest='output_file_path_prefix',
						help='Output files path and prefix', default = './')
	parser.add_argument('-fd', action='store_const', const = True, dest='fix_dataset',
						help='Try to fix dataset', default = False)
	args =  vars(parser.parse_args())

	if args['fix_dataset']:
		fix_aliaces(args['source_file_path'])
		exit()
	# Do the job
	co_occrency_table, item_names = co_occurences(args['source_file_path'])
	condprob_dict = calculate_probs(co_occrency_table, item_names)
	dump_to_csv(condprob_dict, output_file_path_prefix = args['output_file_path_prefix'])

