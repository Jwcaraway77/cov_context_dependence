#script to simulate polymorphisms based on context-dependent polymorphic data for covid19

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec, cm
from scipy import stats
import seaborn as sns
from context_dependence import *
from mutation_data_funcs import convert_mutation_dataset
import re
from calc_4fold import get_gene_info
import time
from functools import reduce
from utility_funcs import *

global rows, columns, mutation_dict, gene_info, fourfold_positions
rows = ["T[X>Y]T","T[X>Y]G","T[X>Y]C","T[X>Y]A","G[X>Y]T","G[X>Y]G","G[X>Y]C","G[X>Y]A","C[X>Y]T","C[X>Y]G","C[X>Y]C","C[X>Y]A"]
rows_figs = ["U[X>Y]U","U[X>Y]G","U[X>Y]C","U[X>Y]A","G[X>Y]U","G[X>Y]G","G[X>Y]C","G[X>Y]A","C[X>Y]U","C[X>Y]G","C[X>Y]C","C[X>Y]A","A[X>Y]U","A[X>Y]G","A[X>Y]C","A[X>Y]A"]
columns = ["T>G","T>C","T>A","G>T","G>C","G>A","C>T","C>G","C>A","A>T","A>G","A>C"]
columns_figs = ["U>G","U>C","U>A","G>U","G>C","G>A","C>U","C>G","C>A","A>U","A>G","A>C"]
columns_shortened = ['T','G','C','A']
columns_shortened_figs = ['U','G','C','A']
mutation_dict = {}
gene_info = get_gene_info()
fourfold_positions = {gene:[position[0] for position in gene_info[gene][0]] for gene in gene_info.keys()}
subset_genes = {'S':[21562,25384], 'E':[26244,26472], 'M':[26522,27191], 'N':[[28273,28283],[28577,28733],[28955,29533]], 'ORF1a':[265,13468], 'ORF3a':[25392,25764], 'ORF6':[27201,27387], 'ORF7a':[27393,27755], 'ORF7b':[27759,27887], 'ORF8':[27893,28259], 'ORF10':[29557,29674]}
print({gene:len(fourfold_positions[gene]) for gene in fourfold_positions.keys()})
nsp_positions = {'nsp1':[265,805], 'nsp2':[805,2719], 'nsp3':[2719,8554], 'nsp4':[8554,10054], 'nsp5':[10054,10972], 'nsp6':[10972,11842],
                 'nsp7':[11842,12090], 'nsp8':[12091,12684], 'nsp9':[12685,13024], 'nsp10':[13024,13441], 'nsp11':[13441,13480], 'rdrp':[13441,16236],
                 'nsp13':[16234,18040], 'nsp14':[18037,19621], 'nsp15':[19618,20659], 'nsp16':[20656,21553]}
rdrp_sub_domains = {'NiRAN':[13441,14191], 'interface':[14191,14536], 'hand':[14542,16201]}
spike_sub_domains = {'NTD':[21598,22474], 'RBD':[22516,23185],'SD1_2':[23188,25186]}
overlapping_genes = {'ORF3b':[25764,26220], 'ORF9b':[28283,28577], 'ORF9c':[28733,28955]}

'''trying to improve computation time'''
def simulate_mutations_5(sim_type, global_fourfold, genes_fourfold, mut_count, contexts, gene_order, scaler=0.05):
    #sim_type: global or gene_specific; whether to use gene-specific matrices at all
    #global_fourfold: gene-averaged matrix for mutation rates
    #genes_fourfold: gene-seperated matrices for mutation rates
    #mut_count: number of mutation to place per simulation run
    #contexts: string; whether to use full contexts, ignore neighboring nucleotides, or completely naive rates
    #gene_order: list of genes used to make genes_fourfold
    #scaler: value to adjust number of mutations placed per cycle through the genome after normalizing rates

    mutation_dict = {} #stores mutations placed during simulation
    fasta_ref = pd.Series([*get_fasta()[:-1]], name='ref') #read in ref fasta
    fasta_mutated = pd.Series([*get_fasta()[:-1]], name='mutated') #copy of ref fasta to store mutations
    current_mut_count = 0 #run sim until current_mut_count >= mut_count
    iteration = 0 #store current iteration number
    a_context_labels = ['A[X>Y]N', 'A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'] #labels for unused 4fold mut contexts
    rng = np.random.default_rng() #random generator

    #convert global_fourfold for global sim
    #normalize rates and scale
    data_wrangling_timer = time.perf_counter()
    if contexts == 'full_contexts':
        if global_fourfold.shape[0] == 12:
            mut_rate_mat = pd.DataFrame(global_fourfold, index=rows, columns=columns)
            mut_rate_mat = pd.concat([mut_rate_mat, pd.DataFrame(np.zeros([4, 12]), index=a_context_labels[1:], columns=columns)], axis=0)
        elif global_fourfold.shape[0] == 16:
            mut_rate_mat = pd.DataFrame(global_fourfold, index=rows+a_context_labels[1:], columns=columns)
        mut_rate_mat = (mut_rate_mat / np.sum(mut_rate_mat.values)) * (scaler)
    else:
        mut_rate_mat = pd.DataFrame(torch.reshape(global_fourfold, [1,12]), index=['N[X>Y]N'], columns=columns)
        mut_rate_mat = (mut_rate_mat / np.sum(mut_rate_mat.values)) * (scaler / 12)

    #checking global rates
    #mut_rate_mat.to_csv('simulation_output/global_mut_rate_mat.csv')
    
    global_mut_rate_mat = mut_rate_mat.copy()

    #convert genes_fourfold for gene_specific sim
    #normalize and scale rates
    if sim_type == 'gene_specific':
        gene_info = get_gene_info()
        gene_mut_rates = {}
        for index, mut_rate_mat in enumerate(genes_fourfold):
            if contexts == 'full_contexts':
                if mut_rate_mat.shape[0] == 12:
                    mut_rate_mat = pd.DataFrame(mut_rate_mat, index=rows, columns=columns)
                    mut_rate_mat = pd.concat([mut_rate_mat, pd.DataFrame(np.zeros([4, 12]), index=a_context_labels[1:], columns=columns)], axis=0)
                elif mut_rate_mat.shape[0] == 16:
                    mut_rate_mat = pd.DataFrame(mut_rate_mat, index=rows+a_context_labels[1:], columns=columns)
                mut_rate_mat = (mut_rate_mat / np.sum(mut_rate_mat.values)) * scaler
            else:
                mut_rate_mat = pd.DataFrame(torch.reshape(mut_rate_mat, [1,12]), index=['N[X>Y]N'], columns=columns)
                mut_rate_mat = (mut_rate_mat / np.sum(mut_rate_mat.values)) * (scaler / 12)
                #print('mut rate mat', gene_order[index], mut_rate_mat)
            
            gene_mut_rates[gene_order[index]] = [mut_rate_mat, subset_genes[gene_order[index]]] #gene_info[gene_order[index]][1]
            #print(f'gene sums {gene_order[index]}, {np.sum(gene_mut_rates[gene_order[index]][0].values)}')
            #mut_rate_mat.to_csv('simulation_output/'+gene_order[index]+'_mut_rates.csv')
        #print(gene_mut_rates)
        pd.DataFrame(torch.mean(torch.stack([torch.tensor(mat[0].values) for mat in list(gene_mut_rates.values())[:-3]]), dim=0), index=global_mut_rate_mat.index, columns=global_mut_rate_mat.columns).to_csv('simulation_output/global_mut_rates_from_genes.csv')
    mut_mat = pd.DataFrame(np.zeros([16,12]), index=rows+a_context_labels[1:], columns=columns)
    print(f'wrangling timer {time.perf_counter()-data_wrangling_timer}')
    #print(f'global sum {np.sum(global_mut_rate_mat.values)}')
    

    fasta_ref_np = fasta_ref.values
    '''loop through genome until mut_count mutations are placed'''
    while current_mut_count < mut_count:
        print('iteration, mutcount: ', iteration, current_mut_count)
        loop_timer = time.perf_counter()
        possible_muts_per_epoch = 0 #stores the number of possible sites that can mutate based on rates
        #gene_mat_check = []
        
        '''loop through each position in the genome except of the first and last nucleotides (these don't have full contexts so we ignore)'''
        for position in range(fasta_ref_np.shape[0])[1:-1]:

            #get current triplet and possible mutations
            prev_nuc, current_nuc, next_nuc = fasta_ref_np[position-1:position+2] #fasta_mutated.loc[position-1:position+1].values
            possible_mut_labels = [label for label in mut_rate_mat.columns if current_nuc+'>' in label]
            
            #next_nuc_temp used for contexts 'N'
            prev_nuc_temp = 'N'
            next_nuc_temp = 'N'
            if contexts == 'full_contexts':
                prev_nuc_temp = prev_nuc
                next_nuc_temp = next_nuc

            if sim_type == 'global':
                cumsum = np.cumsum(global_mut_rate_mat.loc[prev_nuc_temp+'[X>Y]'+next_nuc_temp,possible_mut_labels].values)
            else:
                #pick gene mut_rate_mat
                intron = True
                for gene in gene_order:
                    #if the position is in a gene, use that matrix
                    #check for N which has splits due to 9b and 9c overlapping reading frames
                    if gene == 'N':
                        for sub_region in gene_mut_rates[gene][1]:
                            if position >= sub_region[0] and position <= sub_region[1]-1:
                                cumsum = np.cumsum(gene_mut_rates[gene][0].loc[prev_nuc_temp+'[X>Y]'+next_nuc_temp,possible_mut_labels].values)
                                intron=False
                                #print('inside', position, fasta_ref_np[position])
                                break
                    #check for continuous genes
                    else:
                        if position >= gene_mut_rates[gene][1][0] and position <= gene_mut_rates[gene][1][1]-1:
                            cumsum = np.cumsum(gene_mut_rates[gene][0].loc[prev_nuc_temp+'[X>Y]'+next_nuc_temp,possible_mut_labels].values)
                            intron=False
                            break
                        
                #else use global matrix
                if intron:
                    cumsum = np.cumsum(global_mut_rate_mat.loc[prev_nuc_temp+'[X>Y]'+next_nuc_temp,possible_mut_labels].values)
                    #print('intron')
            
            if contexts == 'blind_contexts':
                cumsum = np.asarray([cumsum[-1]/3, cumsum[-1]*2/3, cumsum[-1]]) #reformat cumsum to give even weighting to all resulting mutations

            #random number to determine which mut if any is chosen
            rng_draw = rng.random(1)
            #flag to determine which mut if any is chosen
            mut_flag = ''

            #iterate possible mutation sites by 1 if the site can be mutated
            if np.sum(cumsum) != 0.0:
                possible_muts_per_epoch += 1
            else:
                continue

            #switch for which mutation to choose if any
            if rng_draw < cumsum[0]:
                #first_mut
                mut_flag=0
            elif rng_draw > cumsum[0] and rng_draw < cumsum[1]:
                #second_mut
                mut_flag=1
            elif rng_draw > cumsum[1] and rng_draw < cumsum[2]:
                #third_mut
                mut_flag=2
            else:
                continue
            
            #update mutation_dict with mutation
            if mut_flag in [0,1,2] and current_mut_count < mut_count:
            #if mut_flag != '':
                #print(mut_flag, cumsum, possible_mut_labels)

                #update mutated fasta
                fasta_ref_np[position] = possible_mut_labels[mut_flag][-1]

                #iterate mut count
                current_mut_count+=1

                #update mut_count df
                mut_mat.loc[prev_nuc+'[X>Y]'+next_nuc, current_nuc+'>'+possible_mut_labels[mut_flag][-1]] += 1

                #store position = ['iteration_number'+'original_triplet'+'->'+'new_triplet']
                if not position+1 in mutation_dict.keys():
                    mutation_dict[position+1] = [(str(current_mut_count) + prev_nuc+current_nuc+next_nuc + '->' + prev_nuc + possible_mut_labels[mut_flag][-1] + next_nuc)]
                else:
                    mutation_dict[position+1].append(str(current_mut_count) + prev_nuc+current_nuc+next_nuc + '->' + prev_nuc + possible_mut_labels[mut_flag][-1] + next_nuc)
        
        #store metadata in key 99999999
        if 99999999 in mutation_dict.keys():
            mutation_dict[99999999].append([iteration, possible_muts_per_epoch, current_mut_count])
        else:
            mutation_dict[99999999] = [['iteration', 'possible_muts_per_epoch', 'current_mut_count'],[iteration, possible_muts_per_epoch, current_mut_count]]
        iteration+=1
        print(f'loop timer {time.perf_counter()-loop_timer}')
    
    #cat_df = pd.concat([fasta_ref, fasta_mutated], axis=1)
    cat_df = pd.concat([fasta_ref, pd.Series(fasta_ref_np, name='mutated')], axis=1)
    #print(cat_df)
    #print(cat_df[cat_df.loc[:,'ref'] != cat_df.loc[:,'mutated']])
    mutations = cat_df[cat_df.loc[:,'ref'] != cat_df.loc[:,'mutated']]
    
    triplet_counts_after = pd.DataFrame(np.zeros([16,4]), index=rows+a_context_labels[1:], columns=['T','G','C','A'])
    '''for position in range(0,cat_df.shape[0]-2,3):
        prev_nuc, current_nuc, next_nuc = cat_df.loc[position:position+2,'mutated']
        triplet_counts_after.loc[prev_nuc+'[X>Y]'+next_nuc, current_nuc] += 1
    triplet_counts_after.to_csv('simulation_output/delete_me_40.csv')'''

    #returning fasta_results, mutation_dict_results, triplet_counts, gene_choice_testing_results
    fasta = ''.join(cat_df.loc[:,'mutated'].values)
    return fasta, mutation_dict, triplet_counts_after, iteration

'''trying to improve computation time'''
def simulate_mutations_6(mut_mat, mut_count, contexts, scaler=1):
    #mut_mat: matrix detailing mutation `rate`
    #mut_count: number of analyzable mutations to place per simulation run
    #contexts: string; whether to use full contexts, ignore neighboring nucleotides, or completely naive rates
    #scaler: value to adjust number of mutations placed per cycle through the genome after normalizing rates

    mutation_dict = {} #stores mutations placed during simulation
    fasta_ref = pd.Series([*get_fasta()[:-1]], name='ref') #read in ref fasta
    fasta_mutated = pd.Series([*get_fasta()[:-1]], name='mutated') #copy of ref fasta to store mutations
    current_mut_count = 0 #run sim until current_mut_count >= mut_count
    total_mut_count = 0 #all mutations regardless of analyzability
    iteration = 0 #store current iteration number
    a_context_labels = ['A[X>Y]N', 'A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'] #labels for unused 4fold mut contexts
    mut_labels = {'T':['T>G','T>C','T>A'], 'G':['G>T','G>C','G>A'], 'C':['C>T','C>G','C>A'], 'A':['A>T','A>G','A>C']}
    rng = np.random.default_rng() #random generator
    starting_position = rng.integers(low=0, high=29903, size=1)[0]

    #normalize rates and scale
    if contexts == 'full_contexts':
        if mut_mat.shape[0] == 12: #12x12 matrix doesn't have info on A[X>Y]N mutations
            mut_mat = pd.DataFrame(mut_mat, index=rows, columns=columns)
            mut_mat = pd.concat([mut_mat, pd.DataFrame(np.zeros([4, 12]), index=a_context_labels[1:], columns=columns)], axis=0)
        elif mut_mat.shape[0] == 16: #16x12 matrix has info on A[X>Y]N mutations
            mut_mat = pd.DataFrame(mut_mat, index=rows+a_context_labels[1:], columns=columns)
        #mut_mat = (mut_mat / np.sum(mut_mat.values)) #normalize
    elif contexts == 'naive_contexts':
        mut_mat = pd.DataFrame(torch.reshape(mut_mat, [1,12]), index=['N[X>Y]N'], columns=columns)
        #mut_mat = (mut_mat / np.sum(mut_mat.values)) * (scaler / 12) #normalize mat and scaler by number of rows in default matrix
    elif contexts == 'blind_contexts':
        mut_mat = pd.DataFrame(torch.reshape(mut_mat, [1,4]), index=['N[X>Y]N'], columns=columns_shortened)
        #mut_mat = (mut_mat / np.sum(mut_mat.values)) * (scaler / 12 / 3) #normalize mat and scaler by number of rows and columns in default matrix
    mut_mat = (mut_mat / np.sum(mut_mat.values)) * scaler #normalize
    #print(np.sum(mut_mat))
    
    #calc cumulative sums of each type of mutation to determine which mutation is placed if any
    cum_sums = {}
    for row in mut_mat.index:
        column_sums = {}
        for column in columns_shortened:
            #calc cumulative sum of rates for triplet and possible mutations
            if contexts == 'blind_contexts':
                column_sums[column] = np.cumsum(mut_mat.loc[row, [column,column,column]].values) / 3 #normalize between possible results
            else:
                column_sums[column] = np.cumsum(mut_mat.loc[row, mut_labels[column]].values)
        cum_sums[row] = column_sums
    #pd.DataFrame.from_dict(cum_sums).to_csv('simulation_output/'+contexts+'_cumsum.csv')
    
    placed_mut_mat = pd.DataFrame(np.zeros([16,12]), index=rows+a_context_labels[1:], columns=columns)
     

    fasta = fasta_ref.to_numpy()
    '''loop through genome until mut_count mutations are placed'''
    loop_timer = time.perf_counter()
    while current_mut_count < mut_count:
        #print(f'iteration : {iteration}, mutcount: {current_mut_count}')
        possible_muts_per_epoch = 0 #stores the number of possible sites that can mutate based on rates
        
        '''loop through each position in the genome except of the first and last nucleotides (these don't have full contexts so we ignore)'''
        for position in range(1, fasta.shape[0]-1):

            #added random starting position to remove bias from not making it fully through the genome on each iteration
            if position < starting_position and total_mut_count == 0:
                pass

            else:
                #get current triplet and possible mutations
                prev_nuc, current_nuc, next_nuc = fasta[position-1:position+2]

                #get cumulative sum of rates for triplet and possible mutations
                if contexts == 'full_contexts':
                    cumsum = cum_sums[prev_nuc+'[X>Y]'+next_nuc][current_nuc]
                else:
                    cumsum = cum_sums['N[X>Y]N'][current_nuc]
                
                if np.sum(cumsum) == 0.0:
                    #skip to next nucleotide since this one can't be mutated with our rates
                    continue
                else: 
                    #iterate possible mutation sites by 1 if the site can be mutated
                    possible_muts_per_epoch += 1

                #random number to determine which mut if any is chosen
                rng_draw = rng.random(1)
                #flag to determine which mut if any is chosen
                mut_flag = ''

                #switch for which mutation to choose if any
                if rng_draw > cumsum[2]:
                    #skip to next nucleotide, this one didn't mutate
                    continue
                elif rng_draw <= cumsum[0]:
                    #first_mut
                    mut_flag=0
                elif rng_draw > cumsum[0] and rng_draw <= cumsum[1]:
                    #second_mut
                    mut_flag=1
                elif rng_draw > cumsum[1] and rng_draw <= cumsum[2]:
                    #third_mut
                    mut_flag=2
                
                #update mutation_dict with mutation
                if mut_flag in [0,1,2] and current_mut_count < mut_count:

                    #update mutated fasta
                    mut_nuc = mut_labels[current_nuc][mut_flag][-1]
                    fasta[position] = mut_nuc

                    #iterate mut count if mut is in an analyzable context (no 5'A)
                    if prev_nuc != 'A':
                        current_mut_count+=1
                    #iterate total mut count
                    total_mut_count+=1 

                    #update mut_count df
                    placed_mut_mat.loc[prev_nuc+'[X>Y]'+next_nuc, current_nuc+'>'+mut_nuc] += 1

                    #store mut_number = 'original_triplet|position|new_triplet
                    mutation_dict[total_mut_count] = [prev_nuc+current_nuc+next_nuc,position+1,prev_nuc+mut_nuc+next_nuc]

        #store metadata in key 99999999
        if 99999999 in mutation_dict.keys():
            mutation_dict[99999999] += f",[{iteration}, {possible_muts_per_epoch}, {current_mut_count}]"
        else:
            mutation_dict[99999999] = f"['starting_position','iteration', 'possible_muts_per_epoch', 'current_mut_count'],[{starting_position}, {iteration}, {possible_muts_per_epoch}, {current_mut_count}]"
        iteration+=1
    print(f'loop timer {time.perf_counter()-loop_timer}, starting position {starting_position}')
    
    #deprecated
    cat_df = pd.concat([fasta_ref, pd.Series(fasta, name='mutated')], axis=1)
    mutations = cat_df[cat_df.loc[:,'ref'] != cat_df.loc[:,'mutated']]
    
    #deprecated
    triplet_counts_after = pd.DataFrame(np.zeros([16,4]), index=rows+a_context_labels[1:], columns=['T','G','C','A'])
    '''for position in range(0,cat_df.shape[0]-2,3):
        prev_nuc, current_nuc, next_nuc = cat_df.loc[position:position+2,'mutated']
        triplet_counts_after.loc[prev_nuc+'[X>Y]'+next_nuc, current_nuc] += 1
    triplet_counts_after.to_csv('simulation_output/delete_me_40.csv')'''

    #returning fasta_results, mutation_dict_results, triplet_counts, gene_choice_testing_results
    #fasta = ''.join(cat_df.loc[:,'mutated'].values)
    fasta = ''.join(fasta)
    return fasta, mutation_dict, triplet_counts_after, iteration


'''read in variant specific mutation data from sim_ref_data'''
def read_mut_ref_data(variant):
    path = 'sim_ref_data/'+variant+'/'
    variant_fasta = ''
    variant_df = 0
    #read in fasta and mutation csv for variant
    for file in os.listdir(path):
        if 'nucleotide-mutations.csv' in file:
            variant_df = pd.read_csv(path+file, header=0, index_col=0)
    variant_df = variant_df.rename({column:column + re.search(r'\(\w+\)', variant).group(0) for column in variant_df.columns}, axis=1)
    return variant_df

'''read in all variant info and make big dataframe'''
def gen_variants_df(path, threshold):
    #find all variant folders
    variant_folders = [folder for folder in os.listdir(path) if 'clade' in folder]
    #convert individual dataframes into single dataframe
    variants_df = pd.concat([read_mut_ref_data(folder) for folder in variant_folders], axis=1)

    #calc actual mut rates using count column and number of sequences contributing to clade on cov-spectrum
    with open('sim_ref_data/info', 'r') as f:
        lines = f.readlines()
    variant_counts = pd.Series({re.search(r'(?<== )\w+', line).group(0).lower():float(re.search(r'(?<=, )\w+', line).group(0)) for line in lines[8:]})
    #print(variant_counts)

    #pull reference fasta
    ref_fasta = pd.Series([*get_fasta()[:-1]], name='ref')
    #extend reference fasta to match simulated fasta length
    ref_fasta = pd.concat([ref_fasta, pd.Series(['A']*15, index=range(ref_fasta.shape[0], ref_fasta.shape[0]+15))], axis=0)
    #print(ref_fasta)
    output_folders = [folder for folder in os.listdir('sim_ref_data/') if 'full' in folder]
    variant_mut_counts = pd.DataFrame({name:0 for name in variant_counts.index}, columns=[0])

    '''loop through each variant'''
    for variant in [column for column in variants_df.columns if 'count' in column]:
        #divide number of sequences with mutation by total number of sequences for variant
        variants_df.loc[:,variant] = variants_df.loc[:,variant] / variant_counts.loc[re.search(r'\(\w+\)', variant).group(0)[1:-1]]
        
        #copy ref fasta and convert char to list of chars for each nucleotide
        var_fasta = ref_fasta.copy()
        for index in var_fasta.index:
            var_fasta.loc[index] = [var_fasta.loc[index]]
        mut_count = 0 #number of muts to place for each variation
        #loop through all mutations for variant
        for index in variants_df.loc[:,variant].dropna().index:
            #subset mutations to only include those with frequency >= threshold and base substitution
            if index[-1] != '-':
                if variants_df.loc[index,variant] >= threshold:
                    if var_fasta.loc[int(index[1:-1])-1] != ref_fasta[int(index[1:-1])-1]:
                        var_fasta.loc[int(index[1:-1])-1].append(index[-1])
                    #number of mutations is generated by number of muts with > 0.01 frequency
                    if variants_df.loc[index,variant] > threshold:
                        mut_count += 1
        #print(var_fasta)
        variant_name = [folder for folder in output_folders if re.search(r'\(\w+\)', variant).group(0) in folder][0]
        if not os.path.exists('sim_ref_data/'+variant_name+'/mutated_fastas/'):
            os.mkdir('sim_ref_data/'+variant_name+'/mutated_fastas/')
        var_fasta.to_csv('sim_ref_data/'+variant_name+'/mutated_fastas/'+str(threshold)+'_mutated.fasta')
        
        variant_mut_counts.loc[re.search(r'\(\w+\)', variant).group(0)[1:-1], 0] = mut_count
    print(variant_mut_counts)

    return variants_df, variant_mut_counts

'''simulation based on data for a single variant'''
def init_sim(variant, num_sims, num_muts, mut_mat, contexts='full_contexts', threshold='5e-05', scaler=.25):
    #variant = current variant being tested
    #num_sims = number of independent simulations to run
    #num_muts = number of mutations to place during a run
    #mut_mat = the number CDP profile for variant
    #contexts = whether to use full 12x12 mut matrix, 1x12 contextless rates, or 1x12 contextless rates that ignore resulting mutations in sim
    #threshold = minimum frequency of mutations to compare against
    #scaler = float for mut_mat scaling (alters how fast num_muts is reached) 
    sim_type = 'global'

    #create file structure
    if not os.path.exists('simulation_output/'+sim_type+'/full_contexts'):
        os.mkdir('simulation_output/'+sim_type+'/full_contexts')
    if not os.path.exists('simulation_output/'+sim_type+'/naive_contexts'):
        os.mkdir('simulation_output/'+sim_type+'/naive_contexts')
    if not os.path.exists('simulation_output/'+sim_type+'/blind_contexts'):
        os.mkdir('simulation_output/'+sim_type+'/blind_contexts')

    base_folder = 'simulation_output/'+sim_type+'/'+contexts+'/'
    #don't overwrite finished runs
    if os.path.exists(base_folder+variant+'/mut_dicts'):
        if os.path.exists(base_folder+variant+'/mut_dicts/'+str(threshold)):
            if len(os.listdir(base_folder+variant+'/mut_dicts/'+str(threshold))) >= num_sims:
                return
        else:
            os.mkdir(base_folder+variant+'/fastas/'+str(threshold))
            os.mkdir(base_folder+variant+'/mut_dicts/'+str(threshold))
    else:
        os.mkdir(base_folder+variant)
        os.mkdir(base_folder+variant+'/fastas')
        os.mkdir(base_folder+variant+'/mut_dicts')
        os.mkdir(base_folder+variant+'/triplet_counts')
        os.mkdir(base_folder+variant+'/spike_data')
        os.mkdir(base_folder+variant+'/fastas/'+str(threshold))
        os.mkdir(base_folder+variant+'/mut_dicts/'+str(threshold))
        
    #run num_sims number of simulations
    for sim_run in range(num_sims):
        print('running simulation # ', sim_run, ' of type ', sim_type, ' for variant ', variant, 'with ', contexts)
        if contexts == 'full':
            num_muts = .7*num_muts
        fasta_results, mutation_dict_results, triplet_counts, iteration_count = simulate_mutations_6(mut_mat, num_muts, contexts, scaler)
        save_fasta(fasta_results, sim_run, num_muts, iteration_count, sim_type, variant, contexts, threshold)
        save_mutation_dict(mutation_dict_results, sim_run, sim_type, variant, contexts, threshold)
        '''if contexts:
            triplet_counts.to_csv('simulation_output/'+sim_type+'/full_contexts/'+variant+'/triplet_counts/'+'tc_'+str(sim_run)+'.csv')
        else:
            triplet_counts.to_csv('simulation_output/'+sim_type+'/naive_contexts/'+variant+'/triplet_counts/'+'tc_'+str(sim_run)+'.csv')'''

'''compare simulation fasta output to reference generated from possible mutations (from cov-spectrum)'''
def compare_to_reference_deprecated(variant, sim_type, contexts, threshold):
    #read in reference fasta
    variant_folder = [folder for folder in os.listdir('sim_ref_data') if re.search(r'\('+variant+'\)', folder)][0]
    ref_fasta = pd.read_csv('sim_ref_data/'+variant_folder+'/mutated_fastas/'+str(threshold)+'_mutated.fasta', header=0, index_col=0)
    '''if contexts:
        contexts = 'full_contexts'
    else:
        contexts = 'naive_contexts'
    '''

    for fasta_num, sim_fasta in enumerate(os.listdir('simulation_output/'+sim_type+'/'+contexts+'/'+variant+'/fastas/'+str(threshold))):
        with open('simulation_output/'+sim_type+'/'+contexts+'/'+variant+'/fastas/'+str(threshold)+'/'+sim_fasta, 'r') as f:
            lines = f.readlines()
        sim_fasta = lines[1]
        if not os.path.exists('simulation_output/'+sim_type+'/'+contexts+'/'+variant+'/analysis'):
            os.mkdir('simulation_output/'+sim_type+'/'+contexts+'/'+variant+'/analysis')
        if not os.path.exists('simulation_output/'+sim_type+'/'+contexts+'/'+variant+'/analysis/'+str(threshold)):
            os.mkdir('simulation_output/'+sim_type+'/'+contexts+'/'+variant+'/analysis/'+str(threshold))
        compare_fastas_2(ref_fasta, sim_fasta, 'simulation_output/'+sim_type+'/'+contexts+'/'+variant+'/analysis/'+str(threshold)+'/'+str(fasta_num))

'''generate heatmap comparing variant contexts'''  
def variant_mut_rates_2(global_subset_avg, genes_avg, variant_order, gene_order):
    #symbols for naming
    variant_order_temp = [chr(945), chr(946), chr(948), chr(949), chr(951), chr(947), chr(953), chr(954), chr(955), chr(956), chr(959)]
    print({variant_order[i]:variant_order_temp[i] for i in range(len(variant_order))})

    #empty 12x12 array
    empty_template = np.asarray((['']*12)*12).reshape([12,12])
    #matrix storing symbols for variants at each context
    high_contexts = pd.DataFrame(empty_template, index = rows, columns=columns, dtype=str)
    low_contexts = pd.DataFrame(empty_template, index = rows, columns=columns, dtype=str)

    #normalize variant rates
    #11x12x12 -> 11x12x12 normalized
    #global_subset_avg_normalized = []
    #for index, mat in enumerate(global_subset_avg):
    #    global_subset_avg_normalized.append(torch.divide(mat, torch.sum(mat)))
    #global_subset_avg_normalized = torch.stack(global_subset_avg_normalized)

    #std and mean matrix across variants for 12x12 contexts
    #11x12x12 -> 12x12
    #context_std, context_mean = torch.std_mean(global_subset_avg_normalized, dim=(0))
    #context_std = pd.DataFrame(context_std, index=rows, columns=columns)
    #context_mean = pd.DataFrame(context_mean, index=rows, columns=columns)
    context_std, context_mean = torch.std_mean(global_subset_avg, dim=(0))
    context_std = pd.DataFrame(context_std, index=rows, columns=columns)
    context_mean = pd.DataFrame(context_mean, index=rows, columns=columns)
    population_std, population_mean = torch.std_mean(global_subset_avg)
    print(population_std, population_mean)
    
    #print(context_std, context_mean)
    #loop through 12x12 and assign variant name to context if: rate > mean+n*std
    for name_index, mat in enumerate(global_subset_avg):
        mat = pd.DataFrame(mat[:12,:], index=rows, columns=columns)
        #mat = mat / np.sum(mat.values)
        for column in mat.columns:
            for index in mat.index:
                #if mat.loc[index, column] != 0 and mat.loc[index, column] > 0.1:
                if mat.loc[index, column] != 0 and mat.loc[index, column] > population_mean + (2*population_std):
                    high_contexts.loc[index, column] += variant_order_temp[name_index]
                elif mat.loc[index, column] != 0 and mat.loc[index, column] < population_mean - (2*population_std):
                    low_contexts.loc[index, column] += variant_order_temp[name_index]
    #print(variant_order)
    #print(global_4fold_rate)
    #mat = mat / 10000000

    #plot high contexts for global mat
    print(high_contexts)
    fig, axs = plt.subplots(figsize=(15,15))
    ax = sns.heatmap(context_mean, cmap='Blues', ax=axs, vmin=0, vmax=1, annot=high_contexts, fmt='', annot_kws={'fontweight':'bold', 'size':'xx-large'})
    #ax = sns.heatmap(context_mean+(1*context_std), cmap='Reds', mask=(context_mean+context_std)<(population_mean+population_std).numpy(),  ax=axs, cbar=False, annot=global_4fold_rate, fmt='', annot_kws={'fontweight':'bold', 'size':'xx-large'})
    #ax = sns.heatmap(context_mean+(1*context_std), cmap='Blues',mask=(context_mean-context_std)>(population_mean-population_std).numpy(),  ax=axs, cbar=False, annot=global_4fold_rate, fmt='', annot_kws={'fontweight':'bold', 'size':'xx-large'})
    ax.set_xticks(range(13), minor=True)
    ax.set_yticks(range(13), minor=True)
    ax.grid(visible=True, which='minor')
    #ax.legend()
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_yticklabels(high_contexts.index, rotation='horizontal')
    ax.set_title('Variant Context Comparisons')
    plt.savefig('sim_ref_data/4fold/global_variant_high_contexts.png')
    plt.close()

    '''#plot low contexts for global mat
    print(low_contexts)
    fig, axs = plt.subplots(figsize=(15,15))
    ax = sns.heatmap(context_mean, cmap='Blues', ax=axs, annot=low_contexts, fmt='', annot_kws={'fontweight':'bold', 'size':'xx-large'})
    #ax = sns.heatmap(context_mean+(1*context_std), cmap='Reds', mask=(context_mean+context_std)<(population_mean+population_std).numpy(),  ax=axs, cbar=False, annot=global_4fold_rate, fmt='', annot_kws={'fontweight':'bold', 'size':'xx-large'})
    #ax = sns.heatmap(context_mean+(1*context_std), cmap='Blues',mask=(context_mean-context_std)>(population_mean-population_std).numpy(),  ax=axs, cbar=False, annot=global_4fold_rate, fmt='', annot_kws={'fontweight':'bold', 'size':'xx-large'})
    ax.set_xticks(range(13), minor=True)
    ax.set_yticks(range(13), minor=True)
    ax.grid(visible=True, which='minor')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_yticklabels(low_contexts.index, rotation='horizontal')
    ax.set_title('Variant Context Comparisons')
    plt.savefig('sim_ref_data/4fold/global_variant_low_contexts.png')
    plt.close()'''

    #normalize gene matrices
    #11,15,12,12 -> 11,15,12,12 normalized
    #genes_avg_normalized = []
    #for variant in range(genes_avg.shape[0]):
    #    variant_norm = []
    #    for gene in range(genes_avg.shape[1]):
    #        variant_norm.append(torch.divide(genes_avg[variant,gene], torch.sum(genes_avg[variant,gene])))
    #    genes_avg_normalized.append(torch.stack(variant_norm))
    #genes_avg_normalized = torch.stack(genes_avg_normalized)

    #std and mean matrix across variants and genes for 12x12 contexts
    #11,15,12,12 -> 15,12,12
    genes_context_std, genes_context_mean = [],[]
    #context_std, context_mean = torch.std_mean(genes_avg_normalized, dim=(0))
    context_std, context_mean = torch.std_mean(genes_avg, dim=(0)) #15,12,12
    #15,12,12
    for gene in range(context_std.shape[0]):
        genes_context_std.append(pd.DataFrame(context_std[gene], index=rows, columns=columns))
        genes_context_mean.append(pd.DataFrame(context_mean[gene], index=rows, columns=columns))
    population_std = torch.std(genes_avg.flatten())
    population_mean = torch.mean(genes_avg.flatten())
    #print(population_std, population_mean)

    fig, axs = plt.subplots(figsize=(25,45), nrows=genes_avg.shape[1])
    for gene_index in range(genes_avg.shape[1]):
        high_contexts = pd.DataFrame(empty_template, index = rows, columns=columns, dtype=str)
        low_contexts = pd.DataFrame(empty_template, index = rows, columns=columns, dtype=str)
        for name_index, mat in enumerate(genes_avg[:,gene_index,:,:]):
            mat = pd.DataFrame(mat[:12,:], index=rows, columns=columns)
            #mat = mat / np.sum(mat.values)
            for column in mat.columns:
                for index in mat.index:
                    #if mat.loc[index, column] != 0 and mat.loc[index, column] > 0.1:
                    #if mat.loc[index, column] != 0:
                    #if mat.loc[index, column] != 0 and mat.loc[index, column] > genes_context_mean[gene_index].loc[index, column]+(1*genes_context_std[gene_index].loc[index, column]):
                    if mat.loc[index, column] != 0 and mat.loc[index, column] > population_mean+(2*population_std):
                        high_contexts.loc[index, column] += variant_order_temp[name_index]
                    elif mat.loc[index, column] != 0 and mat.loc[index, column] < population_mean-(2*population_std):
                        low_contexts.loc[index, column] += variant_order_temp[name_index]
        
        axs[gene_index] = sns.heatmap(genes_context_mean[gene_index], cmap='Blues', ax=axs[gene_index], vmin=0, vmax=1, annot=high_contexts, fmt='', annot_kws={'fontweight':'bold', 'size':'xx-large'})
        axs[gene_index].set_xticks(range(13), minor=True)
        axs[gene_index].set_yticks(range(13), minor=True)
        axs[gene_index].grid(visible=True, which='minor')
        axs[gene_index]
        for _, spine in axs[gene_index].spines.items():
            spine.set_visible(True)
        #axs[gene_index].set_yticklabels(gene_4fold_rate.index, rotation='horizontal')
        axs[gene_index].set_title('Variant Context Comparisons for '+ gene_order[gene_index])
    plt.tight_layout()
    plt.savefig('sim_ref_data/4fold/genes_variant_contexts.png')
    plt.close()

'''read in variant specific mutation data from sim_ref_data'''
def read_mut_ref_data_2(variant):
    path = 'sim_ref_data/'+variant+'/'
    variant_df = 0
    #read in fasta and mutation csv for variant
    for file in os.listdir(path):
        if 'nucleotide-mutations.csv' in file:
            variant_df = pd.read_csv(path+file, header=0, index_col=0)
    
    return variant_df


'''plot all variant mut rates'''
def plot_variant_mut_rates(genes_avg_mat, global_avg_subset_mat, genes_naive_mat, global_naive_subset_mat, variant_order, gene_order):
    for genes_mat_index, big_mat in enumerate([genes_avg_mat, genes_naive_mat]):
        for variant_index in range(big_mat.shape[0]):
            if genes_mat_index == 0:
                mat_type = 'full_contexts'
            else:
                mat_type = 'naive_contexts'
            path = 'sim_ref_data/mut_rates/'+variant_order[variant_index]+'/'+mat_type+'_genes.png'
            plot_mut_mat(variant_mat=big_mat[variant_index], path=path, gene_order=gene_order, data_type='genes', contexts=mat_type)
    for global_mat_index, global_mat in enumerate([global_avg_subset_mat, global_naive_subset_mat]):
        for variant_index in range(global_mat.shape[0]):
            if global_mat_index == 0:
                mat_type = 'full_contexts'
            else:
                mat_type = 'naive_contexts'
            path = 'sim_ref_data/mut_rates/'+variant_order[variant_index]+'/'+mat_type+'_global.png'
            plot_mut_mat(variant_mat=global_mat[variant_index], path=path, gene_order=gene_order, data_type='global', contexts=mat_type)

'''plot a mutation matrix to path'''
def plot_mut_mat(variant_mat, path, gene_order, data_type, contexts, include_a=False, vmin=0, vmax=1, annot=False):
    #potential shapes: 15x16x12 = genes full mat, 15x12 = genes naive mat, 16x12 = global full mat, 12 = global naive mat
    print('generating ' + path)
    print(variant_mat.shape)
    if 'full' in contexts:
        rows_temp = rows_figs
    else:
        rows_temp = ['N[X>Y]N']
    cols_temp = columns_figs
    if data_type == 'genes':
        fig, axs = plt.subplots(figsize=(8,45), nrows=variant_mat.shape[0])
        for gene in range(variant_mat.shape[0]):
            if contexts == 'super_naive_contexts':
                sns.heatmap(torch.unsqueeze(variant_mat[gene],0), cmap='Greys', ax=axs[gene], vmin=vmin, vmax=vmax, annot=annot, linewidth=.5, linecolor='gray')
                axs[gene].set_xticklabels(labels=columns_shortened_figs)
                axs[gene].set_yticklabels(labels=rows_temp, rotation='horizontal')
            elif contexts == 'naive_contexts':
                fig.set_size_inches(8,16)
                sns.heatmap(torch.unsqueeze(variant_mat[gene],0), cmap='Greys', ax=axs[gene], xticklabels=cols_temp, yticklabels=rows_temp, vmin=vmin, vmax=vmax, annot=annot, linewidth=.5, linecolor='gray')
                axs[gene].set_yticklabels(labels=rows_temp, rotation='horizontal')
            elif include_a == False:
                sns.heatmap(variant_mat[gene,:12,:], cmap='Greys', ax=axs[gene], xticklabels=cols_temp, yticklabels=rows_temp[:12], vmin=vmin, vmax=vmax, annot=annot, linewidth=.5, linecolor='gray')
            else:
                sns.heatmap(variant_mat[gene], cmap='Greys', ax=axs[gene], xticklabels=cols_temp, yticklabels=rows_temp, vmin=vmin, vmax=vmax, annot=annot, linewidth=.5, linecolor='gray')
            axs[gene].set_title(gene_order[gene])
    elif data_type == 'global':
        if contexts == 'super_naive_contexts':
            fig, axs = plt.subplots(figsize=(4,1.3), dpi=200, layout='tight')
            sns.heatmap(torch.unsqueeze(variant_mat,0), cmap='Greys', ax=axs, vmin=vmin, vmax=vmax, annot=annot, linewidth=.5, linecolor='gray')
            axs.set_xticklabels(labels=columns_shortened_figs)
            axs.set_yticklabels(labels=rows_temp, rotation='horizontal')
        elif contexts == 'naive_contexts':
            fig, axs = plt.subplots(figsize=(8,1.3), dpi=200, layout='tight')
            sns.heatmap(torch.unsqueeze(variant_mat,0), cmap='Greys', ax=axs, xticklabels=cols_temp, yticklabels=rows_temp, vmin=vmin, vmax=vmax, annot=annot, linewidth=.5, linecolor='gray')
            axs.set_yticklabels(labels=rows_temp, rotation='horizontal')
        elif include_a == False:
            fig, axs = plt.subplots(figsize=(8,7))
            sns.heatmap(variant_mat[:12,:], cmap='Greys', ax=axs, xticklabels=cols_temp, yticklabels=rows_temp[:12], vmin=vmin, vmax=vmax, annot=annot, linewidth=.5, linecolor='gray')
        else:
            fig, axs = plt.subplots(figsize=(8,7))
            sns.heatmap(variant_mat, cmap='Greys', ax=axs, xticklabels=cols_temp, yticklabels=rows_temp, vmin=vmin, vmax=vmax, linewidth=.5, linecolor='gray')
        axs.set_title('global subset rates')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


'''vizualize gwtc counts for genes and genome'''
def viz_gwtc():
    #read in gwtc data
    path = 'sim_ref_data/fourfold_gwtc/triplets/'
    gene_df_dict = {}
    for gene in gene_info.keys():
        gene_df_dict[gene] = pd.read_csv(path+gene+'.csv', header=0, index_col=0)
        print(gene_df_dict[gene])
    
    #plot gwtc data for each gene
    num_genes = len(gene_df_dict)
    fig, axs = plt.subplots(figsize=(8,45), nrows=num_genes)
    for index, gene in enumerate(gene_df_dict.keys()):
        sns.heatmap(gene_df_dict[gene].iloc[:12,:].values, cmap='Blues', ax=axs[index], xticklabels=gene_df_dict[gene].columns, yticklabels=rows[:12])
        axs[index].set_title(gene)
    plt.tight_layout()
    plt.savefig(path+'genes_gwtc.png')
    plt.close()

    #plot total gwtc for coding regions
    fig, axs = plt.subplots(figsize=(8,8))
    total_gwtc = pd.DataFrame(np.zeros([16,4]), index=gene_df_dict['ORF1ab'].index, columns=gene_df_dict['ORF1ab'].columns)
    for gene in ['ORF1ab','E','M','N','ORF3a','ORF6','ORF7a','ORF7b','ORF8','ORF10','S']:
        total_gwtc += gene_df_dict[gene]
    sns.heatmap(total_gwtc.iloc[:12,:], cmap='Blues', ax=axs, xticklabels=total_gwtc.columns, yticklabels=rows[:12])
    axs.set_title('gwtc of coding regions')
    plt.savefig(path+'genome_gwtc.png')
    plt.close()


'''read in mut rates from population data'''
def read_global_mut_rates(regions):
    total_tensor = []
    genes = ['E', 'M', 'N', 'ORF1ab', 'ORF3a', 'ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'ORF10', 'S', 'ORF1a', 'ORF3b', 'ORF9b', 'ORF9c']
    variant_order, gene_order = [],[]
    if not regions:
        for gene in genes:
            try:
                total_tensor.append(torch.tensor(pd.read_csv('population_4fold_rates/global/'+gene+'_subset_rates.csv', index_col=0, header=0).values))
            except:
                total_tensor.append(torch.tensor(np.zeros([12,12])))
            gene_order.append(gene)
        #full contexts
        total_tensor = torch.stack(total_tensor) #shape = [15,12,12]
        global_rate_tensor = torch.tensor(pd.read_csv('population_4fold_rates/global/global_subset_rates.csv', index_col=0, header=0).values)

    else:
        regional_dict = {}
        for region in regions:
            #print(region)
            for gene in genes:
                #print(gene)
                try:
                    mat = torch.tensor(pd.read_csv('population_4fold_rates/'+region+'/'+gene+'_subset_rates.csv', index_col=0, header=0).values)
                except:
                    mat = torch.tensor(np.zeros([12,12]))
                if gene not in regional_dict.keys():
                    regional_dict[gene] = [mat]
                else:
                    regional_dict[gene].append(mat)
                gene_order.append(gene)
        #print(regional_dict)
        regional_dict = {gene:torch.mean(torch.stack(regional_dict[gene]), dim=(0)) for gene in regional_dict.keys()}
        #print(regional_dict)
        gene_order = gene_order[:15]
        #print(gene_order)
        total_tensor = torch.stack([regional_dict[gene] for gene in gene_order]) #shape = [15,12,12]
        #print(total_tensor.shape)
        

        #full contexts
        global_rate_tensor = []
        for region in regions:
            #print(region)
            global_rate_tensor.append(torch.tensor(pd.read_csv('population_4fold_rates/'+region+'/global_subset_rates.csv', index_col=0, header=0).values))
        global_rate_tensor = torch.mean(torch.stack(global_rate_tensor), dim=(0))
        if global_rate_tensor.shape[0] == 1:
            global_rate_tensor = torch.squeeze(global_rate_tensor, dim=(0))
        #print(global_rate_tensor.shape)


    #naive contexts
    genes_naive_mat = torch.mean(total_tensor, dim=(1))
    global_naive_subset_mat = torch.mean(global_rate_tensor, dim=(0))

    print(total_tensor.shape, variant_order, gene_order)
    return total_tensor, global_rate_tensor, genes_naive_mat, global_naive_subset_mat, variant_order, gene_order

#read through mut_dicts for simulation and calculate positional and contextual matches
def analyze_mutations_using_mut_dicts(variant_order, thresholds, analysis_threshold, analysis_variant, contexts, output_info):
    var_folders = [folder for folder in os.listdir('sim_ref_data') if 'clade' in folder and 'jean' not in folder and re.search(r'\(\w+\)', folder).group(0)[1:-1] in variant_order]
    #main_output_df stores positional and contextual matches for variant/threshold/context combinations compared to reference mutations
    main_output_df = pd.DataFrame(np.zeros([2*len(var_folders),len(thresholds)*len(contexts)]), index=pd.MultiIndex.from_product([var_folders,['positional_matches','contextual_matches']]), columns=pd.MultiIndex.from_product([thresholds,contexts]))
    #distribution_df stores positional and contextual matches for each set of sims (mean will be main_output_df value)
    distribution_df = pd.DataFrame()
    #tensor shape: (num_variants,num_thresholds,num_contexts,num_sims,[12,16],12)
    if not os.path.exists('simulation_output/global/analysis_'+output_info):
        os.mkdir('simulation_output/global/analysis_'+output_info)
    #if not os.path.exists('simulation_output/global/analysis/summary_mats'):
    #    os.mkdir('simulation_output/global/analysis/summary_mats')

    positional_tensor,contextual_tensor = [],[]
    #loop through thresholds
    for threshold in thresholds:
        print(threshold)
        #if not os.path.exists('simulation_output/global/analysis/summary_mats/'+threshold):
        #    os.mkdir('simulation_output/global/analysis/summary_mats/'+threshold)
        threshold_positional,threshold_contextual = [],[]

        #read variant mutation list
        ref_muts_dict = {var_folder : pd.read_csv('sim_ref_data/'+var_folder+'/reference_mutations/'+threshold+'_reference_mutations.csv', index_col=0, header=0) for var_folder in var_folders}

        #loop through contexts
        for context_type in contexts:
            #if not os.path.exists('simulation_output/global/analysis/summary_mats/'+threshold+'/'+context_type):
            #    os.mkdir('simulation_output/global/analysis/summary_mats/'+threshold+'/'+context_type)
            sim_positional,sim_contextual = [],[]
            #loop through sims
            for sim_index, sim in enumerate(os.listdir('simulation_output/global/'+context_type+'/'+analysis_variant+'/mut_dicts/'+analysis_threshold)):
                #print(context_type, sim)
                var_positional,var_contextual = [],[]
                sim_muts = pd.read_csv('simulation_output/global/'+context_type+'/'+analysis_variant+'/mut_dicts/'+analysis_threshold+'/'+sim, index_col=0, header=0) #read in mut list for simulation
                sim_muts.drop(99999999, axis=0, inplace=True) #remove metadata row
                #reformat mut list and only include final mutation placed at any position
                #orig_triplets, positions, mut_triplets = np.array([row.split('|') for row in sim_muts.to_numpy().flatten()]).T
                #sim_muts['orig_nuc'] = [triplet[1] for triplet in orig_triplets]
                #sim_muts['position'] = positions.astype(int)
                #sim_muts['mut_nuc'] = [triplet[1] for triplet in mut_triplets]
                #sim_muts['first_nuc'] = [triplet[0] for triplet in orig_triplets]
                sim_muts['position'] = sim_muts['position'].astype(int)
                sim_muts['first_nuc'] = [triplet[0] for triplet in sim_muts['original']]
                sim_muts['orig_nuc'] = [triplet[1] for triplet in sim_muts['original']]
                sim_muts['mut_nuc'] = [triplet[1] for triplet in sim_muts['mutation']]
                sim_muts.drop_duplicates(subset='position', keep='last', inplace=True)
                sim_muts.drop(sim_muts.loc[sim_muts['first_nuc']=='A'].index, axis=0, inplace=True)
                
                #loop through variants
                for var_folder in var_folders:
                    variant = re.search(r'\(\w+\)', var_folder).group(0)[1:-1]
                    #if not os.path.exists('simulation_output/global/analysis/summary_mats/'+threshold+'/'+context_type+'/'+variant):
                    #    os.mkdir('simulation_output/global/analysis/summary_mats/'+threshold+'/'+context_type+'/'+variant)
                
                    ref_muts = ref_muts_dict[var_folder]
                    #subset muts to only include those with known reference positions
                    positional_matches = sim_muts.merge(ref_muts, how='inner', left_on='position', right_on='position')
                    #subset positional matches to only include those with matching original and mutant nucleotides
                    contextual_matches = positional_matches.loc[(positional_matches['orig_nuc']==positional_matches['old']) & (positional_matches['mut_nuc']==positional_matches['mut'])]

                    #place positional matches and contextual matches into 12x12 cdp matrix
                    var_positional.append(torch.tensor(gen_mut_mat(positional_matches['position'], positional_matches['orig_nuc'], positional_matches['mut_nuc'], (16,12)).to_numpy()))
                    var_contextual.append(torch.tensor(gen_mut_mat(contextual_matches['position'], contextual_matches['orig_nuc'], contextual_matches['mut_nuc'], (16,12)).to_numpy()))
                #distribution_df = pd.concat([distribution_df, pd.Series([threshold,context_type,'positional_matches',re.search(r'\(\w+\)',var_folder).group(0)[1:-1],*context_values[0]], index=['threshold','context','match','variant']+[i for i in range(len(context_values[0]))])], axis=1)
                #distribution_df = pd.concat([distribution_df, pd.Series([threshold,context_type,'contextual_matches',re.search(r'\(\w+\)',var_folder).group(0)[1:-1],*context_values[1]], index=['threshold','context','match','variant']+[i for i in range(len(context_values[0]))])], axis=1)
                #var_positional is a list of 15 variant tensors > convert to tensor
                sim_positional.append(torch.stack(var_positional))
                sim_contextual.append(torch.stack(var_contextual))
            #sim_positional is a list of 100x15x16x12 tensors > covnert to tensor
            #[num_vars,16,12]
            threshold_positional.append(torch.stack(sim_positional))
            threshold_contextual.append(torch.stack(sim_contextual))
        #threshold_positional is a list of 5x100x15x16x12
        positional_tensor.append(torch.stack(threshold_positional))
        contextual_tensor.append(torch.stack(threshold_contextual))
    positional_tensor = torch.stack(positional_tensor)
    contextual_tensor = torch.stack(contextual_tensor)
    print(positional_tensor.shape)
    #[5,3,100,17,16,12]
    #pull sum of positional and contextual matches for each sim combination and normalize by num_sims
    for threshold_index, threshold in enumerate(thresholds):
        for context_index, context_type in enumerate(contexts):
            for var_index, var_folder in enumerate(var_folders):
                main_output_df.loc[(var_folder,'positional_matches'),(threshold,context_type)] = torch.sum(positional_tensor[threshold_index,context_index,:,var_index], dim=(0,1,2)).item() / positional_tensor.shape[-4]
                main_output_df.loc[(var_folder,'contextual_matches'),(threshold,context_type)] = torch.sum(contextual_tensor[threshold_index,context_index,:,var_index], dim=(0,1,2)).item() / contextual_tensor.shape[-4]
                
    #main_output_df.loc[(var_folder,'positional_matches'),(threshold,context_type)] = np.mean(context_values[0])
    #main_output_df.loc[(var_folder,'contextual_matches'),(threshold,context_type)] = np.mean(context_values[1])
    #print(np.mean(context_values[0]), np.mean(context_values[1]))
    
    main_output_df.index = pd.MultiIndex.from_product([[re.search(r'\(\w+\)', folder).group(0)[1:-1] for folder in var_folders],['positional_matches','contextual_matches']])
    #main_output_df = main_output_df.loc[['alpha','delta','kraken','omicron','pirola','beta','epsilon','eta','gamma','hongkong','iota','kappa','lambda','mu','all','persistent','transient'],:]
    main_output_df.to_csv('simulation_output/global/analysis_'+output_info+'/thresholded_mat.csv')
    #torch.save(positional_match_tensor, 'simulation_output/global/analysis/positional_tensor.pt')
    #torch.save(contextual_match_tensor, 'simulation_output/global/analysis/contextual_tensor.pt')
    #distribution_df.columns = pd.MultiIndex.from_tuples([(distribution_df.iloc[0:4,index]) for index in range(distribution_df.shape[1])])
    #distribution_df.drop(['threshold','context','match','variant'], axis=0, inplace=True)
    #distribution_df.columns = pd.MultiIndex.from_tuples(distribution_df.columns)
    #distribution_df.T.to_csv('simulation_output/global/analysis/distributions.csv')

    #create analysis tensors
    #positional_tensor = torch.stack([torch.stack([torch.stack([pd.read_csv('simulation_output/global/analysis/summary_mats/'+variant+'/'+threshold+'/'+context_type+'/positional_mat.csv', index_col=0, header=0) for context_type in contexts]) for threshold in thresholds]) for variant in variant_order])
    #contextual_tensor = torch.stack([torch.stack([torch.stack([pd.read_csv('simulation_output/global/analysis/summary_mats/'+variant+'/'+threshold+'/'+context_type+'/contextual_mat.csv', index_col=0, header=0) for context_type in contexts]) for threshold in thresholds]) for variant in variant_order])
    torch.save(positional_tensor, 'simulation_output/global/analysis_'+output_info+'/positional_tensor.pt')
    torch.save(contextual_tensor, 'simulation_output/global/analysis_'+output_info+'/contextual_tensor.pt')
    print(positional_tensor.shape, contextual_tensor.shape)
'''concatenate all analysis output for simulation_output'''
def analyze_sims_from_server(variant_order, thresholds, num_sims):
    broken_sims = {threshold:[] for threshold in thresholds}
    averaged_mats = {}
    for sim_type in ['gene_specific', 'global']:
        for context_type in ['full_contexts', 'naive_contexts', 'blind_contexts']:
            for variant in variant_order:
                for threshold in thresholds:
                    averaged_analysis = pd.DataFrame(np.zeros([17,13]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A','0'], columns=columns+['0'])
                    #print(averaged_analysis)
                    for sim_run in range(num_sims):
                        try:
                            averaged_analysis = averaged_analysis.add(pd.read_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)+'/'+str(sim_run)+'_matching_muts_contexts.csv', header=0, index_col=0))
                            #print(sim_type,context_type,variant,threshold)
                        except:
                            broken_sims[threshold].append('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/')
                            break
                        #print(averaged_analysis)
                    averaged_analysis = averaged_analysis.divide(num_sims)
                    averaged_analysis.to_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)+'/averaged_context_matches.csv')
                    averaged_mats['simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)] = averaged_analysis
                    #print('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold))
    print('broken_sims: ', broken_sims)
    '''df = pd.DataFrame(np.zeros([5,1]), index=[0.001, 0.01, 0.1, 0.2, 0.5], columns=['a'])
    for variant in variant_order:
        for sim_type in ['gene_specific', 'global']:
            for context_type in ['full_contexts', 'naive_contexts']:
                var_series = pd.Series(np.zeros([5]), index=[0.001, 0.01, 0.1, 0.2, 0.5], name=variant+'_'+sim_type+'_'+context_type)
                for threshold in thresholds:
                    #print([key for key in averaged_mats.keys() if variant in key and str(threshold) in key and sim_type in key and context_type in key])
                    #print(averaged_mats[[key for key in averaged_mats.keys() if variant in key and str(threshold) in key and 'gene_specific' in key and 'full_contexts' in key][0]])
                    var_series.loc[threshold] = averaged_mats[[key for key in averaged_mats.keys() if variant in key and str(threshold) in key and sim_type in key and context_type in key][0]][0].loc['0','0']
                #print(var_series)
                df = pd.concat([df, var_series], axis=1)
            print(df)'''
    #print(averaged_mats.keys())
    threshold_comp_mat = pd.DataFrame(dtype=float)
    for threshold in thresholds:
        #variant_comp_mat = pd.DataFrame(dtype=float)
        for variant in variant_order:
            threshold_variant_df = pd.Series(name=variant, dtype=float)
            for sim_type in ['global', 'gene_specific']:
                for context_type in ['blind','naive', 'full']:
                    current_mat = averaged_mats[[key for key in averaged_mats.keys() if sim_type in key and context_type in key and '/'+variant in key and str(threshold) in key][0]]
                    position_context_matches = pd.Series([current_mat.loc['0','0'], np.sum(current_mat.iloc[:12,:12].values)], index=['position', 'context'])
                    #print(variant, global_naive)
                    threshold_variant_df = pd.concat([threshold_variant_df, position_context_matches], axis=0)
            #print(threshold_variant_df)
            threshold_variant_df = threshold_variant_df.to_frame()
            threshold_variant_df.columns = [variant+'_'+str(threshold)]
            threshold_comp_mat = pd.concat([threshold_comp_mat, threshold_variant_df], axis=1)
            
    multiindex_cols = pd.MultiIndex.from_product([thresholds, variant_order], names=['threshold', 'variant'])
    print(multiindex_cols)
    threshold_comp_mat.columns = multiindex_cols
    #threshold_comp_mat = threshold_comp_mat.iloc[[],:]
    multiindex_rows = pd.MultiIndex.from_arrays([['global','global','global','global','global','global','gene','gene','gene','gene','gene','gene'],['blind','blind','naive','naive','full','full','blind','blind','naive','naive','full','full']])
    threshold_comp_mat.index = multiindex_rows
    print(threshold_comp_mat)
    print(threshold_comp_mat.loc[:,[(.001,'alpha'), (.01, 'alpha'), (.1, 'alpha')]])
    threshold_comp_mat.to_csv('simulation_output/threshold_comp_mat.csv')
    threshold_comp_mat_2 = pd.DataFrame(dtype=float)
    for variant in variant_order:
        threshold_comp_mat_2 = pd.concat([threshold_comp_mat_2, threshold_comp_mat.loc[:,[(.001,variant), (.01, variant), (.1, variant)]]], axis=1)
    threshold_comp_mat_2.to_csv('simulation_output/comp_by_threshold.csv')

'''concatenate analysis output and calc t-tests between sim types'''
def analyze_sims_from_server_2_deprecated(variant_order, thresholds, num_sims, contexts, threshold_mat=True, t_test_mat=True, output_mut_mat=None, anova=False, reduced_output=False):
    global rows, columns

    if output_mut_mat != None:
        #gwtc
        gwtc_list = [pd.read_csv('sim_ref_data/fourfold_gwtc/'+gene+'.csv', index_col=0, header=0) for gene in ['S','E','M','N','ORF1ab','ORF3a','ORF6','ORF7a','ORF7b','ORF8','ORF10']]
        for gwtc_mat in gwtc_list[1:]:
            gwtc_list[0] += gwtc_mat
        gwtc_mat = gwtc_list[0]
        gwtc_mat = gwtc_mat.iloc[:12,[0,0,0,1,1,1,2,2,2,3,3,3]]
        gwtc_mat.columns = columns

    broken_sims = {threshold:[] for threshold in thresholds} #track if any analysis files are missing or invalid
    positional_context_dict = {} #stores a list for each sim type combination of positional and contextual matches
    t_tests = {} #stores t-statistic and pvalue dataframes for each sim type combination
    averaged_mats_correlation = {} #stores correlation between averaged post-sim mut rate mat and pre-sim mut rate mat
    sim_mats_correlation = {} #stores correlation between averaged sim mut rate mat and pre-sim mut rate mat
    contextual_dist_dict = {}
    #loop through sim type combinations
    for threshold in thresholds:
        print(threshold)
        for variant in variant_order:
            #print(variant)
            variant_threshold_dict = {} #stores mean contextual matches per sim
            for sim_type in ['gene_specific', 'global']:
                #print(sim_type)
                for context_type in contexts:
                    #print(context_type)
                    threshold_tensor = [] #stores each sim in sim_num for this sim type
                    for sim_run in range(num_sims):
                        try:
                            #read in sim data
                            sim_tensor = torch.tensor(pd.read_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)+'/'+str(sim_run)+'_matching_muts_contexts.csv', header=0, index_col=0).values)
                            #store all sim runs for this type
                            threshold_tensor.append(sim_tensor)
                            #calc contextual matches for run
                            #contextual matches = sum of context matches excluding A[X>Y]N mutations
                            if sim_type + '_' + context_type not in variant_threshold_dict.keys():
                                variant_threshold_dict[sim_type+'_'+context_type] = [torch.sum(sim_tensor[:-5,:-1]).item()]
                            else:
                                variant_threshold_dict[sim_type+'_'+context_type].append(torch.sum(sim_tensor[:-5,:-1]).item())
                    
                        except:
                            broken_sims[threshold].append('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/')
                            break
                        
                    #convert list to tensor
                    threshold_tensor = torch.stack(threshold_tensor)
                    #average positional matches across sim runs
                    positional_matches_mean = torch.mean(threshold_tensor[:,-1,-1], dim=(0)).item()
                    #average contextual matches across sim runs
                    #contextual matches = sum of context matches excluding A[X>Y]N mutations
                    contextual_matches_mean = np.mean(variant_threshold_dict[sim_type+'_'+context_type]).item()
                    #store positional and contextual means
                    positional_context_dict['simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)] = [positional_matches_mean, contextual_matches_mean]
                    pd.DataFrame([positional_matches_mean, contextual_matches_mean], index=['positional_matches', 'contextual_matches']).to_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)+'/positional_context_matches.csv')
                    if output_mut_mat != None:
                        #contextual matches
                        averaged_mat = torch.mean(threshold_tensor[:,:12,:12], dim=(0))
                        if output_mut_mat.shape[0] == 12:
                            corr = torch.corrcoef(torch.stack([averaged_mat.flatten(), output_mut_mat.flatten()])).numpy()[0,1]
                        else:
                            corr = torch.corrcoef(torch.stack([averaged_mat.flatten(), output_mut_mat[variant_order.index(variant)].flatten()])).numpy()[0,1]
                        t_stat = corr * np.sqrt((144-2)/(1-(corr**2)))
                        #t_stat = corr / (np.sqrt((1 - (corr**2)) / (144-2)))
                        averaged_mats_correlation['simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)] = [corr,t_stat]
                        pd.DataFrame(averaged_mat, index=rows, columns=columns).to_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)+'/averaged_context_matches.csv')
                        
                        #look at all mutations placed
                        averaged_mat = pd.read_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)+'/sim_mut_rate.csv', index_col=0, header=0).iloc[:12,:]
                        #averaged_mat = averaged_mat / gwtc_mat / num_sims
                        averaged_mat = torch.tensor(averaged_mat.iloc[:12,:].values)
                        
                        if output_mut_mat.shape[0] == 12:
                            corr = torch.corrcoef(torch.stack([averaged_mat.flatten(), output_mut_mat.flatten()])).numpy()[0,1]
                        else:
                            corr = torch.corrcoef(torch.stack([averaged_mat.flatten(), output_mut_mat[variant_order.index(variant)].flatten()])).numpy()[0,1]
                        t_stat = corr * np.sqrt((144-2)/(1-(corr**2)))
                        #t_stat = corr / (np.sqrt((1 - (corr**2)) / (144-2)))
                        sim_mats_correlation['simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/analysis/'+str(threshold)] = [corr,t_stat]

            if anova == True:
                contextual_dist_dict[str(threshold)+'/'+variant] = zip(variant_threshold_dict.keys(), variant_threshold_dict.values())

            if t_test_mat == True:
                #t-test between each type of sim for contextual matches to see if distributions are different
                t_statistics = pd.DataFrame(np.zeros([len(variant_threshold_dict.keys()),len(variant_threshold_dict.keys())]), index=variant_threshold_dict.keys(), columns=variant_threshold_dict.keys())
                p_values = t_statistics.copy()
                #loop through each sim type
                print(str(threshold), variant)
                for first_sim_index, first_sim in variant_threshold_dict.items():
                    for second_sim_index, second_sim in variant_threshold_dict.items():
                        print(first_sim_index, second_sim_index)
                        if first_sim_index != second_sim_index:
                            #test that both distributions are normal
                            #test first_sim dist
                            first_sim_pvalue = stats.normaltest(first_sim).pvalue
                            #test second_sim dist
                            second_sim_pvalue = stats.normaltest(second_sim).pvalue

                            #if normal dists, normal t-test
                            if first_sim_pvalue <0.05 and second_sim_pvalue < 0.05:
                                t_result = stats.ttest_ind(first_sim, second_sim, equal_var=True, alternative='greater')
                            #else welch's t-test
                            else:
                                t_result = stats.ttest_ind(first_sim, second_sim, equal_var=False, alternative='greater')
                            t_statistics.loc[first_sim_index, second_sim_index] = t_result.statistic
                            p_values.loc[first_sim_index, second_sim_index] = t_result.pvalue
                t_tests[str(threshold)+'_'+variant] = [t_statistics, p_values]
    
    print('broken_sims: ', broken_sims)
    
    if t_test_mat == True:
        #convert t-tests dict to df
        t_tests_df = pd.concat([pd.concat([t_tests[key][0], t_tests[key][1]], axis=0) for key in t_tests.keys()], axis=1)
        multiindex_cols = pd.MultiIndex.from_product([thresholds, variant_order, variant_threshold_dict.keys()], names=['threshold', 'variant', 'sim_type'])
        t_tests_df.columns = multiindex_cols
        t_tests_df.to_csv('simulation_output/t_tests.csv')

    if threshold_mat == True:
        #convert positional_context_dict to df
        '''threshold_comp_mat = pd.DataFrame(dtype=float)
        for threshold in thresholds:
            #variant_comp_mat = pd.DataFrame(dtype=float)
            for variant in variant_order:
                threshold_variant_df = pd.Series(name=variant, dtype=float)
                for sim_type in ['global', 'gene_specific']:
                    for context_type in [context.split('_')[0] for context in contexts]:
                        current_mat = positional_context_dict[[key for key in positional_context_dict.keys() if sim_type in key and context_type in key and '/'+variant in key and str(threshold) in key][0]]
                        position_context_matches = pd.Series(current_mat, index=['position', 'context'])
                        threshold_variant_df = pd.concat([threshold_variant_df, position_context_matches], axis=0)
                threshold_variant_df = threshold_variant_df.to_frame()
                threshold_variant_df.columns = [variant+'_'+str(threshold)]
                threshold_comp_mat = pd.concat([threshold_comp_mat, threshold_variant_df], axis=1)
        #update formatting of df for legibility
        multiindex_cols = pd.MultiIndex.from_product([thresholds, variant_order], names=['threshold', 'variant'])
        threshold_comp_mat.columns = multiindex_cols
        #multiindex_rows = pd.MultiIndex.from_arrays([['global','global','global','global','global','global','gene','gene','gene','gene','gene','gene'],['blind','blind','naive','naive','full','full','blind','blind','naive','naive','full','full']])
        print((['global']*len(contexts)*2)+(['gene']*len(contexts)*2))
        print(np.asarray([[context.split('_')[0]]*2 for context in contexts]*2).flatten())
        multiindex_rows = pd.MultiIndex.from_arrays([(['global']*len(contexts)*2)+(['gene']*len(contexts)*2),np.asarray([[context.split('_')[0]]*2 for context in contexts]*2).flatten()])
        threshold_comp_mat.index = multiindex_rows
        threshold_comp_mat.to_csv('simulation_output/threshold_comp_mat.csv')
        threshold_comp_mat_2 = pd.DataFrame(dtype=float)
        for variant in variant_order:
            threshold_comp_mat_2 = pd.concat([threshold_comp_mat_2, threshold_comp_mat.loc[:,[(.001,variant), (.01, variant), (.1, variant)]]], axis=1)
        threshold_comp_mat_2.to_csv('simulation_output/comp_by_threshold.csv')'''
        thresholds = [str(threshold) for threshold in thresholds]
        threshold_comp_mat = pd.DataFrame(index = pd.MultiIndex.from_product([variant_order+['average', 't-test'], ['positional', 'contextual']], names=['variant', 'mut_type']), columns = pd.MultiIndex.from_product([thresholds, ['global', 'gene_specific'], contexts], names=['threshold', 'rate_type', 'contexts'])) #np.zeros([len(variant_order)*2, len(thresholds)*2*len(contexts)])
        #print(threshold_comp_mat)
        for threshold in thresholds:
            for sim_type in ['global','gene_specific']:
                for context_type in contexts:
                    for variant in variant_order:
                        current_mat = positional_context_dict[[key for key in positional_context_dict.keys() if sim_type in key and context_type in key and '/'+variant in key and str(threshold) in key][0]]
                        threshold_comp_mat.loc[(variant,'positional'),(threshold,sim_type,context_type)] = current_mat[0]
                        threshold_comp_mat.loc[(variant,'contextual'),(threshold,sim_type,context_type)] = current_mat[1]
                    threshold_comp_mat.loc[('average','positional'),(threshold,sim_type,context_type)] = threshold_comp_mat.loc[(variant_order,'positional'), (threshold,sim_type,context_type)].mean()
                    threshold_comp_mat.loc[('average','contextual'),(threshold,sim_type,context_type)] = threshold_comp_mat.loc[(variant_order,'contextual'), (threshold,sim_type,context_type)].mean()
                    if context_type != contexts[0]:
                        #test that both distributions are normal
                        #stuff for montecarlo
                        rvs = lambda size: stats.norm.rvs(size=size, random_state=np.random.default_rng())
                        def mc_statistic(x, axis):
                            return stats.skew(x, axis)

                        #test first_dist
                        first_dist = threshold_comp_mat.loc[(variant_order,'contextual'),(threshold,sim_type,context_type)].values.astype(float)
                        if first_dist.shape[0] > 8: #enough data for skewtest
                            first_dist_pvalue = stats.normaltest(first_dist).pvalue
                        else: #not enough data for skewtest
                            first_dist_pvalue = stats.monte_carlo_test(first_dist, rvs, mc_statistic).pvalue
                        #test second_sim dist
                        second_dist = threshold_comp_mat.loc[(variant_order,'contextual'),(threshold,sim_type,contexts[contexts.index(context_type)-1])].values.astype(float)
                        if second_dist.shape[0] > 8:
                            second_dist_pvalue = stats.normaltest(second_dist).pvalue
                        else:
                            second_dist_pvalue = stats.monte_carlo_test(second_dist, rvs, mc_statistic).pvalue
                        
                        #if normal dists, normal t-test
                        if first_dist_pvalue <0.05 and second_dist_pvalue < 0.05:
                            t_result = stats.ttest_ind(first_dist, second_dist, equal_var=True, alternative='greater')
                        #else welch's t-test
                        else:
                            t_result = stats.ttest_ind(first_dist, second_dist, equal_var=False, alternative='greater')
                        threshold_comp_mat.loc[('t-test','positional'),(threshold,sim_type,context_type)] = t_result.statistic
                        threshold_comp_mat.loc[('t-test','contextual'),(threshold,sim_type,context_type)] = t_result.pvalue
        #remove excess output info
        if reduced_output == True:
            #print(threshold_comp_mat.columns)
            drop_labels = []
            for threshold in thresholds:
                if 'blind_contexts' in contexts:
                    for label in [['global','blind_contexts'], ['gene_specific','blind_contexts'], ['gene_specific','naive_contexts']]:
                        drop_labels.append((threshold, label[0], label[1]))
                else:
                    for label in [['gene_specific', 'naive_contexts']]:
                        drop_labels.append((threshold, label[0], label[1]))
            '''if 'blind_contexts' in contexts:
                drop_labels = [(threshold,label[0],label[1]) for label in [['global','blind_contexts'], ['gene_specific','blind_contexts'], ['gene_specific','naive_contexts']]]
            else:
                drop_labels = [(threshold,'gene_specific','naive_contexts')]'''
            #print(drop_labels)
            threshold_comp_mat.drop(labels=drop_labels, axis=1, inplace=True)
            for threshold in thresholds:
                keys = [(threshold, 'global','naive_contexts'), (threshold, 'global','full_contexts'), (threshold, 'gene_specific','full_contexts')]
                #print(keys)
                for index, key in enumerate(keys[1:]):
                    #test that both distributions are normal
                    #test first_dist
                    first_dist = threshold_comp_mat.loc[(variant_order,'contextual'),key].values.astype(float)
                    first_dist_pvalue = stats.normaltest(first_dist).pvalue
                    #test second_sim dist
                    second_dist = threshold_comp_mat.loc[(variant_order,'contextual'),keys[index]].values.astype(float)
                    second_dist_pvalue = stats.normaltest(second_dist).pvalue
                    print(first_dist, second_dist)

                    #if normal dists, normal t-test
                    if first_dist_pvalue <0.05 and second_dist_pvalue < 0.05:
                        t_result = stats.ttest_ind(first_dist, second_dist, equal_var=True, alternative='greater')
                    #else welch's t-test
                    else:
                        t_result = stats.ttest_ind(first_dist, second_dist, equal_var=False, alternative='greater')
                    threshold_comp_mat.loc[('t-test','positional'),key] = t_result.statistic
                    threshold_comp_mat.loc[('t-test','contextual'),key] = t_result.pvalue

                    '''#if normal dists, normal t-test
                    if first_dist_pvalue <0.05 and second_dist_pvalue < 0.05:
                        t_result = stats.ttest_ind(first_dist, second_dist, equal_var=True, alternative='less')
                    #else welch's t-test
                    else:
                        t_result = stats.ttest_ind(first_dist, second_dist, equal_var=False, alternative='less')
                    threshold_comp_mat.loc[('t-test','positional'),keys[index]] = t_result.statistic
                    threshold_comp_mat.loc[('t-test','contextual'),keys[index]] = t_result.pvalue'''
                
        threshold_comp_mat.to_csv('simulation_output/threshold_comp_mat.csv')
        thresholds = [float(threshold) for threshold in thresholds]
        #threshold_comp_mat.loc[(variant_order,'contextual'),([0.00001, 0.001, 0.1],['global','gene_specific'],['naive_contexts', 'full_contexts'])].to_csv('simulation_output/threshold_comp_mat_reduced.csv')
        #print(threshold_comp_mat_reduced)
        #threshold_comp_mat_reduced.drop((0.1,'gene_specific','naive_contexts'), axis=1, inplace=True).to_csv('simulation_output/threshold_comp_mat_reduced.csv')
    
    if output_mut_mat != None:
        #create df of pre-sim mut rates and post-sim mut rates
        #have dict of [corr,t_stat]
        #want matrix of corrs and t_stats
        #rows = variants
        #cols = threshold,sim_type/context_type
        print('yep')
        sim_combos = ['gene_specific_full_contexts', 'gene_specific_naive_contexts', 'gene_specific_blind_contexts', 'global_full_contexts', 'global_naive_contexts', 'global_blind_contexts']
        multiindex_cols = pd.MultiIndex.from_product([[str(threshold) for threshold in thresholds], sim_combos], names=['threshold', 'sim'])
        multiindex_rows = pd.MultiIndex.from_product([['correlation', 't_stat'], variant_order], names=['stat', 'variant'])
        output_mut_mat_corr_df = pd.DataFrame(np.zeros([2*len(variant_order)*len(thresholds)*len(sim_combos)]).reshape([2*len(variant_order),len(thresholds)*len(sim_combos)]), index=multiindex_rows, columns=multiindex_cols)
        print(averaged_mats_correlation)
        #print(output_mut_mat_corr_df)
        #contextual mut rates
        for key, corr_t_stat in averaged_mats_correlation.items():
            print(key, corr_t_stat)
            key_split = key.split('/')
            variant = key_split[3]
            threshold = key_split[-1]
            sim = key_split[1]+'_'+key_split[2]
            output_mut_mat_corr_df.loc[('correlation', variant), (threshold,sim)] = corr_t_stat[0]
            output_mut_mat_corr_df.loc[('t_stat', variant), (threshold,sim)] = corr_t_stat[1]
        output_mut_mat_corr_df.to_csv('simulation_output/post_sim_corr.csv')

        #look at all mutations placed not just correct ones
        for key, corr_t_stat in sim_mats_correlation.items():
            print(key, corr_t_stat)
            key_split = key.split('/')
            variant = key_split[3]
            threshold = key_split[-1]
            sim = key_split[1]+'_'+key_split[2]
            output_mut_mat_corr_df.loc[('correlation', variant), (threshold,sim)] = corr_t_stat[0]
            output_mut_mat_corr_df.loc[('t_stat', variant), (threshold,sim)] = corr_t_stat[1]
        output_mut_mat_corr_df.to_csv('simulation_output/sim_mut_rate_corr.csv')

    if anova == True:
        print(contextual_dist_dict)
        #dict of each threshold/sim_type/context/variant combo
        contextual_df = pd.DataFrame(np.zeros([len(thresholds)*2*len(contexts)*len(variant_order), num_sims]), index=pd.MultiIndex.from_product([[str(threshold) for threshold in thresholds],['global', 'gene_specific'],contexts,variant_order]))
        for threshold in thresholds:
            for variant in variant_order:
                variant_list = list(contextual_dist_dict[str(threshold)+'/'+variant])
                for sim_context in variant_list:
                    df_index = [str(threshold)]
                    if 'global' in sim_context[0]:
                        df_index.append('global')
                    elif 'gene_specific' in sim_context[0]:
                        df_index.append('gene_specific')
                    if 'blind' in sim_context[0]:
                        df_index.append('blind_contexts')
                    if 'naive' in sim_context[0]:
                        df_index.append('naive_contexts')
                    if 'full' in sim_context[0]:
                        df_index.append('full_contexts')
                    df_index.append(variant)
                    #print(df_index, len(sim_context[1]))
                    contextual_df.loc[tuple(df_index)] = sim_context[1]

        anova_df = pd.DataFrame(np.zeros([len(variant_order)*2+2, len(thresholds)*2*len(contexts)]), index=pd.MultiIndex.from_product([['f']+variant_order, ['mean','std']], names=['variant', 'stat']), columns=pd.MultiIndex.from_product([[str(threshold) for threshold in thresholds],['global', 'gene_specific'],contexts], names=['threshold','sim_type','context_type']))
        for threshold in thresholds:
            for sim_type in ['global', 'gene_specific']:
                for context_type in contexts:
                    anova = stats.f_oneway(*contextual_df.loc[(str(threshold),sim_type,context_type)].values)
                    anova_df.loc[('f','mean'), (str(threshold),sim_type,context_type)] = anova[0]
                    anova_df.loc[('f','std'), (str(threshold),sim_type,context_type)] = anova[1]
                    for variant in variant_order:
                        mean = np.mean(contextual_df.loc[(str(threshold),sim_type,context_type,variant)].values)
                        std = np.std(contextual_df.loc[(str(threshold),sim_type,context_type,variant)].values)
                        anova_df.loc[(variant), (str(threshold),sim_type,context_type)] = [mean, std]
        anova_df.to_csv('simulation_output/anova_df.csv')

'''normalize a simulation's output by the number of mutations placed and the number of possible matches'''
def normalize_threshold_mat_by_placed_and_matches(threshold_mat, mut_count):
    normalized_threshold_mat = threshold_mat.copy()

    #remove higher-frequency mutations from lower-frequency counts
    thresholds = np.unique(threshold_mat.columns.get_level_values(0))
    thresholds = np.sort([float(threshold) for threshold in thresholds])[::-1] #fix ordering
    thresholds = [str(threshold) for threshold in thresholds]
    
    contexts = np.unique(threshold_mat.columns.get_level_values(1))
    for threshold_index, threshold in enumerate(thresholds):
        print(threshold)
        if threshold_index > 0:
            for context in contexts:
                normalized_threshold_mat.loc[:,(threshold,context)] = threshold_mat.loc[:,(threshold,context)] - threshold_mat.loc[:,(thresholds[threshold_index-1],context)]
    
    #read in possible matches for each variant,match_type,threshold
    possible_matches = pd.read_csv('simulation_output/final_info/final_sim_analysis/possible_matches.csv', header=0)
    possible_matches.drop([col for col in possible_matches.columns][:2], axis=1, inplace=True)
    possible_matches.index = threshold_mat.index

    print(normalized_threshold_mat)
    print(possible_matches)
    #normalize match_counts by possible matches and number of mutations placed
    for threshold in possible_matches.columns:
        normalized_threshold_mat.loc[:,threshold] = threshold_mat.loc[:,threshold].to_numpy() / np.repeat(possible_matches.loc[:,threshold].to_numpy(), 3).reshape(-1,3) / mut_count
        print(np.repeat(possible_matches.loc[:,threshold].to_numpy(), 3).reshape(-1,3))
    return normalized_threshold_mat



'''concatenate analysis output and calc t-tests between sim types'''
def analyze_sims_from_server_3(variant_order, thresholds, contexts, analysis_threshold, analysis_variant, output_info):
    
    #gene wide triplet counts for non-overlapping genes
    gwtc_dict = {gene[:-4]:pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/'+gene, index_col=0, header=0) for gene in os.listdir('sim_ref_data/fourfold_gwtc/triplets') if gene[:-4]=='.csv'}

    positional_context_dict = {} #stores a list for each sim type combination of positional and contextual matches
    t_tests = {} #stores t-statistic and pvalue dataframes for each sim type combination
    averaged_mats_correlation = {} #stores correlation between averaged post-sim mut rate mat and pre-sim mut rate mat
    sim_mats_correlation = {} #stores correlation between averaged sim mut rate mat and pre-sim mut rate mat
    contextual_dist_dict = {}
    
    #check if analysis has been prepared
    analysis_folder = 'analysis_'+output_info
    if not os.path.exists('simulation_output/global/'+analysis_folder+'/'):
        #variant_order, thresholds, analysis_threshold, analysis_variant, contexts
        analyze_mutations_using_mut_dicts(variant_order, thresholds, analysis_threshold, analysis_variant, contexts, output_info)
    thresholded_mat = pd.read_csv('simulation_output/global/'+analysis_folder+'/thresholded_mat.csv', index_col=None, header=[0,1])
    thresholded_mat.index = pd.MultiIndex.from_tuples([(thresholded_mat.iloc[index,0], thresholded_mat.iloc[index,1]) for index in range(len(thresholded_mat.index))])
    thresholded_mat.drop(thresholded_mat.columns.to_numpy()[:2], axis=1, inplace=True)
    analysis_df = thresholded_mat.copy()
    print(analysis_df)

    #t-test between context_types at each threshold to show dists are different for each set of sims
    positional_match_tensor = torch.load('simulation_output/global/'+analysis_folder+'/positional_tensor.pt')
    contextual_match_tensor = torch.load('simulation_output/global/'+analysis_folder+'/contextual_tensor.pt')
    #shapes are (num_variants,num_thresholds,num_contexts,[16,12],12)
    tensor_shape = positional_match_tensor.shape
    print(tensor_shape)

    #read in thresholded_mut_counts
    thresholded_mut_counts = pd.read_csv('sim_ref_data/thresholded_mut_counts.csv', index_col=0, header=0)
    #normalize analysis_df by mut counts
    for index in analysis_df.index:
        for col in analysis_df.columns:
            analysis_df.loc[index,col] /= thresholded_mut_counts.loc[index[0],col[0]]
    print(analysis_df)

    ttest_df = pd.DataFrame(np.zeros([2, tensor_shape[0]*tensor_shape[1]]), index=['t-stat','p-value'], columns=pd.MultiIndex.from_product([thresholds,contexts]))

    #loop through thresholds
    for threshold_index, threshold in enumerate(thresholds):
        #loop through context_types
        for context_index, context_type in enumerate(contexts):
            if context_index > 0:
                #first set of contextual matches
                first_dist = analysis_df.loc[(variant_order,'contextual_matches'),(threshold,context_type)]
                first_dist_pvalue = stats.normaltest(first_dist).pvalue
                #second set of contextual matches
                second_dist = analysis_df.loc[(variant_order,'contextual_matches'),(threshold,contexts[context_index-1])]
                second_dist_pvalue = stats.normaltest(second_dist).pvalue

                #if normal dists, normal t-test
                if first_dist_pvalue <0.05 and second_dist_pvalue < 0.05:
                    t_result = stats.ttest_ind(first_dist, second_dist, equal_var=True, alternative='greater')
                #else welch's t-test
                else:
                    t_result = stats.ttest_ind(first_dist, second_dist, equal_var=False, alternative='greater')
                ttest_df.loc['t-stat',(threshold,context_type)] = t_result.statistic
                ttest_df.loc['p-value',(threshold,context_type)] = t_result.pvalue
    ttest_df.index = pd.MultiIndex.from_product([[''],['t-stat','p-value']])
    print(f'ttest df {ttest_df}')

    '''
    #anova between context_types
    anova_df = pd.DataFrame(np.zeros([4, tensor_shape[0]*tensor_shape[1]]), index=pd.MultiIndex.from_product([['positional','contextual'],['f-stat','p-value']]), columns=pd.MultiIndex.from_product([thresholds,contexts]))
    variants = [index[0] for index in analysis_df.index][::2]
    print(variants)
    for threshold in thresholds:
        for match_index, match_type in enumerate(['positional','contextual']):
            vals = analysis_df.loc[:,(threshold,)].to_numpy()[match_index::2].T
            anova = stats.f_oneway(*vals)
            anova_df.loc[(match_type,'f-stat'),(threshold,contexts[-1])] = anova[0]
            anova_df.loc[(match_type,'p-value'),(threshold,contexts[-1])] = anova[1]

    analysis_df = pd.concat([analysis_df, ttest_df, anova_df], axis=0).to_csv('simulation_output/global/analysis/thresholded_mat_complete.csv')
    '''

    '''
    #anova
    #read in positional and contextual match counts for each sim run in sim combination
    distributions = pd.read_csv('simulation_output/global/analysis/distributions.csv', index_col=[0,1,2,3], header=0)
    anova_df = pd.DataFrame(np.zeros([len(variant_order)*2+2, len(thresholds)*len(contexts)]), index=pd.MultiIndex.from_product([['f']+variant_order, ['mean','std']], names=['variant', 'stat']), columns=pd.MultiIndex.from_product([thresholds,contexts], names=['threshold','context_type']))
    for threshold in thresholds: #loop through thresholds
        for context_type in contexts: #loop through contexts
            #perform anova across variants
            anova = stats.f_oneway(*distributions.loc[(float(threshold),context_type,'contextual_matches',)].values)
            anova_df.loc[('f','mean'), (threshold,context_type)] = anova[0]
            anova_df.loc[('f','std'), (threshold,context_type)] = anova[1]
            #store means and stds, can be checked against thresholded_output_mat
            for variant in variant_order:
                mean = np.mean(distributions.loc[(float(threshold),context_type,'contextual_matches',variant)].values)
                std = np.std(distributions.loc[(float(threshold),context_type,'contextual_matches',variant)].values)
                anova_df.loc[(variant), (threshold,context_type)] = [mean, std]
    print(f'anova df {anova_df}')
    anova_df.to_csv('simulation_output/global/analysis/anova_mat.csv')
    '''

    
    #vars = ['all','alpha','delta']
    #[5,3,100,17,16,12]
    fig, axs = plt.subplots(figsize=(30,12), dpi=200, nrows=2, ncols=5)
    fig_index = 0
    pos_con_df = pd.DataFrame()
    chi_square_df = pd.DataFrame(np.zeros([4,2]), index=pd.MultiIndex.from_product([['positional','contextual'],['t-stat','p-value']]), columns=['blind->naive','naive->full'])
    #loop through contexts
    for context_type in range(positional_match_tensor.shape[1]):
        pos_mat = torch.mean(positional_match_tensor[0,context_type,:,0], dim=(0))
        con_mat = torch.mean(contextual_match_tensor[0,context_type,:,0], dim=(0))
        #average positional matches and contextual matches for 'all' variant of context_type and threshold of '5e-05'
        pos_con_df = pd.concat([pos_con_df, pd.concat([pd.DataFrame(pos_mat, index=rows_figs, columns=columns_figs), pd.DataFrame(con_mat, index=rows_figs, columns=columns_figs)], axis=0)], axis=1)
        sns.heatmap(pos_mat, ax=axs[0,fig_index], annot=True, cbar=False, yticklabels=rows_figs, xticklabels=columns_figs, cmap='Greys')
        sns.heatmap(con_mat, ax=axs[1,fig_index], annot=True, cbar=False, yticklabels=rows_figs, xticklabels=columns_figs, cmap='Greys')
        fig_index+=1
        #create difference matrix between context_types to compare blind/naive and naive/full
        if context_type != 2:
            #subtract averaged matrices
            pos_mat = torch.mean(positional_match_tensor[0,context_type+1,:,0], dim=(0)) - torch.mean(positional_match_tensor[0,context_type,:,0], dim=(0))
            con_mat = torch.mean(contextual_match_tensor[0,context_type+1,:,0], dim=(0)) - torch.mean(contextual_match_tensor[0,context_type,:,0], dim=(0))
            pos_con_df = pd.concat([pos_con_df, pd.concat([pd.DataFrame(pos_mat, index=rows_figs, columns=columns_figs), pd.DataFrame(con_mat, index=rows_figs, columns=columns_figs)], axis=0)], axis=1)
            sns.heatmap(pos_mat, ax=axs[0,fig_index], annot=True, cbar=False, yticklabels=rows_figs, xticklabels=columns_figs, cmap='coolwarm', center=0)
            sns.heatmap(con_mat, ax=axs[1,fig_index], annot=True, cbar=False, yticklabels=rows_figs, xticklabels=columns_figs, cmap='coolwarm', center=0)
            fig_index+=1
            #perform t-test to determine if context_type distributions are different
            positional_t_test = stats.ttest_ind(torch.mean(positional_match_tensor[0,context_type+1,:,0], dim=(0)).reshape(-1), torch.mean(positional_match_tensor[0,context_type,:,0], dim=(0)).reshape(-1))
            contextual_t_test = stats.ttest_ind(torch.mean(contextual_match_tensor[0,context_type+1,:,0], dim=(0)).reshape(-1), torch.mean(contextual_match_tensor[0,context_type,:,0], dim=(0)).reshape(-1))
            print(positional_t_test)
            chi_square_df.loc[('positional','t-stat'),chi_square_df.columns[context_type]] = positional_t_test[0]
            chi_square_df.loc[('positional','p-value'),chi_square_df.columns[context_type]] = positional_t_test[1]
            chi_square_df.loc[('contextual','t-stat'),chi_square_df.columns[context_type]] = contextual_t_test[0]
            chi_square_df.loc[('contextual','p-value'),chi_square_df.columns[context_type]] = contextual_t_test[1]
            
    plt.savefig('simulation_output/final_info/pos_con.png')
    plt.close()
    pos_con_df.index = pd.MultiIndex.from_product([['positional','contextual'],rows_figs])
    pos_con_df.columns = pd.MultiIndex.from_product([['blind','blind->naive','naive','naive->full','full'], columns_figs])
    pos_con_df.to_csv('simulation_output/final_info/pos_con_df.csv')
    chi_square_df.to_csv('simulation_output/final_info/pos_con_df_t_tests.csv')
    '''fig, axs = plt.subplots(figsize=(20,20), nrows=positional_match_tensor.shape[-3], ncols=positional_match_tensor.shape[1])
    for var_index in range(positional_match_tensor.shape[-3]):
        for context_type in range(positional_match_tensor.shape[1]):
            print(f'{variants[var_index]}, {contexts[context_type]}')
            sns.heatmap(torch.mean(positional_match_tensor[0,context_type,:,var_index], dim=(0)), ax=axs[var_index,context_type], annot=True, cbar=False, yticklabels=rows_figs, xticklabels=columns_figs, cmap='Greys')
            axs[var_index,context_type].set_title(variants[var_index]+contexts[context_type])
    plt.savefig('simulation_output/positional_mats.png')
    for var_index in range(contextual_match_tensor.shape[-3]):
        for context_type in range(contextual_match_tensor.shape[1]):
            sns.heatmap(torch.mean(contextual_match_tensor[0,context_type,:,var_index], dim=(0)), ax=axs[var_index,context_type], annot=True, cbar=False, yticklabels=rows_figs, xticklabels=columns_figs, cmap='Greys')
            axs[var_index,context_type].set_title(variants[var_index]+contexts[context_type])
    plt.savefig('simulation_output/contextual_mats.png')'''


    '''
    updated methodology
        - doing anova to see if blind/naive/full are different distributions
        - then doing tukey to see which ones
    '''
    
    #anova between context_types
    anova_df = pd.DataFrame(np.zeros([4, tensor_shape[0]*tensor_shape[1]]), index=pd.MultiIndex.from_product([['positional','contextual'],['f-stat','p-value']]), columns=pd.MultiIndex.from_product([thresholds,contexts]))
    tukey_df = pd.DataFrame(np.zeros([4, tensor_shape[0]*tensor_shape[1]]), index=pd.MultiIndex.from_product([['positional','contextual'],['stat (g0:g1,g0:g2,g1:g2)','p-value']]), columns=pd.MultiIndex.from_product([thresholds,contexts]))
    variants = [index[0] for index in analysis_df.index][::2]
    #print(variants)
    for threshold in thresholds:
        for match_index, match_type in enumerate(['positional','contextual']):
            vals = analysis_df.loc[:,(threshold,)].to_numpy()[match_index::2].T
            anova = stats.f_oneway(*vals)
            anova_df.loc[(match_type,'f-stat'),(threshold,contexts[-1])] = anova[0]
            anova_df.loc[(match_type,'p-value'),(threshold,contexts[-1])] = anova[1]

            tukey = stats.tukey_hsd(*vals)
            #print(tukey)
            for i, indexer in enumerate([(0,1),(0,2),(1,2)]):
                #print(tukey.statistic[indexer])
                tukey_df.loc[(match_type,'stat (g0:g1,g0:g2,g1:g2)'),(threshold,contexts[i])] = tukey.statistic[indexer]
                tukey_df.loc[(match_type,'p-value'),(threshold,contexts[i])] = tukey.pvalue[indexer]


    pd.concat([analysis_df, ttest_df, anova_df, tukey_df], axis=0).to_csv('simulation_output/global/'+analysis_folder+'/thresholded_mat_complete.csv')
    

    #going to normalize the number of matching mutations by the number of mutations placed in each iteration and the total number of possible matches at that threshold
    normalize_threshold_mat_by_placed_and_matches(thresholded_mat, int(re.search(r'\d+', analysis_folder).group(0))).to_csv('simulation_output/global/'+analysis_folder+'/thresholded_mat_normalized.csv')

#show base naive simulation performance and change for naive>tstv and naive>full
def reformat_pos_con_fig():
    #read in pos_con df
    pos_con_df = pd.read_csv('simulation_output/final_info/pos_con_df.csv', index_col=[0,1], header=[0,1])
    print(pos_con_df)

    '''normalized performance'''
    full_sums = [np.sum(pos_con_df.loc[(match_type,rows_figs[:12]),'full'].to_numpy()) for match_type in ['positional','contextual']]
    full_maxs = [np.max(pos_con_df.loc[match_type,'full'].to_numpy()[:12,:12]) / full_sums[['positional','contextual'].index(match_type)] for match_type in ['positional','contextual']]
    full_mins = [np.min(pos_con_df.loc[match_type,'full'].to_numpy()[:12,:12]) / full_sums[['positional','contextual'].index(match_type)] for match_type in ['positional','contextual']]
    #create figure
    fig = plt.figure(layout='constrained', dpi=300, figsize=(36,20))
    cols = gridspec.GridSpec(1,6, figure=fig, wspace=.1, hspace=.1, width_ratios=[0.31,.02,0.3,.02,0.32,0.02])
    rows_naive = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=cols[0,0], height_ratios=[0.5,.5], hspace=.07)
    rows_tstv = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=cols[0,2], height_ratios=[0.5,.5], hspace=.07)
    rows_full = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=cols[0,4], height_ratios=[0.5,.5], hspace=.07)
    ax_cbar = fig.add_subplot(cols[0,5])
    axs_naive = [fig.add_subplot(rows_naive[i,0]) for i in range(2)]
    axs_tstv = [fig.add_subplot(rows_tstv[i,0]) for i in range(2)]
    axs_full = [fig.add_subplot(rows_full[i,0]) for i in range(2)]
    axs = np.array([axs_naive,axs_tstv,axs_full])

    #rename model types
    models = ['Naive','TSTV','Context-dependent']
    matches = ['Positional Matches','Contextual Matches']
    #loop through sim types
    for match_index, match_type in enumerate(['positional','contextual']):
        for model_index, model_type in enumerate(['blind','naive','full']):
            mat = pos_con_df.loc[match_type,model_type].to_numpy()[:12,:12] / full_sums[match_index]
            annot = np.round(mat, 2)
            sns.heatmap(mat, ax=axs[model_index,match_index], cmap='Greys', linecolor='grey', linewidth=.5, cbar=False, annot=annot, annot_kws={'fontsize':20}, vmin=np.min(full_mins), vmax=np.max(full_maxs))
            axs[model_index,match_index].set_yticklabels([row.replace('-','') for row in rows_figs][:12], rotation='horizontal', fontsize=22) #update rows
            axs[model_index,match_index].set_xticklabels(columns_figs, fontsize=22, rotation=45) #update columns
            axs[model_index,match_index].set_title(models[model_index], fontsize=36) #update title
            if model_index == 0:
                axs[model_index,match_index].set_ylabel(matches[match_index], fontsize=36, rotation='vertical')
    norm = colors.Normalize(vmin=np.min(full_mins), vmax=np.max(full_maxs))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.Greys), cax=ax_cbar, orientation='vertical')
    interval = np.arange(0,np.round(np.max(full_maxs)+.01, 2), .01)
    cbar.set_ticklabels(interval, fontsize=28)
    plt.savefig('simulation_output/final_info/final_figs/supp_figs/pos_con_sim_performance_normalized.png')
    plt.close()

    '''comparative performance'''
    full_maxs = [[np.max(pos_con_df.loc[match_type,model_type].to_numpy()[:12,:12] - pos_con_df.loc[match_type,'blind'].to_numpy()[:12,:12]) for match_type in ['positional','contextual'] for model_type in ['full','naive']]]
    full_mins = [[np.min(pos_con_df.loc[match_type,model_type].to_numpy()[:12,:12] - pos_con_df.loc[match_type,'blind'].to_numpy()[:12,:12]) for match_type in ['positional','contextual'] for model_type in ['full','naive']]]
    max = np.max(full_maxs)
    min = -1 * max
    print(min,max)
    #create figure
    fig = plt.figure(layout='constrained', dpi=300, figsize=(36,20))
    cols = gridspec.GridSpec(1,6, figure=fig, wspace=.1, hspace=.1, width_ratios=[0.31,.02,0.3,.02,0.32,0.02])
    rows_naive = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=cols[0,0], height_ratios=[0.5,.5], hspace=.07)
    rows_tstv = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=cols[0,2], height_ratios=[0.5,.5], hspace=.07)
    rows_full = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=cols[0,4], height_ratios=[0.5,.5], hspace=.07)
    ax_cbar = fig.add_subplot(cols[0,5])
    axs_naive = [fig.add_subplot(rows_naive[i,0]) for i in range(2)]
    axs_tstv = [fig.add_subplot(rows_tstv[i,0]) for i in range(2)]
    axs_full = [fig.add_subplot(rows_full[i,0]) for i in range(2)]
    axs = np.array([axs_naive,axs_tstv,axs_full])
    
    #loop through sim types
    for match_index, match_type in enumerate(['positional','contextual']):
        for model_index, model_type in enumerate(['blind','naive','full']):
            if model_index == 0:
                mat = pos_con_df.loc[match_type,model_type].to_numpy()[:12,:12]
                annot = np.round(mat, 2)
                sns.heatmap(mat, ax=axs[model_index,match_index], cmap='Greys', linecolor='grey', linewidth=.5, cbar=False, annot=annot, annot_kws={'fontsize':18})
                axs[model_index,match_index].set_ylabel(matches[match_index], fontsize=36, rotation='vertical')
            else:
                mat = pos_con_df.loc[match_type,model_type].to_numpy()[:12,:12] - pos_con_df.loc[match_type,'blind'].to_numpy()[:12,:12]
                annot = np.round(pos_con_df.loc[match_type,model_type].to_numpy()[:12,:12], 2)
                sns.heatmap(mat, ax=axs[model_index,match_index], cmap='coolwarm', linecolor='grey', linewidth=.5, cbar=False, annot=annot, annot_kws={'fontsize':18}, center=0, vmin=min, vmax=max)
            axs[model_index,match_index].set_yticklabels([row.replace('-','') for row in rows_figs][:12], rotation='horizontal', fontsize=22) #update rows
            axs[model_index,match_index].set_xticklabels(columns_figs, fontsize=22, rotation=45) #update columns
            axs[model_index,match_index].set_title(models[model_index], fontsize=36) #update title
    norm = colors.Normalize(vmin=min, vmax=max)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), cax=ax_cbar, orientation='vertical')
    interval = [-1+min,-15,-10,-5,0,5,10,15,max+1]
    cbar.set_ticklabels(interval, fontsize=28)
    plt.savefig('simulation_output/final_info/final_figs/supp_figs/pos_con_sim_performance_comparison.png')
    plt.close()
    

#analyze simulations to pull the gene-specific information
def analyze_sims_genes(thresholds, contexts, analysis_threshold, analysis_variant, gene_dict, sim_folder, sim_output_flag=False):
    #looking to loop through each sim type and create list of mutations at each gene
    
    #convert start and end positions of each gene to range for easier indexing of mutations
    gene_dict_ranges = {}
    for key, value in gene_dict.items():
        if len(value) > 2:
            vals = np.array([])
            for sub_position in value:
                vals = np.append(vals, np.arange(sub_position[0],sub_position[1]))
            gene_dict_ranges[key] = vals
        else:
            gene_dict_ranges[key] = np.arange(value[0],value[1])

    if not os.path.exists('simulation_output/global/'+sim_folder+'/genes'):
        os.mkdir('simulation_output/global/'+sim_folder+'/genes')

    #table with binary presence of mutation at each site
    if sim_output_flag:
        sim_output_table = np.array([])
    
    for context_type in contexts:
        if not os.path.exists('simulation_output/global/'+sim_folder+'/genes/'+context_type):
            os.mkdir('simulation_output/global/'+sim_folder+'/genes/'+context_type)

        sim_hist = pd.read_csv('simulation_output/context_counts/genome_wide_list.csv', index_col=0, header=0)
        sim_hist['contextual_matches'] = np.zeros([sim_hist.shape[0]])

        #read in reference matches for min threshold for 'all' variant
        ref_muts = pd.read_csv('sim_ref_data/0(all)_full_clade/reference_mutations/5e-05_reference_mutations.csv', index_col=0, header=0, dtype={'position':np.float64})

        #loop through each simulation
        for sim in os.listdir('simulation_output/global/'+context_type+'/'+analysis_variant+'/mut_dicts/'+analysis_threshold):
            #create sim output array
            if sim_output_flag and context_type == 'full_contexts':
                sim_output_array = np.zeros([sim_hist.shape[0]])

            #read in simulated mutation data
            sim_muts = pd.read_csv('simulation_output/global/'+context_type+'/'+analysis_variant+'/mut_dicts/'+analysis_threshold+'/'+sim, index_col=0, header=0)
            #drop metadata row
            sim_muts.drop(99999999, axis=0, inplace=True)
            #ensure that position is correct type
            sim_muts['position'] = sim_muts['position'].astype(int)
            #convert to numpy
            sim_muts = sim_muts.to_numpy()
            #loop through mutations
            for index in range(sim_muts.shape[0]):
                original, position, mutation = sim_muts[index]
                #print(original, position, mutation)
                sim_hist.loc[position-1,mutation[1]] += 1 #increment mutation
                #matching means that the context that was mutated matched the original triplet so there were no neighboring mutations affecting the mutation placement
                if original == sim_hist.loc[position-1,'triplet']:
                    sim_hist.loc[position-1,'matching'] += 1
                    if position in ref_muts['position']:
                        if mutation[1] in ref_muts.loc[ref_muts['position']==position]['mut']:
                            sim_hist.loc[position-1,'contextual_matches'] += 1
                if sim_output_flag and context_type == 'full_contexts':
                    sim_output_array[position-1] = 1
            if sim_output_flag and context_type == 'full_contexts':
                if len(sim_output_table)==0:
                    sim_output_table = np.array([sim_output_array])
                else:
                    sim_output_table = np.append(sim_output_table, sim_output_array)
        
        #total genome hist
        sim_hist.to_csv('simulation_output/global/'+sim_folder+'/genes/'+context_type+'/total_hist.csv')
        #seperate gene hists
        for gene, positions in gene_dict_ranges.items():
            sim_hist.loc[sim_hist.index.isin(positions)].to_csv('simulation_output/global/'+sim_folder+'/genes/'+context_type+'/'+gene+'_hist.csv')
        #save sim output table
        if sim_output_flag and context_type == 'full_contexts':
            sim_output_table = sim_output_table.reshape(29903,-1)
            sim_output_table = pd.DataFrame(sim_output_table, index=np.arange(1,29904), columns=np.arange(1,sim_output_table.shape[1]+1))
            avg_matches_series = sim_hist.loc[:,['T','G','C','A']].sum(axis=1) / sim_output_table.shape[1]
            avg_matches_series.name = 'average_matches'
            avg_matches_series.index = np.arange(1,29904)
            sim_output_table = pd.concat([sim_output_table,avg_matches_series], axis=1)
            sim_output_table.to_csv('simulation_output/final_info/final_tables/supp_tables/sim_matching_table.csv')

#create histogram for sim analysis result
def gen_sim_hist(gene, context_type, sim_folder):
    #read in simulation output for gene
    sim_hist = pd.read_csv('simulation_output/global/'+sim_folder+'/genes/'+context_type+'/'+gene+'_hist.csv', index_col=0, header=0)
    #create figure
    fig = plt.figure(layout='constrained', dpi=200, figsize=(15,15))
    grid = gridspec.GridSpec(5,1, figure=fig, wspace=.1, hspace=.1)
    ax0,ax1,ax2,ax3,ax4 = [fig.add_subplot(grid[i,0]) for i in range(5)]

    #total hist
    #creating bins of size 10 and summing mut counts for all resultant nucleotides
    #total_hist = {position: np.sum(sim_hist.loc[position:position+10, ['T','G','C','A']]) for position in range(sim_hist.index[0], sim_hist.index[-1], 10)}
    #total_hist = pd.DataFrame.from_dict({index: np.sum(sim_hist.loc[index, ['T','G','C','A']]) for index in sim_hist.index}, orient='index')
    #print(total_hist)
    #sns.histplot(sim_hist.loc[:,['T','G','C','A']].T, ax=ax0, binwidth=20)
    #hist = pd.DataFrame()
    hist = np.array([])
    for index in sim_hist.index:
        #print(index)
        #position_df = pd.DataFrame()
        for col in ['T','G','C','A']:
            hist = np.append(hist, np.tile([col,index], int(sim_hist.loc[index,col])))
            #for iter in range(int(sim_hist.loc[index, col])):
            #    col_series = pd.Series([col,index], index=['nuc','position'])
            #    position_df = pd.concat([position_df, col_series], axis=1)
        #hist = pd.concat([hist, position_df], axis=1)
    hist = hist.reshape(-1,2)
    print(hist, hist.shape)
    sns.histplot(pd.DataFrame(hist, columns=['nuc','position']), x='position', hue='nuc', multiple='stack', ax=ax0, binwidth=200)
    plt.savefig('simulation_output/global/'+sim_folder+'/genes/'+gene+'_hist.png')

#create histogram for sim analysis result
def gen_sim_hist_2(gene, context_type, sim_folder):
    #read in simulation output for gene
    sim_hist = pd.read_csv('simulation_output/global/'+sim_folder+'/genes/'+context_type+'/'+gene+'_hist.csv', index_col=0, header=0)
    #create figure
    fig,axs = plt.subplots(layout='constrained', dpi=200, figsize=(30,24), nrows=6, ncols=2)
    
    #calc non-matches
    non_matches = sim_hist.loc[:,'matching'] - sim_hist.loc[:,['T','G','C','A']].sum(axis=1)
    sim_hist['non_matching'] = non_matches

    for index, window_size in enumerate([1, 250, 500, 750, 1000]):
        if index in [0,1]:
            axs[index,1].plot(sim_hist['matching'], color='k')
        matches_avg = np.array([np.mean(sim_hist.loc[i:i+window_size,'matching']) for i in range(0,29900,window_size)])
        axs[index,0].plot(matches_avg, color='r')
        matches_avg = np.repeat(matches_avg, 29900/matches_avg.shape[0])
        axs[index,1].plot(matches_avg, color='r')
        #non_matches_avg = np.array([np.mean(sim_hist.loc[i:i+window_size,'non_matching']) for i in range(0,29900,window_size)])
        #axs[index,0].plot(np.absolute(non_matches_avg), color='b')
        #non_matches_avg = np.repeat(non_matches_avg, 29900/non_matches_avg.shape[0])
        #axs[index,1].plot(np.absolute(non_matches_avg), color='b')
        #minima = np.where(matches_avg==np.min(np.unique(matches_avg)), matches_avg, 0)
        #axs[index,1].plot(minima, color='y')
        
        
        x_positions = np.array([265,21555,25384,26220,26472,27191,27387,27887,28259,29533,29674])
        x_labels = ['start','ORF1ab','S','ORF3ab','E','M','ORF6','ORF7ab','ORF8','N','ORF10']
        axs[index,0].set_xticks(x_positions/window_size, labels=x_labels, rotation=45)
        axs[index,1].set_xticks(x_positions, labels=x_labels, rotation=45)
        axs[index,0].grid()
        axs[index,1].grid()
        axs[index,0].set_title(str(window_size))
        axs[index,1].set_title(str(window_size))
    
    #avg across entire gene
    #for gene, positions in zip(['ORF1ab','S','ORF3ab','E','M','ORF6','ORF7ab','ORF8','N','ORF10'], np.array([[265,21555],[21562:25384],[25392,26220],[26244:26472],[26522:27191],[27201:27387],[27755:27887],[27893:28259],[28273:29533],[29557:29674]])):
    genes = ['ORF1ab','S','ORF3ab','E','M','ORF6','ORF7ab','ORF8','N','ORF10']
    positions = np.array([[265,21555],[21562,25384],[25392,26220],[26244,26472],[26522,27191],[27201,27387],[27755,27887],[27893,28259],[28273,29533],[29557,29674]])
    gene_avg_matches = [np.mean(sim_hist.loc[pos[0]:pos[1],'matching']) for pos in positions]
    gene_avg_non_matches = [np.mean(sim_hist.loc[pos[0]:pos[1],'non_matching']) for pos in positions]
    axs[-1,0].plot(gene_avg_matches, color='r')
    axs[-1,0].plot(np.absolute(gene_avg_non_matches), color='b')
    axs[-1,0].set_title('gene avgs')
    axs[-1,0].set_xticks(np.arange(len(gene_avg_matches)), labels=genes)
    axs[-1,0].grid()

    plt.savefig('simulation_output/global/'+sim_folder+'/genes/'+gene+'_hist.png')


#create fourier transform of simulation output mutation info
def gen_sim_dft(gene, context_type, sim_folder):
    #read in simulation output for gene
    sim_hist = pd.read_csv('simulation_output/global/'+sim_folder+'/genes/'+context_type+'/'+gene+'_hist.csv', index_col=0, header=0)
    
    #fourier transform
    #positions = sim_hist.index.to_numpy() #discrete nucleotide positions
    data = sim_hist.loc[:,'matching'].to_numpy()
    data_shortened = np.array([])
    #for i in range(0,len(data),100):
    #    data_shortened = np.append(data_shortened, np.max(data[i:i+100]))
    #transform = fft.rfft(data_shortened)
    #fig, ax = plt.figure(layout='constrained', dpi=200, figsize=(15,15))
    fig, ax = plt.subplots(figsize=(10,5), dpi=200)
    #plt.plot(transform)
    #labels = sim_hist.index.to_numpy()
    #sns.lineplot(data_shortened, xticks=np.arange(labels[0], labels[-1], 10))
    #plt.xlim(10,transform.shape[0]-10)
    #ax.set_xticklabels(sim_hist.index.to_numpy()[10:-10])    
    #ax.set_xticklabels()
    #plt.plot(sim_hist.loc[:,'matching'])
    #sns.lineplot(sim_hist.loc[:,'matching'], ax=ax)

    #maybe exponential line through scatterplot
    #data = sim_hist
    #data.reset_index(inplace=True, names=['position'])
    #data.columns = ['position','hits']
    #sns.lmplot(x='position', y='matching', data=data, order=5)

    #maybe two lines, 1 for max and 1 for mean
    window_size=50
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_masked = pd.Series(np.where(data>data_mean+(2*data_std), data, data_mean), index=sim_hist.index)
    plt.plot(data_masked, color='grey', alpha=.5)
    data_masked = pd.Series(np.where(data>data_mean+(4*data_std), data, data_mean), index=sim_hist.index)
    plt.plot(data_masked, color='black', alpha=.5)
    
    max_arr = [np.max(data[i:i+window_size]) for i in range(0,len(data),window_size)]
    mean_arr = [np.mean(data[i:i+window_size]) for i in range(0,len(data),window_size)]
    plot_data = pd.DataFrame([max_arr,mean_arr]).T
    #print(plot_data.shape) #(2,383)
    plot_data.columns = ['max', 'mean']
    plot_data.index = np.arange(sim_hist.index.to_numpy()[0],sim_hist.index.to_numpy()[-1],window_size)
    sns.lineplot(plot_data, ax=ax)

    #add labels for high points along top axis
    ax.tick_params(axis='x', which='minor', top=False, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticks(ticks=data_masked.loc[data_masked>data_mean].index, labels=data_masked.loc[data_masked>data_mean].index, minor=True, rotation=45)

    

    plt.savefig('simulation_output/global/'+sim_folder+'/ft.png')
    
    
'''compare population>global mutation list to variant mutation list'''
def compare_mutations(variant):
    #read in global mutation list
    population_mutations = pd.read_csv('population_4fold_rates/global/total_mut_df.csv', index_col=0, header=0)
    #print(population_mutations.shape)

    #read in variant mutation list
    variant_folder_name = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder][0]
    if not os.path.exists('sim_ref_data/'+variant_folder_name+'/collapsed_mutation_list.csv'):
        variant_mutations = pd.read_csv('sim_ref_data/'+variant_folder_name+'/nucleotide-mutations.csv', header=0)
        #print(variant_mutations.shape)
        
        #drop indels
        indels = [index for index in variant_mutations.index if variant_mutations.loc[index, 'mutation'][-1]=='-']
        variant_mutations = variant_mutations.drop(indels)
        #print(variant_mutations.shape, ' after removing indels')
        #reindex variant_mutations
        variant_mutations.index = np.arange(variant_mutations.shape[0])
        #variant_mutations.to_csv('simulation_output/compare_mutations_testing.csv')
        #print('after indels ', variant_mutations.shape)

        #drop mutations with frequency < error rate
        error_rates = {1:1000, 2:3003003, 3:9001800270}
        with open('sim_ref_data/info', 'r') as f:
            lines = f.readlines()
        lines = lines[8:]
        #print(lines)
        seq_counts = {re.search(r' \w+,', line).group(0)[1:-1].lower():int(re.search(r', \d+', line).group(0)[2:]) for line in lines}
        #print(seq_counts)
        seq_count = seq_counts[variant]
        for key,value in error_rates.items():
            if seq_count > value:
                indices = np.where(variant_mutations['count'] == key)[0]
                #print(len(indices))
                variant_mutations = variant_mutations.drop(indices)
                variant_mutations.index = np.arange(variant_mutations.shape[0])
        #print('final shape ', variant_mutations.shape)
        
        #change formatting of variant mutations
        variant_mutations_collapsed = pd.DataFrame(np.zeros([variant_mutations.shape[0],6]), columns = ['position', 'original', 'T', 'G', 'C', 'A'])
        #print(variant_mutations_collapsed.shape, ' start of collapsed df')
        for index in variant_mutations.index:
            mut = variant_mutations.loc[index, 'mutation']
            #print(mut, index)
            variant_mutations_collapsed.loc[index, 'position'] = int(mut[1:-1])
            variant_mutations_collapsed.loc[index, 'original'] = mut[0]
            variant_mutations_collapsed.loc[index, mut[-1]] = variant_mutations.loc[index, 'count']
        #print(variant_mutations_collapsed, variant_mutations_collapsed.shape)

        #collapse mutations at same position
        if len(np.unique(variant_mutations_collapsed.loc[:,'position'])) < variant_mutations_collapsed.shape[0]: #check if there are duplicate positions
            for position in np.unique(variant_mutations_collapsed.loc[:,'position']): #loop through each unique position
                if variant_mutations_collapsed.loc[variant_mutations_collapsed['position'] == position].shape[0] > 1: #check if current position has multiple entries
                    position_df = variant_mutations_collapsed.loc[variant_mutations_collapsed['position'] == position] #subset dataframe for position
                    for i in range(1, variant_mutations_collapsed.loc[variant_mutations_collapsed['position'] == position].shape[0]): #loop through duplicate entries
                        position_df.iloc[0,2:-1] = position_df.iloc[0,2:-1].add(position_df.iloc[i,2:-1]) #sum duplicate entries with original entry
                    variant_mutations_collapsed.loc[(variant_mutations_collapsed['position'] == position)] = position_df #update df
        variant_mutations_collapsed.drop_duplicates(subset='position', inplace=True) #remove duplicate positions
        #print(variant_mutations_collapsed.shape)
        variant_mutations_collapsed.to_csv('sim_ref_data/'+variant_folder_name+'/collapsed_mutation_list.csv')
    else:
        variant_mutations_collapsed = pd.read_csv('sim_ref_data/'+variant_folder_name+'/collapsed_mutation_list.csv', index_col=0, header=0)
    
    overlap_num = 0
    context_num = 0
    total_contexts = 0
    #print(population_mutations, variant_mutations_collapsed)
    for position in population_mutations.loc[:,'position']:
        if position in variant_mutations_collapsed.loc[:,'position']:
            overlap_num += 1
            for mut_base in ['T', 'G', 'C', 'A']:
                if population_mutations.loc[position,mut_base] != 0.0 and variant_mutations_collapsed.loc[position,mut_base] != 0.0:
                    context_num += 1
                    total_contexts += 1
                elif population_mutations.loc[position,mut_base] != 0.0 or variant_mutations_collapsed.loc[position,mut_base] != 0.0:
                    total_contexts += 1
    print(f'{variant} positional mutations \n\tpopulation: {population_mutations.shape[0]}\n\tvariant: {variant_mutations_collapsed.shape[0]}')
    print(f"\tcontextual mutations \n\tpopulation: {np.count_nonzero(population_mutations.loc[:,['T','G','C','A']].values)}\n\tvariant: {np.count_nonzero(variant_mutations_collapsed.loc[:,['T','G','C','A']].values)}")
    print('\tpositional overlap: %3.2f' % ((overlap_num / population_mutations.shape[0])*100))
    print('\tcontextual overlap: %3.2f' % ((context_num / np.count_nonzero(population_mutations.loc[:,['T','G','C','A']].values))*100))
    print('\tpositional precision: %3.2f' % ((overlap_num / (population_mutations.shape[0] + variant_mutations_collapsed.shape[0]))*100))
    print('\tcontextual precision: %3.2f' % ((context_num / (np.count_nonzero(population_mutations.loc[:,['T','G','C','A']].values) + np.count_nonzero(variant_mutations_collapsed.loc[:,['T','G','C','A']].values)))*100))
    print('\t%3.2f' % ((context_num/total_contexts)*100))

'''calc correlation between variant mutation rate matrices'''
def correlate_variants(global_avg_subset_mat, variant_order, extra_variants=False, custom_path=False, var_corr_order=[], vmin=False, vmax=False):

    if extra_variants == True:
        pirola = torch.load('sim_ref_data/pirola_timestamps/BA.2.86(pirola)_full_clade_10_5_23/global_mut_mat.pt').unsqueeze(0)
        kraken = torch.load('sim_ref_data/kraken_timestamps/XBB.1.5(kraken)_full_clade_3_4_23/global_mut_mat.pt').unsqueeze(0)
        global_avg_subset_mat = torch.concat([global_avg_subset_mat, pirola, kraken], dim=0)
        variant_order = variant_order + ['pirola\n10_5_23', 'kraken\n3_4_23']
        #corr_df = pd.DataFrame(np.zeros([len(variant_order), len(variant_order)]), index=variant_order, columns=variant_order)

    #stores correlations between variant mut rates
    corr_df = pd.DataFrame(np.zeros([len(variant_order), len(variant_order)]), index=variant_order, columns=variant_order)
    #stores t-statistics for correlations using formula: t = r * sqrt((n-2)/(1-r^2))
    p_value_df = pd.DataFrame(np.zeros([len(variant_order), len(variant_order)]), index=variant_order, columns=variant_order)
    
    #calc correlations
    for variant_index in range(global_avg_subset_mat.shape[0]):
        for variant_index_2 in range(global_avg_subset_mat.shape[0]):
            #print(variant_order[variant_index], variant_order[variant_index_2])
            #print(torch.corrcoef(torch.stack([global_avg_subset_mat[variant_index].flatten(), global_avg_subset_mat[variant_index_2].flatten()])).numpy()[0,1])
            #corr = torch.corrcoef(torch.stack([global_avg_subset_mat[variant_index].flatten(), global_avg_subset_mat[variant_index_2].flatten()])).numpy()[0,1]
            #corr_df.loc[variant_order[variant_index], variant_order[variant_index_2]] = corr
            #t_stat_df.loc[variant_order[variant_index], variant_order[variant_index_2]] = corr_df.loc[variant_order[variant_index], variant_order[variant_index_2]] / (np.sqrt((1 - (corr_df.loc[variant_order[variant_index], variant_order[variant_index_2]]**2)) / (144-2)))
            #t_stat_df.loc[variant_order[variant_index], variant_order[variant_index_2]] = corr * np.sqrt((144-2)/(1-(corr**2)))
            corr = stats.pearsonr(global_avg_subset_mat[variant_index].numpy().flatten(), global_avg_subset_mat[variant_index_2].numpy().flatten())
            corr_df.loc[variant_order[variant_index], variant_order[variant_index_2]] = corr[0]
            p_value_df.loc[variant_order[variant_index], variant_order[variant_index_2]] = corr[1]
    
    #calc multi-regression
    #shape: tensor(9,12,12)
    '''data = torch.flatten(global_avg_subset_mat, start_dim=1).numpy()
    labels = np.arange(data.shape[0])

    print(data.shape, labels.shape)
    model = LinearRegression()
    model.fit(data,labels)

    print(variant_order)
    print(model.coef_)'''

    
    


    #plot correlations
    if extra_variants:
        fig, axs = plt.subplots(figsize=(12.5,12.5))
        var_corr_order = ['delta','omicron', 'alpha','kraken', 'pirola','kraken\n3_4_23','gamma','pirola\n10_5_23', 'epsilon', 'beta', 'iota', 'lambda', 'mu', 'kappa', 'eta']#corr_df.sort_values('alpha', ascending=False).index.values
    #elif custom_path:
    #    fig, axs = plt.subplots(figsize=(8,8))
    #    var_corr_order = ['delta','omicron','alpha','gamma','pirola','kraken','epsilon','mu', 'beta', 'iota','eta','kappa','lambda'] #, 'beta', 'iota'
    else:
        fig, axs = plt.subplots(figsize=(12,12))
        #var_corr_order = ['omicron','alpha', 'pirola','gamma','beta','epsilon','iota','delta','kraken']# 'mu',    'kappa', 'eta',   'lambda',
        #var_corr_order = ['delta','omicron','alpha','gamma','mu','beta','lambda','epsilon','iota','pirola','kraken','kappa','eta'] #17329
        #var_corr_order = ['delta','omicron','alpha','gamma','kraken','pirola','beta','lambda','mu','epsilon','iota','kappa','eta'] #45131
        #var_corr_order = ['omicron','delta','kraken','pirola','alpha','gamma','epsilon','mu','beta','lambda','iota','kappa','eta'] #76...
    print(corr_df)
    #corr_df.to_csv('simulation_output/000.csv')
    annotations = corr_df.loc[var_corr_order, var_corr_order].round(2) #round df for formatting
    #annotations = annotations.mask(annotations < .9, '-') #replace any insignificant correlations with dash
    annotations = annotations.mask(p_value_df.loc[var_corr_order,var_corr_order] > .05, '-')
    #var_corr_order[:3]+['kraken_3_4_23','pirola_10_5_23']+var_corr_order[5:]
    if vmin != False and vmax != False:
        sns.heatmap(corr_df.loc[var_corr_order, var_corr_order], cmap='Greys', ax=axs, xticklabels=var_corr_order, yticklabels=var_corr_order, annot=annotations, fmt='', annot_kws={'fontsize':22}, vmin=vmin, vmax=vmax, mask=p_value_df.loc[var_corr_order,var_corr_order]>.05, linewidth=.5, linecolor='grey')
    else:
        sns.heatmap(corr_df.loc[var_corr_order, var_corr_order], cmap='Greys', ax=axs, xticklabels=var_corr_order, yticklabels=var_corr_order, annot=annotations, fmt='', annot_kws={'fontsize':22})
    axs.set_yticklabels([var.capitalize() for var in var_corr_order], rotation=45, fontsize=22, wrap=True)
    axs.set_xticklabels([var.capitalize() for var in var_corr_order], rotation=45, fontsize=22)
    axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs.set_title('Variant Correlation Comparison', fontsize=28)
    plt.tight_layout()
    if custom_path:
        plt.savefig(custom_path)
    else:
        plt.savefig('simulation_output/variant_correlation.png')
    plt.close()

    #plot p values
    fig, axs = plt.subplots(figsize=(12,12))
    sns.heatmap(p_value_df.loc[var_corr_order, var_corr_order], cmap='Greys_r', ax=axs, xticklabels=var_corr_order, yticklabels=var_corr_order, annot=True, annot_kws={'fontsize':24}, vmin=.05, vmax=.1) #mask=t_stat_df < 1.962
    axs.set_yticklabels([var.capitalize() for var in var_corr_order], rotation=45, fontsize=24, wrap=True)
    axs.set_xticklabels([var.capitalize() for var in var_corr_order], rotation=45, fontsize=24)
    axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs.set_title('Variant p Values', fontsize=28)
    plt.tight_layout()
    if custom_path:
        plt.savefig(custom_path.replace('correlation', 'p_values'))
    else:
        plt.savefig('simulation_output/variant_p_values.png')
    plt.close()
    print(p_value_df)
    return [corr_df, p_value_df]

'''calc correlation between variant group mutation rate matrices'''
def correlate_variant_groups():
    #create output df for comparing correlations
    group_order = ['Group_A', 'Group_C', 'All', 'Group_B']
    #stores correlations between variant group mut rates
    corr_df = pd.DataFrame(np.zeros([len(group_order), len(group_order)]), index=group_order, columns=group_order)
    #stores t-statistics for correlations using formula: t = r * sqrt((n-2)/(1-r^2))
    t_stat_df = pd.DataFrame(np.zeros([len(group_order), len(group_order)]), index=group_order, columns=group_order)
    
    #read in group mutation matrices
    group_mut_mats = [pd.read_csv('simulation_output/global_group'+str(group_index)+'_rates.csv', index_col=0, header=0) for group_index in range(1,5)]

    #calc correlations
    for group_index in range(len(group_mut_mats)):
        for group_index_2 in range(len(group_mut_mats)):
            #print(variant_order[variant_index], variant_order[variant_index_2])
            #print(torch.corrcoef(torch.stack([global_avg_subset_mat[variant_index].flatten(), global_avg_subset_mat[variant_index_2].flatten()])).numpy()[0,1])
            print(torch.stack([torch.tensor(group_mut_mats[group_index].values.flatten()), torch.tensor(group_mut_mats[group_index_2].values.flatten())]))
            print(torch.corrcoef(torch.stack([torch.tensor(group_mut_mats[group_index].values.flatten()), torch.tensor(group_mut_mats[group_index_2].values.flatten())])))
            corr = torch.corrcoef(torch.stack([torch.tensor(group_mut_mats[group_index].values.flatten()), torch.tensor(group_mut_mats[group_index_2].values.flatten())])).numpy()[0,1]
            corr_df.loc[group_order[group_index], group_order[group_index_2]] = corr
            #t_stat_df.loc[group_order[group_index], group_order[group_index_2]] = corr / (np.sqrt((1 - (corr**2)) / (144-2)))
            t_stat_df.loc[group_order[group_index], group_order[group_index_2]] = corr * np.sqrt((144-2)/(1-(corr**2)))

    #plot correlations
    fig, axs = plt.subplots(figsize=(10,10))
    #var_corr_order = ['alpha', 'delta', 'omicron', 'beta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu']
    #var_corr_order = ['delta','omicron', 'alpha','kraken', 'gamma', 'pirola', 'epsilon', 'beta', 'iota', 'lambda', 'mu', 'kappa', 'eta']#corr_df.sort_values('alpha', ascending=False).index.values
    var_corr_order = ['Group_A', 'Group_B', 'Group_C', 'All']
    annotations = corr_df.loc[var_corr_order, var_corr_order].round(2) #round df for formatting
    #annotations = annotations.mask(annotations < .5, '-') #replace any insignificant correlations with dash
    sns.heatmap(corr_df.loc[var_corr_order, var_corr_order], cmap='Blues', ax=axs, xticklabels=var_corr_order, yticklabels=var_corr_order, annot=annotations, fmt='')
    axs.set_yticklabels(var_corr_order, rotation='horizontal')
    axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs.set_title('Group Correlation Comparison')
    plt.tight_layout()
    plt.savefig('simulation_output/group_correlation.png')
    plt.close()

    #plot t_statistics
    fig, axs = plt.subplots(figsize=(10,10))
    sns.heatmap(t_stat_df.loc[var_corr_order, var_corr_order], cmap='Blues', ax=axs, xticklabels=var_corr_order, yticklabels=var_corr_order, annot=True, vmax=1.962, mask=t_stat_df > 100) #mask=t_stat_df < 1.962
    axs.set_yticklabels(var_corr_order, rotation='horizontal')
    axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs.set_title('Group t Statistics')
    plt.tight_layout()
    plt.savefig('simulation_output/group_t_stats.png')
    plt.close()
    print(t_stat_df)

'''compare the mutations > .1 for all variants to find overlap'''
def compare_variant_mutations():
    #store variant names and create empty df
    folders = [folder for folder in os.listdir('sim_ref_data/') if '_full_clade' in folder]
    matching_muts_df = pd.DataFrame(np.zeros([11,11]), index = folders, columns = folders)
    #loop through each variant
    for var_1_index, var_1 in enumerate(folders):
        var_df_1 = pd.read_csv('sim_ref_data/'+var_1+'/reference_mutations/0.1_reference_mutations.csv', header=0, index_col=0)
        for var_2_index, var_2 in enumerate([folder for folder in os.listdir('sim_ref_data/') if '_full_clade' in folder]):
            var_df_2 = pd.read_csv('sim_ref_data/'+var_2+'/reference_mutations/0.1_reference_mutations.csv', header=0, index_col=0)
            #check if variant 1 mut position in variant 2
            for position in var_df_1.loc[:,'position']:
                if position in var_df_2.loc[:,'position'].values:
                    #check if same mutation
                    #print(var_df_1[var_df_1['position'] == position].loc[:,'mut'].values)
                    if var_df_1[var_df_1['position'] == position].loc[:,'mut'].values[0] == var_df_2[var_df_2['position']==position].loc[:,'mut'].values[0]:
                        matching_muts_df.loc[var_1, var_2] += 1
            #normalize by number of unique mutations in both variants
            matching_muts_df.iloc[var_1_index, var_2_index] = matching_muts_df.iloc[var_1_index, var_2_index] / len(np.unique(pd.concat([var_df_1, var_df_2], axis=0).loc[:,'position'].values))
    #reorganize naming
    variant_names = [re.search(r'\(\w+\)', var_name).group(0)[1:-1] for var_name in folders]
    matching_muts_df.index = variant_names
    matching_muts_df.columns = variant_names
    matching_muts_df = matching_muts_df.loc[np.sort(variant_names), np.sort(variant_names)]
    matching_muts_df.to_csv('simulation_output/variant_mutation_comparison.csv')

#create 4x4 output plot for variants
def plot_variants_grid(global_avg_subset_mat, global_naive_subset_mat, variant_order, shape='default', threshold=0):
    global rows, columns
    #normalize rates even though data is between 0 and 1 by default because its normalized by the triplet count
    #for mat_index, mat in enumerate(global_avg_subset_mat):
    #    global_avg_subset_mat[mat_index] = (mat - np.min(mat.numpy())) / (np.max(mat.numpy())-np.min(mat.numpy()))


    #create figure
    #old
    '''grid = gridspec.GridSpec(1,3, figure=fig, wspace=.15, hspace=.1, width_ratios=[.47,.47,.06]) #, width_ratios=[.5/4,.5/4,.5/4,.5/4, .5/3,.5/3,.5/3], height_ratios=[1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8]
    alpha_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=grid[0,0], height_ratios=[0.9,.05], hspace=.2)
    rest_grid = gridspec.GridSpecFromSubplotSpec(4,3,subplot_spec=grid[0,1])
    cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=grid[0,2])'''
    #new
    if shape == 'default':
        fig = plt.figure(layout='constrained', dpi=100, figsize=(16,10)) #, edgecolor='Black', linewidth=3
        grid = gridspec.GridSpec(1,3, figure=fig, wspace=.15, hspace=.1, width_ratios=[.47,.47,.06])
        epsilon_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=grid[0,1], height_ratios=[0.9,.05], hspace=.2)
        rest_grid = gridspec.GridSpecFromSubplotSpec(4,3,subplot_spec=grid[0,0])
        cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=grid[0,2])
        #old
        '''ax0 = fig.add_subplot(alpha_grid[0,0])
        ax13 = fig.add_subplot(alpha_grid[1,0])
        ax1,ax2,ax3 = [fig.add_subplot(rest_grid[0,i]) for i in range(0,3)]
        ax4,ax5,ax6 = [fig.add_subplot(rest_grid[1,i]) for i in range(0,3)]
        ax7,ax8,ax9 = [fig.add_subplot(rest_grid[2,i]) for i in range(0,3)]
        ax10,ax11,ax12 = [fig.add_subplot(rest_grid[3,i]) for i in range(0,3)]'''
    elif shape == 'aggregate':
        fig = plt.figure(layout='constrained', dpi=200, figsize=(16,8)) #, edgecolor='Black', linewidth=3
        grid = gridspec.GridSpec(1,3, figure=fig, wspace=.1, hspace=.1, width_ratios=[.35,.04,.35])
        epsilon_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=grid[0,0], height_ratios=[0.89,.14], hspace=.14, wspace=.1)
        rest_grid = gridspec.GridSpecFromSubplotSpec(5,3,subplot_spec=grid[0,2], height_ratios = [.1,.35,.1,.35,.1])
        cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=grid[0,1])
        #fig.get_layout_engine().set(w_pad=.5, wspace=.4)
        ax12 = fig.add_subplot(epsilon_grid[0,0])
        ax13 = fig.add_subplot(epsilon_grid[1,0])
        ax0,ax1,ax2 = [fig.add_subplot(rest_grid[1,i]) for i in range(0,3)]
        ax_cbar = fig.add_subplot(cbar_grid[0,0])
        if 'jean' in variant_order[0]:
            ax3,ax4,ax5 = [fig.add_subplot(rest_grid[3,i]) for i in range(0,3)]
            axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax12,ax13] #,ax6,ax7,ax8
        else:
            ax3,ax4,ax5 = [fig.add_subplot(rest_grid[3,i]) for i in range(0,3)]
            axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax12,ax13]
        if shape == 'default':
            ax9,ax10,ax11 = [fig.add_subplot(rest_grid[3,i]) for i in range(0,3)]
            axes = axes[:-2]+[ax9,ax10,ax11]+axes[-2:]
            var_corr_order = ['delta','alpha','pirola', 'omicron','kraken','gamma', 'beta','eta','iota','kappa','lambda','mu', 'epsilon']
        elif shape == 'aggregate':
            #var_corr_order = ['omicron','alpha','aggregate','kraken','gamma','beta','pirola','epsilon','iota','delta']
            #var_corr_order = ['pirola','gamma','epsilon','omicron','alpha','aggregate','kraken','beta','iota','delta']
            var_corr_order = ['alpha','delta','kraken','omicron','pirola','transient','aggregate']
            #var_corr_order = ['jean_alpha','jean_beta','jean_delta','jean_gamma','jean_omicron','jean_usa','jean_total']
    elif shape == '8_vars':
        fig = plt.figure(layout='constrained', dpi=200, figsize=(20,20))
        grid = gridspec.GridSpec(1,3, figure=fig, wspace=.1, hspace=.1, width_ratios=[.7,.05,.25])
        left_grid = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=grid[0,0], height_ratios=[.7,.05,.245], hspace=.14, wspace=.1)
        primary_variant_grid = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=left_grid[0,0], width_ratios=[.96, .04], hspace=.14, wspace=.1)
        cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=primary_variant_grid[0,1])
        primary_variant_sub_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=primary_variant_grid[0,0], height_ratios=[0.89,.14], hspace=.1)
        right_grid = gridspec.GridSpecFromSubplotSpec(7,1,subplot_spec=grid[0,2], height_ratios = [.25,.01,.25,.01,.25,.01,.25])
        bottom_grid = gridspec.GridSpecFromSubplotSpec(1,5,subplot_spec=left_grid[2,0], width_ratios = [.3,.05,.3,.05,.3])
        
        #create axes
        var_corr_order = ['persistent', 'transient', 'alpha', 'delta', 'kraken', 'omicron', 'pirola','aggregate']
        ax12, ax13 = [fig.add_subplot(primary_variant_sub_grid[i,0]) for i in range(0,2)] #aggregate_full, aggregate_naive
        ax_cbar = fig.add_subplot(primary_variant_grid[0,1])
        ax2, ax3, ax4, ax5 = [fig.add_subplot(right_grid[i,0]) for i in [0,2,4,6]]
        ax6, ax7, ax8 = [fig.add_subplot(bottom_grid[0,i]) for i in [0,2,4]]
        axes = [ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax12,ax13]
        fontsize_1 = 25
        fontsize_2 = 30
    elif shape == '8_vars_2':
        fig = plt.figure(layout='constrained', dpi=200, figsize=(20,8))
        grid = gridspec.GridSpec(1,3, figure=fig, wspace=.1, hspace=.1, width_ratios=[.45,.35,.1])
        main_variant = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=grid[0,0], width_ratios=[.95,.05], hspace=.14, wspace=.1)
        main_variant_left = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=main_variant[0,0],height_ratios=[0.85,.15]) #, width_ratios=[0.95,.05]
        #main_variant_lower = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=main_variant[1,0])
        #cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=main_variant_upper[0,1])
        secondary_variants = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=grid[0,1])
        aggregate_variants = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=grid[0,2])
        ax12 = fig.add_subplot(main_variant_left[0,0])
        ax_cbar = fig.add_subplot(main_variant[0,1])
        ax13 = fig.add_subplot(main_variant_left[1,0])
        ax0,ax1 = [fig.add_subplot(secondary_variants[0,i]) for i in range(2)]
        ax2,ax3 = [fig.add_subplot(secondary_variants[1,i]) for i in range(2)]
        ax4,ax5,ax6 = [fig.add_subplot(aggregate_variants[i,0]) for i in range(3)]
        axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax12,ax13]
        var_corr_order = ['delta','omicron','kraken','pirola','all','persistent','transient','alpha']
        fontsize_1 = 20
        fontsize_2 = 28


    elif shape == 'dynamic_aggregate':
        if len(variant_order) >= 0 and len(variant_order) <= 3:
            nrows = 3
            filler_rows = 2
            height = 6
            width = 16
        elif len(variant_order) > 3 and len(variant_order) <= 6:
            nrows = 5
            filler_rows = 3
            height = 8
            width = 16
        elif len(variant_order) > 6 and len(variant_order) <=9:
            nrows = 7
            filler_rows = 4
            height = 12
            width = 22
        else:
            nrows = 9
            filler_rows = 5
            height = 16
            width = 26
        
        fig = plt.figure(layout='constrained', dpi=200, figsize=(width,height)) #, edgecolor='Black', linewidth=3
        grid = gridspec.GridSpec(1,3, figure=fig, wspace=.1, hspace=.1, width_ratios=[.35,.02,.35])
        primary_variant_grid = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=grid[0,0], height_ratios=[0.89,.14], hspace=.14, wspace=.1)

        heights = np.zeros([nrows])
        heights[0::2] = .3 / filler_rows
        heights[1::2] = .7 / nrows-filler_rows

        rest_grid = gridspec.GridSpecFromSubplotSpec(nrows,3,subplot_spec=grid[0,2], height_ratios = heights)
        cbar_grid = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=grid[0,1], wspace=.1)
        
        ax12 = fig.add_subplot(primary_variant_grid[0,0])
        ax13 = fig.add_subplot(primary_variant_grid[1,0])
        axes = []
        for row in range(1,nrows,2):
            for i in range(0,3):
                if not len(axes) >= len(variant_order)-1:
                    axes.append(fig.add_subplot(rest_grid[row,i]))
        axes = axes + [ax12,ax13]
        ax_cbar = fig.add_subplot(cbar_grid[0,0])
        var_corr_order = ['alpha','delta','kraken','omicron','pirola','persistent','transient','aggregate']
        #var_corr_order = ['jean_alpha','jean_beta','jean_delta','jean_gamma','jean_omicron','jean_usa','jean_total']
        fontsize_1 = 15
        fontsize_2 = 20

    elif shape == '3_vars':
        fig = plt.figure(layout='constrained', dpi=200, figsize=(12,8))
        grid = gridspec.GridSpec(1,2, figure=fig, wspace=.2, hspace=.1, width_ratios=[.75,.25])
        main_variant = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=grid[0,0], width_ratios=[.85,.12], hspace=.14)
        main_variant_left = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=main_variant[0,0],height_ratios=[0.85,.15]) #, width_ratios=[0.95,.05]
        secondary_variants = gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=grid[0,1], height_ratios=[.05,.42,.05,.42,.05])
        cbar = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=main_variant[0,1], height_ratios=[.04,.88,.08])
        ax12 = fig.add_subplot(main_variant_left[0,0])
        ax_cbar = fig.add_subplot(cbar[1])
        ax13 = fig.add_subplot(main_variant_left[1,0])
        ax0,ax1 = [fig.add_subplot(secondary_variants[i]) for i in range(1,5,2)]
        axes = [ax0,ax1,ax12,ax13]
        var_corr_order = ['persistent','transient','all']
        fontsize_1 = 20
        fontsize_2 = 28

    #plot full mut rate mats
    for axs_index, variant in enumerate(var_corr_order):
        print(variant, variant_order.index(variant))
        mat = global_avg_subset_mat[variant_order.index(variant)]
        if variant == var_corr_order[-1]:
            #big_mat = sns.heatmap(np.log10(mat.numpy()), cmap='Greys', cbar=False, xticklabels=False, ax=axes[-2], linecolor='gray', linewidth=.5)
            big_mat = sns.heatmap(mat, cmap='Greys', cbar=False, xticklabels=False, ax=axes[-2], linecolor='gray', linewidth=.5)
            axes[axs_index].set_yticklabels(rows_figs[:-4], rotation='horizontal', fontsize=fontsize_1)
            axes[axs_index].set_title(var_corr_order[axs_index].capitalize(), fontsize=fontsize_2)
            variances = (torch.max(mat, dim=0)[0] - torch.min(mat, dim=0)[0]).numpy()
            
        else:
            sns.heatmap(mat, cmap='Greys', cbar=False, xticklabels=False, yticklabels=False, ax=axes[axs_index], linecolor='gray', linewidth=.5)
            axes[axs_index].set_title(var_corr_order[axs_index].capitalize(), fontsize=fontsize_2)
    #plot naive alpha = old, epsilon = new
    naive = global_naive_subset_mat[variant_order.index(var_corr_order[-1])].reshape([1,12])[0].numpy()
    print(naive)
    print('variance')
    print(variances)
    sns.heatmap(pd.DataFrame([naive,variances]), cmap='Greys', cbar=False, ax=ax13, linecolor='gray', linewidth=.5) #'alpha'
    ax13.set_xticklabels(columns_figs, fontsize=fontsize_1, rotation=45)
    ax13.set_yticklabels(['N[X>Y]N','max.var'], fontsize=fontsize_1, rotation='horizontal')
    pd.Series(variances, index=columns_figs).to_csv('simulation_output/final_info/'+var_corr_order[-1]+'_max_var.csv')


    #plot cbar
    #fig.colorbar(alpha, ax=[ax3,ax6,ax9,ax12], orientation='vertical', fraction=0.1)
    #cbar_ax = matplotlib.inset_axes(ax12, width='5%', height='400%', loc='right', bbox_to_anchor=(1.05,0.3,1,1), bbox_transform=ax12.transAxes, borderpad=0)
    color_map = cm.Greys
    #norm = colors.LogNorm(vmin=torch.min(global_avg_subset_mat[variant_order.index(var_corr_order[axs_index])]), vmax=torch.max(global_avg_subset_mat[variant_order.index(var_corr_order[axs_index])]))
    norm = colors.Normalize(vmin=torch.min(global_avg_subset_mat[variant_order.index(var_corr_order[axs_index])]), vmax=torch.max(global_avg_subset_mat[variant_order.index(var_corr_order[axs_index])]))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map), cax=ax_cbar, orientation='vertical')
    cbar.set_ticklabels([i/10 for i in range(0,11,2)], fontsize=fontsize_1)
    fig.get_layout_engine().set(w_pad=8 / 72, h_pad=8 / 72, hspace=0, wspace=0)
    plt.savefig('simulation_output/variant_comparisons_'+str(threshold)+'.png')
    plt.close()

#plot simplest matrices for mutation rates (T,G,C,A)
def plot_super_naive_mats(naive_mat, path, variant_order, gene_order, sim_type):
    print(naive_mat.shape)
    #variants
    if naive_mat.shape[0] == len(variant_order):
        #super_naive_mats = []
        for var_index, var_mat in enumerate(naive_mat):
            if sim_type == 'global':
                super_naive_mats = torch.stack([torch.mean(var_mat[i:i+3]) for i in range(0,var_mat.shape[0],3)])
                plot_mut_mat(super_naive_mats, 'simulation_output/super_naive_mats/'+sim_type+'/'+variant_order[var_index]+'.png', gene_order, sim_type, contexts='super_naive_contexts')
            elif sim_type == 'genes':
                super_naive_mats = torch.stack([torch.stack([torch.mean(var_mat[gene_index, i:i+3]) for i in range(0,var_mat.shape[1],3)]) for gene_index in range(0,var_mat.shape[0])])
                plot_mut_mat(super_naive_mats, 'simulation_output/super_naive_mats/'+sim_type+'/'+variant_order[var_index]+'.png', gene_order, sim_type, contexts='super_naive_contexts')
        #super_naive_mats = torch.stack(super_naive_mats)
    #grouped
    else:
        if sim_type == 'global':
            super_naive_mats = torch.stack([torch.mean(naive_mat[i:i+3]) for i in range(0,naive_mat.shape[0],3)])
            #super_naive_mats = (super_naive_mats - np.min(super_naive_mats.numpy())) / (np.max(super_naive_mats.numpy()) - np.min(super_naive_mats.numpy()))
            plot_mut_mat(super_naive_mats, 'simulation_output/super_naive_mats/'+sim_type+'/'+path+'.png', gene_order, sim_type, contexts='super_naive_contexts')
        elif sim_type == 'genes': #15x12
            super_naive_mats = torch.stack([torch.stack([torch.mean(naive_mat[gene_index, i:i+3]) for i in range(0,naive_mat.shape[1],3)]) for gene_index in range(0,naive_mat.shape[0])])
            plot_mut_mat(super_naive_mats, 'simulation_output/super_naive_mats/'+sim_type+'/'+path+'.png', gene_order, sim_type, contexts='super_naive_contexts')
        

#compare mutations in rdrp across variants (looking at overlap for all mutations in lists)
def compare_rdrp():
    global rows, columns
    with open('sim_ref_data/info', 'r') as f:
        lines = f.readlines()
    seq_counts = {re.search(r'= \w+,', line).group(0)[2:-1].lower():int(re.search(r', \d+ ', line).group(0)[2:-1]) for line in lines[8:]}
    variant_dfs = {}
    for variant_folder in [folder for folder in os.listdir('sim_ref_data') if '_full_clade' in folder]:
        variant_df = pd.read_csv('sim_ref_data/'+variant_folder+'/nucleotide-mutations.csv', header=0)
        variant_df = pd.concat([variant_df, pd.DataFrame([[mut[0], int(mut[1:-1]), mut[-1]] for mut in variant_df.loc[:,'mutation']], columns=['orig', 'position', 'mut'])], axis=1)
        variant_df.drop(variant_df.loc[variant_df['mut']=='-'].index, inplace=True)
        variant_df.drop(variant_df.loc[variant_df['position']<13468].index, inplace=True)
        variant_df.drop(variant_df.loc[variant_df['position']>16264].index, inplace=True)
        variant_name = re.search(r'\(\w+\)', variant_folder).group(0)[1:-1]
        if seq_counts[variant_name] < 1000:
            mut_threshold = 1
        elif seq_counts[variant_name] < 3003003:
            mut_threshold = 2
        elif seq_counts[variant_name] > 3003003:
            mut_threshold = 3
        variant_df.drop(variant_df.loc[variant_df['count']<mut_threshold].index, inplace=True)
        variant_dfs[variant_name] = variant_df
    output_df = pd.DataFrame(np.zeros([len(variant_dfs.keys()),len(variant_dfs.keys())]), index=variant_dfs.keys(), columns=variant_dfs.keys())
    for variant_1, df_1 in variant_dfs.items():
        for variant_2, df_2 in variant_dfs.items():
            common_muts_df = pd.merge(df_1, df_2, how='inner', on=['mutation'])
            total_muts_df = pd.merge(df_1, df_2, how='outer', on=['mutation'])
            output_df.loc[variant_1, variant_2] = common_muts_df.shape[0]/total_muts_df.shape[0]
    output_df.to_csv('simulation_output/rdrp_comp.csv')
    #plot
    fig, axs = plt.subplots(figsize=(10,10))
    #var_corr_order = ['alpha', 'delta', 'omicron', 'beta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu']
    var_corr_order = ['delta','omicron', 'alpha','kraken', 'gamma', 'pirola', 'epsilon', 'beta', 'iota', 'lambda', 'mu', 'kappa', 'eta']#corr_df.sort_values('alpha', ascending=False).index.values
    #var_corr_order = ['Group_A', 'Group_B', 'Group_C', 'All']
    annotations = output_df.loc[var_corr_order, var_corr_order].round(2) #round df for formatting
    annotations = annotations.mask(annotations < .3, '-') #replace any insignificant correlations with dash
    sns.heatmap(output_df.loc[var_corr_order, var_corr_order], cmap='Blues', ax=axs, xticklabels=var_corr_order, yticklabels=var_corr_order, annot=annotations, fmt='')
    axs.set_yticklabels(var_corr_order, rotation='horizontal')
    axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs.set_title('Group Correlation Comparison')
    plt.tight_layout()
    plt.savefig('simulation_output/rdrp_comp.png')
    plt.close()

    fasta = get_fasta()[:-1]
    #gwtc
    gwtc = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=['T','G','C','A'])
    four_fold_codes = ['CT', 'GT', 'TC', 'CC', 'AC', 'GC', 'CG', 'GG']
    fourfold_sites = []
    for triplet_index in range(13467, 16263, 3):
        triplet = fasta[triplet_index:triplet_index+3]
        if triplet[:2] in four_fold_codes:
            #print(triplet)
            gwtc.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
            fourfold_sites.append(triplet_index+2)
    gwtc.to_csv('simulation_output/rdrp_gwtc.csv')
    
    for variant, df in variant_dfs.items():
        mut_rate_df = pd.DataFrame(np.zeros([12,12]), index=rows, columns=columns)
        for index in df.index:
            mutation = df.loc[index, 'mutation']
            position = int(df.loc[index, 'position'])-1
            if position in fourfold_sites:
                prev_nuc, orig_nuc, next_nuc = fasta[position-1:position+2]
                #print(mutation, prev_nuc, orig_nuc, next_nuc)
                mut_rate_df.loc[prev_nuc+'[X>Y]'+next_nuc, orig_nuc+'>'+mutation[-1]] += 1
        for column in mut_rate_df.columns:
            mut_rate_df.loc[:,column] = mut_rate_df.loc[:,column] / gwtc.loc[:'C[X>Y]A',column[0]].replace(0.0, np.inf)
        variant_dfs[variant] = mut_rate_df
    t_df = output_df.copy()
    for variant_1, df_1 in variant_dfs.items():
        for variant_2, df_2 in variant_dfs.items():
            corr = torch.corrcoef(torch.stack([torch.tensor(df_1.values.flatten()), torch.tensor(df_2.values.flatten())])).numpy()[0,1]
            output_df.loc[variant_1, variant_2] = corr
            t_df.loc[variant_1, variant_2] = corr * np.sqrt(142 / (1 - (corr**2)))
    output_df.to_csv('simulation_output/rdrp_corr.csv')
    t_df.to_csv('simulation_output/rdrp_t.csv')
    #plot
    fig, axs = plt.subplots(figsize=(10,10))
    #var_corr_order = ['alpha', 'delta', 'omicron', 'beta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu']
    var_corr_order = ['delta','omicron', 'alpha','kraken', 'gamma', 'pirola', 'epsilon', 'beta', 'iota', 'lambda', 'mu', 'kappa', 'eta']#corr_df.sort_values('alpha', ascending=False).index.values
    #var_corr_order = ['Group_A', 'Group_B', 'Group_C', 'All']
    annotations = output_df.loc[var_corr_order, var_corr_order].round(2) #round df for formatting
    #annotations = annotations.mask(annotations < .5, '-') #replace any insignificant correlations with dash
    sns.heatmap(output_df.loc[var_corr_order, var_corr_order], cmap='Blues', ax=axs, xticklabels=var_corr_order, yticklabels=var_corr_order, annot=annotations, fmt='')
    axs.set_yticklabels(var_corr_order, rotation='horizontal')
    axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs.set_title('Group Correlation Comparison')
    plt.tight_layout()
    plt.savefig('simulation_output/rdrp_corr.png')
    plt.close()
    #plot
    fig, axs = plt.subplots(figsize=(10,10))
    #var_corr_order = ['alpha', 'delta', 'omicron', 'beta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu']
    var_corr_order = ['delta','omicron', 'alpha','kraken', 'gamma', 'pirola', 'epsilon', 'beta', 'iota', 'lambda', 'mu', 'kappa', 'eta']#corr_df.sort_values('alpha', ascending=False).index.values
    #var_corr_order = ['Group_A', 'Group_B', 'Group_C', 'All']
    annotations = t_df.loc[var_corr_order, var_corr_order].round(2) #round df for formatting
    #annotations = annotations.mask(annotations < .05, '-') #replace any insignificant correlations with dash
    sns.heatmap(t_df.loc[var_corr_order, var_corr_order], cmap='Blues', ax=axs, xticklabels=var_corr_order, yticklabels=var_corr_order, annot=annotations, fmt='', vmax=1.962, mask=t_df.loc[var_corr_order, var_corr_order] > 100)
    axs.set_yticklabels(var_corr_order, rotation='horizontal')
    axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs.set_title('Group Correlation Comparison')
    plt.tight_layout()
    plt.savefig('simulation_output/rdrp_t.png')
    plt.close()

#compare high frequency mutations in nsps and spike
def compare_rdrp_thresholded_replaced(variant_order):
    #read in sequence counts for each variant
    with open('sim_ref_data/info', 'r') as f:
        lines = f.readlines()
    seq_counts = {re.search(r'= \w+,', line).group(0)[2:-1].lower():int(re.search(r', \d+ ', line).group(0)[2:-1]) for line in lines[8:]}
    
    #read in ref fasta
    fasta = pd.Series([*get_fasta()[:-1]], name='ref')

    #indexing is based on NCBI reference sequence NC_045512.2
    '''rdrp_fasta = fasta[nsp_positions['rdrp'][0]:nsp_positions['rdrp'][1]+1]'''

    #codon matrix
    aa_codes = np.array([[['Phe','Phe','Leu','Leu'],['Ser','Ser','Ser','Ser'],['Tyr','Tyr','stop','stop'],['Cys','Cys','stop','Trp']],
                        [['Leu','Leu','Leu','Leu'],['Pro','Pro','Pro','Pro'],['His','His','Gln','Gln'],['Arg','Arg','Arg','Arg']],
                        [['Ile','Ile','Ile','Met'],['Thr','Thr','Thr','Thr'],['Asn','Asn','Lys','Lys'],['Ser','Ser','Arg','Arg']],
                        [['Val','Val','Val','Val'],['Ala','Ala','Ala','Ala'],['Asp','Asp','Glu','Glu'],['Gly','Gly','Gly','Gly']]])
    #amino acid abbreviation codes
    aa_abbrevs = {'Ala':'A', 'Arg':'R', 'Asn':'N', 'Asp':'D', 'Cys':'C', 'Glu':'E', 'Gln':'Q', 'Gly':'G', 'His':'H', 'Ile':'I',
                  'Leu':'L', 'Lys':'K', 'Met':'M', 'Phe':'F', 'Pro':'P', 'Ser':'S', 'Thr':'T', 'Trp':'W', 'Tyr':'Y', 'Val':'V', 'stop':''}
    #nucleotide order for indexing aa_codes
    nuc_order = ['U','C','A','G']

    #array storing mutation output
    output_mat = np.array([[[],[],[],[]]])

    #loop through regions
    for gene,positions in nsp_positions.items():
        subset_fasta = fasta.loc[positions[0]:positions[1]]
        if gene == 'rdrp':
            subset_fasta = pd.concat([fasta.loc[positions[0]:positions[0]+26], pd.Series(['C']), fasta.loc[positions[0]+27:positions[1]]])
            subset_fasta.index = range(13441, 13441+subset_fasta.shape[0])
            #print(subset_fasta.loc[13441:13475])
        print(gene)
        #print(subset_fasta)

        #loop through variants
        for variant in variant_order:
            #read in variant mutation data
            variant_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder][0]
            variant_muts = pd.read_csv('sim_ref_data/'+variant_folder+'/collapsed_mutation_list.csv', index_col=0, header=0)

            #drop mutations not in region
            variant_muts = variant_muts.loc[(variant_muts['position'].isin(np.arange(positions[0],positions[1])))]
            #print(variant_muts)

            #loop through mutations and remove any that are less than 50% frequency
            for index in variant_muts.index:
                for mut in variant_muts.columns[2:6]: #T,G,C,A
                    '''if variant_muts.loc[index, mut] < seq_counts[variant]/2: #check for muts with frequency > 50%
                        variant_muts.loc[index, mut] = 0
                    else:
                        #
                        #print(variant_muts.loc[index,:])
                        #triplet isn't centered on mutation
                        position = int(variant_muts.loc[index,'position']) #position of mutation
                        triplet_position = (position - positions[0]) % 3
                        if gene == 'rdrp':
                            position += 1
                        #accounts for open reading frame differences
                        if gene not in ['rdrp']: #['nsp1','nsp2','nsp3','nsp4','nsp5','nsp6','nsp8','nsp9','nsp10','nsp13','nsp14','nsp15','nsp16','NTD', 'RBD']:
                            triplet_position -= 1
                        #print(position, triplet_position, mut)
                        if triplet_position == 0: #triplet_position == 1:
                            nuc_triplet = subset_fasta.loc[position-1:position+1].to_numpy() #original nucleotide triplet; indexed to accomodate fasta starting at 1
                            mut_triplet = np.copy(nuc_triplet)
                            mut_triplet[0] = mut #mutated nucleotide triplet
                        elif triplet_position == 1: #triplet_position == 2:
                            nuc_triplet = subset_fasta.loc[position-2:position].to_numpy()
                            mut_triplet = np.copy(nuc_triplet)
                            mut_triplet[1] = mut
                        else:
                            nuc_triplet = subset_fasta.loc[position-3:position-1].to_numpy()
                            mut_triplet = np.copy(nuc_triplet)
                            mut_triplet[2] = mut
                        #print(nuc_triplet, mut_triplet)
                        nuc_triplet = np.where(nuc_triplet=='T', 'U', nuc_triplet) #replace T->U
                        mut_triplet = np.where(mut_triplet=='T', 'U', mut_triplet) #replace T->U
                        aa_orig = aa_codes[nuc_order.index(nuc_triplet[0]),nuc_order.index(nuc_triplet[1]),nuc_order.index(nuc_triplet[2])] #original amino acid
                        aa_mut = aa_codes[nuc_order.index(mut_triplet[0]),nuc_order.index(mut_triplet[1]),nuc_order.index(mut_triplet[2])] #mutated amino acid
                        aa_change = aa_orig != aa_mut #check if amino acid changed
                        #print(position, mut)
                        #print(output_mat, output_mat.shape)
                        position -= 1 #reindex position
                        matching_mut = reduce(np.intersect1d, (np.where(output_mat[:,0]==gene)[0], np.where(output_mat[:,1]==str(position))[0], np.where(output_mat[:,3]==mut_triplet[triplet_position])[0])) #check if mutation already in output_mat
                        #format: region, position, original nuc, mut nuc, original triplet, mut triplet, mut name (nuc), original aa, mut aa, orig aa name, mut aa name, mut name (aa), aa change, variant frequencies
                        mut_row = np.array([[gene, position, nuc_triplet[triplet_position], mut_triplet[triplet_position], ''.join(nuc_triplet), ''.join(mut_triplet), nuc_triplet[triplet_position]+str(position)+mut_triplet[triplet_position], aa_abbrevs[aa_orig], aa_abbrevs[aa_mut], aa_orig, aa_mut, aa_abbrevs[aa_orig]+str(int(np.floor((position-positions[0])/3)))+aa_abbrevs[aa_mut], aa_change] + [np.nan]*len(variant_order)])

                        #mutation not found
                        if not matching_mut.size > 0:
                            #add mutation
                            if output_mat.size > 0:
                                output_mat = np.append(output_mat, mut_row, axis=0)
                            else:
                                output_mat = mut_row
                            #get index of new mutation row
                            matching_mut = reduce(np.intersect1d, (np.where(output_mat[:,0]==gene)[0], np.where(output_mat[:,1]==str(position))[0], np.where(output_mat[:,3]==mut_triplet[triplet_position])[0]))
                        
                        #update mutation row with frequency for variant
                        output_mat[matching_mut[0], 13+variant_order.index(variant)] = variant_muts.loc[index, mut] / seq_counts[variant]'''
                    
                    if variant_muts.loc[index, mut+'_freq'] > .5:
                        
                        #triplet isn't centered on mutation
                        position = int(variant_muts.loc[index,'position']) #position of mutation
                        print(variant, position, variant_muts.loc[index, mut+'_freq'])
                        triplet_position = (position - positions[0]) % 3
                        if gene == 'rdrp':
                            position += 1
                        #accounts for open reading frame differences
                        if gene not in ['rdrp']:
                            triplet_position -= 1

                        if triplet_position == 0: #triplet_position == 1:
                            nuc_triplet = subset_fasta.loc[position-1:position+1].to_numpy() #original nucleotide triplet; indexed to accomodate fasta starting at 1
                            mut_triplet = np.copy(nuc_triplet)
                            mut_triplet[0] = mut #mutated nucleotide triplet
                        elif triplet_position == 1: #triplet_position == 2:
                            nuc_triplet = subset_fasta.loc[position-2:position].to_numpy()
                            mut_triplet = np.copy(nuc_triplet)
                            mut_triplet[1] = mut
                        else:
                            nuc_triplet = subset_fasta.loc[position-3:position-1].to_numpy()
                            mut_triplet = np.copy(nuc_triplet)
                            mut_triplet[2] = mut
                        #print(nuc_triplet, mut_triplet)
                        nuc_triplet = np.where(nuc_triplet=='T', 'U', nuc_triplet) #replace T->U
                        mut_triplet = np.where(mut_triplet=='T', 'U', mut_triplet) #replace T->U
                        aa_orig = aa_codes[nuc_order.index(nuc_triplet[0]),nuc_order.index(nuc_triplet[1]),nuc_order.index(nuc_triplet[2])] #original amino acid
                        aa_mut = aa_codes[nuc_order.index(mut_triplet[0]),nuc_order.index(mut_triplet[1]),nuc_order.index(mut_triplet[2])] #mutated amino acid
                        aa_change = aa_orig != aa_mut #check if amino acid changed
                        #print(position, mut)
                        #print(output_mat, output_mat.shape)
                        position -= 1 #reindex position
                        matching_mut = reduce(np.intersect1d, (np.where(output_mat[:,0]==gene)[0], np.where(output_mat[:,1]==str(position))[0], np.where(output_mat[:,3]==mut_triplet[triplet_position])[0])) #check if mutation already in output_mat
                        #format: region, position, original nuc, mut nuc, original triplet, mut triplet, mut name (nuc), original aa, mut aa, orig aa name, mut aa name, mut name (aa), aa change, variant frequencies
                        mut_row = np.array([[gene, position, nuc_triplet[triplet_position], mut_triplet[triplet_position], ''.join(nuc_triplet), ''.join(mut_triplet), nuc_triplet[triplet_position]+str(position)+mut_triplet[triplet_position], aa_abbrevs[aa_orig], aa_abbrevs[aa_mut], aa_orig, aa_mut, aa_abbrevs[aa_orig]+str(int(np.floor((position-positions[0])/3)))+aa_abbrevs[aa_mut], aa_change] + [np.nan]*len(variant_order)])

                        #mutation not found
                        if not matching_mut.size > 0:
                            #add mutation
                            if output_mat.size > 0:
                                output_mat = np.append(output_mat, mut_row, axis=0)
                            else:
                                output_mat = mut_row
                            #get index of new mutation row
                            matching_mut = reduce(np.intersect1d, (np.where(output_mat[:,0]==gene)[0], np.where(output_mat[:,1]==str(position))[0], np.where(output_mat[:,3]==mut_triplet[triplet_position])[0]))
                        
                        #update mutation row with frequency for variant
                        output_mat[matching_mut[0], 13+variant_order.index(variant)] = variant_muts.loc[index, mut+'_freq']
        
    output_mat = pd.DataFrame(output_mat, columns=['region','position','original','mutation','original_nucleotide','mutated_nucleotide','mut_shortened','original_aa_code','mutated_aa_code','original_amino_acid','mutated_amino_acid','aa_shortened','aa_change']+variant_order).sort_values(['region','position'])
    
    #add in values for selected mutations
    for variant in variant_order:
        for row in output_mat.index:
            #read in variant muts again
            variant_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder][0]
            variant_muts = pd.read_csv('sim_ref_data/'+variant_folder+'/collapsed_mutation_list.csv', index_col=0, header=0)

            #update mutation row with frequency for variant if mutation exists
            if float(output_mat.loc[row,'position']) in variant_muts['position']:
                matching_positions = variant_muts[variant_muts['position']==float(output_mat.loc[row,'position'])] #subset to mutations with matching position
                print(matching_positions)
                mut = output_mat.loc[row,'mutation'] #find which resulting nucleotide was pulled
                if mut == 'U': #un-convert nucleotide
                    mut = 'T'
                matching_mut = matching_positions.loc[:,mut+'_freq'] #subset to matching mutation nucleotide
                #print(matching_mut)
                if not matching_mut.empty: #check if mutation exists in variant_muts
                    output_mat.loc[row, variant] = matching_mut.values[0] #update output df with rate
    #add columns to count persistent/transiant/omicron+sublineages mutations
    count_columns = pd.DataFrame(np.zeros([output_mat.shape[0],3]), index=output_mat.index, columns=['persistent_count','transient_count','omicron+subs_count'])
    '''for row in output_mat.index:
        count_columns.loc[row, 'persistent_count'] = np.count_nonzero(np.isnan(output_mat.loc[row, ['alpha','delta','kraken','omicron','pirola']].to_numpy().astype('float')))
        count_columns.loc[row, 'transient_count'] = np.count_nonzero(np.isnan(output_mat.loc[row, ['beta','epsilon','eta','gamma','hongkong','iota','kappa','lambda','mu']].to_numpy().astype('float')))
        count_columns.loc[row, 'omicron+subs_count'] = np.count_nonzero(~np.isnan(output_mat.loc[row, ['hongkong','kraken','omicron','pirola']].to_numpy().astype('float')))
    output_mat = pd.concat([output_mat, count_columns], axis=1)'''

    print(output_mat)
    output_mat.to_csv('simulation_output/rdrp_and_spike/all_muts.csv')
    
    '''
    output_mat.to_numpy()
    AA_analysis_df = pd.DataFrame(np.zeros([len(aa_abbrevs),len(aa_abbrevs)]), index=aa_abbrevs.values(), columns=aa_abbrevs.values())
    for row_index in range(output_mat.shape[0]):
        AA_analysis_df.loc[output_mat[row_index, 7], output_mat[row_index, 8]] += 1
    AA_analysis_df.to_csv('simulation_output/rdrp_and_spike/aa_analysis.csv')

    mut_analysis_df = pd.DataFrame(np.zeros([19,len(variant_order)]), index=['nsp'+str(i+1) for i in range(11)]+['rdrp']+['nsp'+str(i) for i in range(13,17)]+['NTD','RBD','SD1_2'], columns=variant_order)
    for row_index in range(output_mat.shape[0]):
        print(output_mat[row_index,:])
        print(np.where(output_mat[row_index,13:].astype(float)>0))
        matching_cols = np.where(output_mat[row_index,13:].astype(float)>0)[0]
        #check if mutation is unique
        if len(matching_cols) == 1:
            mut_analysis_df.loc[output_mat[row_index,0], variant_order[matching_cols[0]]] += 1

        #special case for mutation originating in omicron and being passed on to kraken/pirola/hongkong      
        elif np.all(matching_cols == np.array([6,9,12,13])) or np.all(matching_cols == np.array([12,13])) or np.all(matching_cols == np.array([9,12])) or np.all(matching_cols == np.array([6,12])):
            mut_analysis_df.loc[output_mat[row_index,0], variant_order[12]] += 1
    print(mut_analysis_df)
    mut_analysis_df.to_csv('simulation_output/rdrp_and_spike/unique_mut_counts.csv')
    '''          

#calc triplet counts across entire genome after simulation. Uses mut_dicts
#has a version for independent variants and averaged variants
def calc_triplet_counts_post_sim(contexts, variant_order, thresholds, num_sims):
    #read in reference fasta
    fasta = np.asarray([*get_fasta()][:-1])
    #default triplet count
    triplet_counts = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=['T','G','C','A'])
    for gene, gene_data in gene_info.items():
        if gene not in ['ORF1a', 'ORF9b', 'ORF9c', 'ORF3b']:
            gene_position = gene_data[-1]
            for triplet_index in range(gene_position[0], gene_position[1], 3):
                triplet = fasta[triplet_index:triplet_index+3]
                triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
    #triplet_counts.to_csv('simulation_output/ref_triplet_counts_2.csv')

    #store averaged_sim_triplet_count matrices for final analysis
    averaged_triplet_counts_dict = {}

    
    #loop through each sim_type,context_type,variant combo
    '''for sim_type in ['global', 'gene_specific']:
        print(sim_type)
        for context_type in contexts:
            print(context_type)
            for variant in variant_order:
                #check if averaged triplet counts have been created already
                if not os.path.exists('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/triplet_counts/averaged_sim_triplet_counts.csv'):
                    print(variant)
                    #loop through each mut dict
                    for file in os.listdir('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/mut_dicts/'+str(thresholds[0])):
                        #print(file)
                        with open('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/mut_dicts/'+str(thresholds[0])+'/'+file) as f:
                            mut_dict = f.readlines()[:-1]

                        #pull mutations from mut dict and modify sim_triplet_counts to account for mutations
                        sim_triplet_counts = triplet_counts.copy()
                        #print('start')
                        #print(sim_triplet_counts)
                        for mut in mut_dict:
                            mut_position = int(re.search(r'\d+\[', mut).group(0)[:-1])
                            valid_position = False
                            for gene in gene_info.keys():
                                if gene not in ['ORF1a', 'ORF3b', 'ORF9b', 'ORF9c']:
                                    if mut_position > gene_info[gene][1][0] and mut_position < gene_info[gene][1][1]:
                                        valid_position = gene_info[gene][1][0]
                            if valid_position != False:
                                mut = re.search(r'\[.*\]', mut).group(0)
                                if ',' in mut:
                                    mut = mut.split(',')[-1]
                                mut = re.search(r'([A-Z]+)->([A-Z]+)', mut)
                                original = mut.group(1)
                                mut = mut.group(2)
                                #print(mut_position, original, mut)
                                #third position in triplet changed
                                if (mut_position-valid_position) %3 == 0:
                                    #tac -> tat
                                    triplet = fasta[mut_position-3:mut_position]
                                    sim_triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] -= 1 #t[a]c
                                    sim_triplet_counts.loc[triplet[0]+'[X>Y]'+mut[1], triplet[1]] += 1 #t[a]t
                                    #print('at third position')
                                elif (mut_position-valid_position) %3 == 1:
                                    triplet = fasta[mut_position-1:mut_position+2]
                                    sim_triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] -= 1
                                    sim_triplet_counts.loc[mut[1]+'[X>Y]'+triplet[-1], triplet[1]] += 1
                                    #print('at first position')
                                else:
                                    triplet = fasta[mut_position-2:mut_position+1]
                                    sim_triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] -= 1
                                    sim_triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], mut[1]] += 1
                                    #print('at second position')
                                #print(triplet)
                                #sim_triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] -= 1
                                #sim_triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], mut[1]] += 1
                        sim_triplet_counts.to_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/triplet_counts/triplet_counts_'+str(re.search(r'\d+', file).group())+'.csv')
                        #print('end')
                        #print(sim_triplet_counts)
                    
                    #averaged analysis
                    averaged_sim_triplet_counts = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=['T','G','C','A'])
                    for file in os.listdir('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/triplet_counts/'):
                        averaged_sim_triplet_counts += pd.read_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/triplet_counts/'+file, index_col=0, header=0)
                    averaged_sim_triplet_counts = averaged_sim_triplet_counts / num_sims
                    averaged_sim_triplet_counts.to_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/triplet_counts/averaged_sim_triplet_counts.csv')
                    averaged_triplet_counts_dict[sim_type+'_'+context_type+'_'+variant] = averaged_sim_triplet_counts.copy()
                    print(averaged_sim_triplet_counts)
    
                #read in already calculated averaged triplet counts
                else:
                    averaged_triplet_counts_dict[sim_type+'_'+context_type+'_'+variant] = pd.read_csv('simulation_output/'+sim_type+'/'+context_type+'/'+variant+'/triplet_counts/averaged_sim_triplet_counts.csv', index_col=0, header=0)
    #'''


    #final analysis for independent variants
    #calc correlation between variant averaged triplet counts
    '''print('final analysis')
    final_triplet_analysis = pd.DataFrame()
    for sim_type in ['global', 'gene_specific']:
        context_df = pd.DataFrame()
        for context_type in contexts:
            variant_df = pd.DataFrame(np.zeros([len(variant_order)*2,len(variant_order)+1]), index=pd.MultiIndex.from_product([['corr','t'], variant_order], names=['stat', 'variant']), columns=variant_order+['ref'])
            for variant in variant_order:
                for variant_2 in variant_order:
                    df_1 = averaged_triplet_counts_dict[sim_type+'_'+context_type+'_'+variant]
                    df_2 = averaged_triplet_counts_dict[sim_type+'_'+context_type+'_'+variant_2]
                    corr = np.corrcoef(df_1.values.flatten(), df_2.values.flatten())[0,1]
                    t_stat = corr * np.sqrt((64-2)/(1-(corr**2)))
                    #print(corr, t_stat)
                    variant_df.loc[('corr',variant), variant_2] = corr
                    variant_df.loc[('t', variant), variant_2] = t_stat
                #reference triplet counts
                df_1 = averaged_triplet_counts_dict[sim_type+'_'+context_type+'_'+variant]
                corr = np.corrcoef(df_1.values.flatten(), triplet_counts.values.flatten())[0,1]
                t_stat = corr * np.sqrt((64-2)/(1-(corr**2)))
                variant_df.loc[('corr',variant), 'ref'] = corr
                variant_df.loc[('t', variant), 'ref'] = t_stat
            print(variant_df)
            context_df = pd.concat([context_df, variant_df], axis=1)
        final_triplet_analysis = pd.concat([final_triplet_analysis, context_df], axis=0)
    final_triplet_analysis.index = pd.MultiIndex.from_product([['global','gene_specific'],['corr','t'], variant_order], names=['sim_type','stat', 'variant'])
    final_triplet_analysis.columns = pd.MultiIndex.from_product([contexts, variant_order+['ref']], names=['context_type', 'variant'])
    final_triplet_analysis.to_csv('simulation_output/final_triplet_counts.csv')
    #'''

    #final analysis for averaged variants
    #calc correlation between variant reference triplet counts and their sim results
    #get triplet counts for each variant reference
    variant_triplet_dicts = {}
    ref_data_folders = [folder for folder in os.listdir('sim_ref_data') if '_full_clade' in folder]
    for variant in variant_order:
        variant_folder = [folder for folder in ref_data_folders if '('+variant+')' in folder][0]
        with open('sim_ref_data/'+variant_folder+'/nucleotide-consensus.fasta', 'r') as f:
            fasta = np.asarray([*f.readlines()[1]][:-1])

        #variant reference triplet count
        var_triplet_counts = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=['T','G','C','A'])
        var_context_counts = var_triplet_counts.copy()
        for gene, gene_data in gene_info.items():
            #if gene not in ['ORF1a', 'ORF9b', 'ORF9c', 'ORF3b']:
            if gene in ['S']:
                gene_position = gene_data[-1]
                for triplet_index in range(gene_position[0], gene_position[1], 3):
                    triplet = fasta[triplet_index:triplet_index+3]
                    var_triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
                for triplet_index in range(gene_position[0]+1, gene_position[1]-1):
                    triplet = fasta[triplet_index-1:triplet_index+2]
                    var_context_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
        variant_triplet_dicts[variant] = var_triplet_counts
        var_triplet_counts.to_csv('simulation_output/'+variant+'_S_triplet_counts.csv')
        var_context_counts.to_csv('simulation_output/'+variant+'_S_context_counts.csv')

    #independent variants
    '''final_triplet_analysis = pd.DataFrame()
    for sim_type in ['global', 'gene_specific']:
        context_df = pd.DataFrame()
        for context_type in contexts:
            variant_df = pd.DataFrame(np.zeros([4,len(variant_order)]), index=pd.MultiIndex.from_product([['var_ref','true_ref'],['corr','t']], names=['ref_type','stat']), columns=variant_order)
            for variant in variant_order:
                #variant ref
                df_1 = averaged_triplet_counts_dict[sim_type+'_'+context_type+'_'+variant]
                df_2 = variant_triplet_dicts[variant]
                corr = np.corrcoef(df_1.values.flatten(), df_2.values.flatten())[0,1]
                t_stat = corr * np.sqrt((64-2)/(1-(corr**2)))
                variant_df.loc[('var_ref','corr'), variant] = corr
                variant_df.loc[('var_ref','t'), variant] = t_stat
                #global ref
                corr = np.corrcoef(df_1.values.flatten(), triplet_counts.values.flatten())[0,1]
                t_stat = corr * np.sqrt((64-2)/(1-(corr**2)))
                variant_df.loc[('true_ref','corr'), variant] = corr
                variant_df.loc[('true_ref','t'), variant] = t_stat
            context_df = pd.concat([context_df, variant_df], axis=0)
        final_triplet_analysis = pd.concat([final_triplet_analysis, context_df], axis=0)
    final_triplet_analysis.index = pd.MultiIndex.from_product([['global','gene_specific'],contexts,['var_ref','true_ref'],['corr','t']], names=['sim_type','context_type','ref_type','stat'])
    final_triplet_analysis.to_csv('simulation_output/final_triplet_counts.csv')
    #'''    

#convert fastas for vaccine targets into triplet counts and reference fasta
def vaccine_triplet_counts():
    '''codon_map = {'TTT':8/9,'TTG':7/9,'TTC':8/9,'TTA':7/9, 'TGT':8/9,'TGG':9/9,'TGC':8/9,'TGA':8/9, 'TCT':6/9,'TCG':6/9,'TCC':6/9,'TCA':6/9, 'TAT':8/9,'TAG':8/9,'TAC':8/9,'TAA':7/9,
                'GTT':6/9,'GTG':6/9,'GTC':6/9,'GTA':6/9, 'GGT':6/9,'GGG':6/9,'GGC':6/9,'GGA':6/9, 'GCT':6/9,'GCG':6/9,'GCC':6/9,'GCA':6/9, 'GAT':8/9,'GAG':8/9,'GAC':8/9,'GAA':8/9,
                'CTT':6/9,'CTG':5/9,'CTC':6/9,'CTA':5/9, 'CGT':6/9,'CGG':5/9,'CGC':6/9,'CGA':5/9, 'CCT':6/9,'CCG':6/9,'CCC':6/9,'CCA':6/9, 'CAT':8/9,'CAG':8/9,'CAC':8/9,'CAA':8/9,
                'ATT':8/9,'ATG':9/9,'ATC':8/9,'ATA':8/9, 'AGT':8/9,'AGG':7/9,'AGC':8/9,'AGA':7/9, 'ACT':6/9,'ACG':6/9,'ACC':6/9,'ACA':6/9, 'AAT':8/9,'AAG':8/9,'AAC':8/9,'AAA':8/9}'''
    '''
    ttttt: ttt first position = 3 changes, ttt second position = 3 changes, ttt third position = 2 changes : ttt:[3,3,2]
    tgcat: tgc first position = 3 changes, gca second position = 3 changes, cat third position = 0 changes : gca:[3,3,0]
    '''
    codon_map = {'TTT':[3,3,2],'TTG':[2,3,2],'TTC':[3,3,2],'TTA':[2,3,2], 'TGT':[3,3,2],'TGG':[3,3,3],'TGC':[3,3,2],'TGA':[3,2,3], 'TCT':[3,3,0],'TCG':[3,3,0],'TCC':[3,3,0],'TCA':[3,3,0], 'TAT':[3,3,2],'TAG':[3,3,2],'TAC':[3,3,2],'TAA':[3,2,2],
                'GTT':[3,3,0],'GTG':[3,3,0],'GTC':[3,3,0],'GTA':[3,3,0], 'GGT':[3,3,0],'GGG':[3,3,0],'GGC':[3,3,0],'GGA':[3,3,0], 'GCT':[3,3,0],'GCG':[3,3,0],'GCC':[3,3,0],'GCA':[3,3,0], 'GAT':[3,3,2],'GAG':[3,3,2],'GAC':[3,3,2],'GAA':[3,3,2],
                'CTT':[3,3,0],'CTG':[2,3,0],'CTC':[3,3,0],'CTA':[2,3,0], 'CGT':[3,3,0],'CGG':[2,3,0],'CGC':[3,3,0],'CGA':[2,3,0], 'CCT':[3,3,0],'CCG':[3,3,0],'CCC':[3,3,0],'CCA':[3,3,0], 'CAT':[3,3,2],'CAG':[3,3,2],'CAC':[3,3,2],'CAA':[3,3,2],
                'ATT':[3,3,1],'ATG':[3,3,3],'ATC':[3,3,1],'ATA':[3,3,1], 'AGT':[3,3,2],'AGG':[2,3,2],'AGC':[3,3,2],'AGA':[2,3,2], 'ACT':[3,3,0],'ACG':[3,3,0],'ACC':[3,3,0],'ACA':[3,3,0], 'AAT':[3,3,2],'AAG':[3,3,2],'AAC':[3,3,2],'AAA':[3,3,2]}
    
    context_counts = {}
    for file_index, vaccine_file in enumerate(['pfizer_spike_vaccine.fasta', 'moderna_spike_vaccine.fasta', 'EPI_ISL_402124.fasta', 'EPI_ISL_402124.fasta']): #
        with open(vaccine_file, 'r') as f:
            lines=f.readlines()
        seq = lines[1].strip()
        seq_copy = seq
        if file_index == 3:
            seq = seq[subset_genes['S'][0]:subset_genes['S'][1]]
            vaccine_file = 'S'
        triplet_counts = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=['T','G','C','A'])
        all_contexts = triplet_counts.copy()
        meaningful_contexts = triplet_counts.copy()

        #normal triplet count
        for i in range(0,len(seq),3):
            triplet = seq[i:i+3]
            if len(triplet) == 3:
                triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] += 1

        #all context count
        for i in range(1, len(seq)-1):
            triplet = seq[i-1:i+2]
            if len(triplet) == 3: #shouldnt be necessary
                all_contexts.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] += 1

        #meaningful context count
        #each triplet has a maximum of 9 mutations, of which x are degenerate so 9-x / 9
        if 'vaccine' in vaccine_file:
            seq = 'G'+seq+'T'
        elif file_index == 3:
            seq = seq_copy[subset_genes['S'][0]-1:subset_genes['S'][1]+1] #subset_genes['S'][1]+1
            #print(seq)
        for triplet_index in range(1,len(seq)-1,3):
            triplet = seq[triplet_index-1:triplet_index+4]
            #print(triplet)
            #TGCAT
            for nuc_index in range(1,4):
                #1=TGC, 2=GCA, 3=CAT
                nuc_triplet = triplet[nuc_index-1:nuc_index+2]
                #print(nuc_index, nuc_triplet)
                #print(nuc_triplet[0]+'[X>Y]'+nuc_triplet[2], nuc_triplet[1], codon_map[triplet[1:-1]][nuc_index-1])
                meaningful_contexts.loc[nuc_triplet[0]+'[X>Y]'+nuc_triplet[2], nuc_triplet[1]] += codon_map[triplet[1:-1]][nuc_index-1]
        meaningful_contexts = meaningful_contexts / 3
                

        triplet_counts.to_csv('simulation_output/'+vaccine_file.split('_')[0]+'_triplet_counts.csv')
        all_contexts.to_csv('simulation_output/'+vaccine_file.split('_')[0]+'_all_contexts.csv')
        meaningful_contexts.to_csv('simulation_output/'+vaccine_file.split('_')[0]+'_meaningful_contexts.csv')
        #context_counts[vaccine_file.split('_')[0]] = all_contexts
        context_counts[vaccine_file.split('_')[0]] = meaningful_contexts
        #print(all_contexts)
    return context_counts
    
#correlate vaccine context counts to variant matrices
def correlate_context_counts(context_counts, global_avg_subset_mat, variant_order, shape):
    print(context_counts)

    if shape[0] > 12:
        #change global_avg_subset_mat to be nx4 instead of nx12 for correlation with context_counts
        #global_avg_subset_mat = torch.cat([global_avg_subset_mat, torch.tensor(np.zeros([global_avg_subset_mat.shape[0],4,12]))], dim=1)
        #collapse columns of global_avg_subset_mat
        global_collapsed = torch.stack([torch.concat([torch.stack([torch.sum(global_avg_subset_mat[variant_index,:,i:i+3], dim=(1)) for i in range(0, global_avg_subset_mat.shape[1], 3)]), torch.tensor(np.zeros([4,4]))], axis=1).transpose(1,0) for variant_index in range(len(variant_order))])

    else:
        #collapse columns of global_avg_subset_mat
        global_collapsed = torch.stack([torch.stack([torch.sum(global_avg_subset_mat[variant_index,:,i:i+3], dim=(1)) for i in range(0, global_avg_subset_mat.shape[1], 3)]).transpose(1,0) for variant_index in range(len(variant_order))])

    print(global_collapsed.shape, global_collapsed[0])
    #correlate vaccine context counts to variant mut matrices
    output_df = pd.DataFrame(np.zeros([len(context_counts)*2, len(variant_order)]), index=pd.MultiIndex.from_product([['corr','p'],context_counts.keys()], names=['stat','contexts']), columns=variant_order)
    for context_key,context_mat in context_counts.items():
        for variant_index in range(global_avg_subset_mat.shape[0]):
            print(context_key+' vs '+variant_order[variant_index])
            #collapse mutation columns for mut rate matrix
            #mut_rate_mat = torch.transpose(torch.stack([torch.cat([torch.mean(global_avg_subset_mat[variant_index,:,i:i+3], dim=(1)), torch.tensor([0,0,0,0])]) for i in range(0,global_avg_subset_mat.shape[2],3)]),0,1)
            #print(global_avg_subset_mat[variant_index])
            #mut_rate_mat = torch.stack([torch.sum(global_avg_subset_mat[variant_index,:shape[0],i:i+3], dim=1) for i in range(0,11,3)]).transpose(0,1)
            mut_rate_mat = global_collapsed[variant_index]
            pd.DataFrame(mut_rate_mat, index=context_mat.index.to_numpy()[:shape[0]], columns=context_mat.columns).to_csv('simulation_output/context_counts/'+variant_order[variant_index]+'_mut_rate_mat.csv')
            #print(mut_rate_mat)
            #corr = torch.corrcoef(torch.stack([torch.tensor(context_mat.values[:shape[0],:shape[1]]).flatten(), mut_rate_mat[:shape[0],:shape[1]].flatten()]))[0,1].item()
            #t_stat = corr * np.sqrt((mut_rate_mat[:12,:].flatten().shape[0]-2)/(1-(corr**2)))

            corr = stats.pearsonr(context_mat.to_numpy()[:shape[0],:shape[1]].flatten(), mut_rate_mat[:shape[0],:shape[1]].flatten())

            #print(corr, t_stat)
            output_df.loc[('corr',context_key), variant_order[variant_index]] = corr[0]
            output_df.loc[('p',context_key), variant_order[variant_index]] = corr[1]
            fig, axs = plt.subplots(figsize=(8,7))
            sns.heatmap(pd.DataFrame(mut_rate_mat, index=context_mat.index.to_numpy()[:shape[0]], columns=context_mat.columns), cmap='Greys', ax=axs, xticklabels=context_mat.columns, yticklabels=context_mat.index)
            axs.set_title(variant_order[variant_index]+'_context_counts')
            plt.savefig('simulation_output/context_counts/'+variant_order[variant_index]+'_context_counts.png')
            plt.close()
    
    #correlate grouped variants
    grouped_dict = {'g1': torch.stack([global_collapsed[variant_order.index(variant)].flatten() for variant in ['alpha', 'delta', 'gamma', 'omicron', 'pirola', 'kraken']]), 
                    'g2': torch.stack([global_collapsed[variant_order.index(variant)].flatten() for variant in ['beta', 'epsilon', 'eta', 'iota', 'kappa', 'lambda', 'mu']])}
    output_df_2 = pd.DataFrame(np.ones([2*len(context_counts),len(grouped_dict)]), index=pd.MultiIndex.from_product([['corr','p'],context_counts.keys()], names=['stat','contexts']), columns=grouped_dict.keys())
    for context_key, context_mat in context_counts.items():
        for var_index, var_df in grouped_dict.items():
            print(context_key+' vs '+var_index)
            corr = stats.pearsonr(np.tile(context_mat.to_numpy()[:shape[0],:].flatten(), var_df.shape[0]), var_df.flatten())
            output_df_2.loc[('corr',context_key),var_index] = corr[0]
            output_df_2.loc[('p',context_key),var_index] = corr[1]




    '''#correlate vaccine context counts and genome context counts
    second_df = pd.DataFrame(np.zeros([len(context_counts)*2,len(context_counts)]), index=pd.MultiIndex.from_product([['corr', 't'],context_counts.keys()], names=['stat','contexts']), columns=context_counts.keys())
    for context_key,context_mat in context_counts.items():
        for context_key_2,context_mat_2 in context_counts.items():
            corr = torch.corrcoef(torch.stack([torch.tensor(context_mat.values).flatten(), torch.tensor(context_mat_2.values).flatten()]))[0,1].item()
            if context_key != context_key_2:
                t_stat = corr * np.sqrt((mut_rate_mat.flatten().shape[0]-2)/(1-(corr**2)))
            else:
                t_stat = np.inf
            second_df.loc[('corr',context_key), context_key_2] = corr
            second_df.loc[('t',context_key), context_key_2] = t_stat
    #output_df = pd.concat([output_df, second_df], axis=1)

    #correlate vaccine context counts, genome context counts, gene context counts
    third_df = pd.DataFrame(np.zeros([len(context_counts)*2,len(subset_genes.keys())]), index=pd.MultiIndex.from_product([['corr','t'],context_counts.keys()], names=['stat', 'contexts']), columns=subset_genes.keys())
    for context_key,context_mat in context_counts.items():
        for gene in subset_genes.keys():
            gene_mat = pd.read_csv('simulation_output/context_counts/genes/'+gene+'_context_counts.csv', index_col=0, header=0)
            corr = torch.corrcoef(torch.stack([torch.tensor(context_mat.values).flatten(), torch.tensor(gene_mat.values).flatten()]))[0,1].item()
            t_stat = corr * np.sqrt((gene_mat.size-2)/(1-(corr**2)))
            third_df.loc[('corr',context_key), gene] = corr
            third_df.loc[('t',context_key), gene] = t_stat
    output_df = pd.concat([output_df, second_df, third_df], axis=1)'''
    output_df = pd.concat([output_df, output_df_2], axis=1)
    output_df.to_csv('simulation_output/context_counts_comparison_correlations.csv')

'''#generate context counts for each gene in reference fasta, indexed according to subset_genes
def gene_context_counts():
    #get reference fasta
    fasta = pd.Series([*get_fasta()[:-1]])
    #dict to store context count dfs
    context_counts = {}
    #loop through each gene
    all_genes = list(subset_genes.items())+list(nsp_positions.items())+list(rdrp_sub_domains.items())
    for gene, position in all_genes:
        #empty df
        context_counts_df = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=columns_shortened)
        #N
        if len(position) > 2:
            for sub_position in position:
                sub_fasta = fasta.loc[sub_position[0]:sub_position[1]-1]
                #print(sub_fasta)
                for triplet_index in range(1, sub_fasta.shape[0]-1):
                    triplet = sub_fasta.iloc[triplet_index-1:triplet_index+2].values
                    #print(triplet)
                    context_counts_df.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] += 1
        #rest
        else:
            sub_fasta = fasta.loc[position[0]:position[1]-1]
            if gene == 'S':
                with open('simulation_output/s.fasta', 'w') as f:
                    f.write(''.join(sub_fasta.values))
            for triplet_index in range(1, sub_fasta.shape[0]-1):
                triplet = sub_fasta.iloc[triplet_index-1:triplet_index+2].values
                context_counts_df.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] += 1
        print(gene, sub_fasta.shape)
        context_counts_df.to_csv('simulation_output/context_counts/genes/'+gene+'_context_counts.csv')
        context_counts[gene] = context_counts_df
        
        #plot context counts
        if not os.path.exists('simulation_output/context_counts/genes/figs'):
            os.mkdir('simulation_output/context_counts/genes/figs')
        fig, axs = plt.subplots(figsize=(3,4), layout='tight', dpi=200)
        sns.heatmap(context_counts_df, ax=axs, annot=False, cmap='Greys', linewidth=.5, linecolor='gray', xticklabels=columns_shortened_figs, yticklabels=rows_figs)
        plt.savefig('simulation_output/context_counts/genes/figs/'+gene+'.png')

    gene_context_correlations = pd.DataFrame(np.zeros([len(subset_genes.keys())*2, len(subset_genes.keys())]), index=pd.MultiIndex.from_product([['corr','p'],subset_genes.keys()], names=['stat','gene']), columns=subset_genes.keys())
    for gene, context_df in context_counts.items():
        for gene_2, context_df_2 in context_counts.items():
            corr = stats.pearsonr(context_df.values.flatten(), context_df_2.values.flatten())
            gene_context_correlations.loc[('corr',gene), gene_2] = corr[0]
            gene_context_correlations.loc[('p',gene), gene_2] = corr[1]
    gene_context_correlations.to_csv('simulation_output/context_counts/genes/correlations.csv')

    #plot total context counts
    total_context_counts = pd.DataFrame(np.zeros([16,4]), index=rows_figs, columns=columns_shortened_figs)
    for gene,mat in context_counts.items():
        if gene in ['S','E','M','N','ORF1a','ORF3a','ORF6','ORF7a','ORF7b','ORF8','ORF10']:
            total_context_counts += mat.to_numpy()
    total_context_counts.to_csv('simulation_output/context_counts/genes/total.csv')
    fig, axs = plt.subplots(figsize=(3,4), layout='tight', dpi=200)
    sns.heatmap(total_context_counts, ax=axs, annot=False, cmap='Greys', linewidth=.5, linecolor='gray')
    plt.savefig('simulation_output/context_counts/genes/figs/total.png')'''

#generate context/triplet count info for each analyzed gene
def gen_subset_gene_info(genes_dict):
    #lists to store count dfs
    context_counts,triplet_counts,triplet_fourfolds_left,triplet_fourfolds_center = [],[],[],[]
    #loop through each gene
    for gene, positions in genes_dict.items():
        #gen array of analyzable positions
        if len(positions) > 2: #N
            position_indices = np.concatenate([np.arange(positions[i][0],positions[i][1]) for i in range(len(positions))])
        else: #rest
            position_indices = np.arange(positions[0],positions[1])

        for calc_type in ['triplets','contexts']:
            if calc_type == 'triplets':
                for sites in ['fourfold','all']:
                    if sites == 'fourfold':
                        #read in valid fourfold positions for gene
                        valid_pos = pd.read_csv('sim_ref_data/fourfold_gwtc/valid_fourfold_positions/'+gene+'.csv', index_col=0, header=0).to_numpy()
                        valid_pos = np.intersect1d(valid_pos, position_indices)
                        for centering in ['left','center']:
                            #left means 4fold is 3rd position of triplet, center means 4fold in 2nd position of triplet
                            if centering == 'left':
                                triplet_fourfolds_left.append(calc_count_mat(valid_pos-1))
                                save_count_fig(triplet_fourfolds_left[-1], 'triplet_fourfolds_left/'+gene+'.png')
                            else:
                                triplet_fourfolds_center.append(calc_count_mat(valid_pos))
                                save_count_fig(triplet_fourfolds_center[-1], 'triplet_fourfolds_center/'+gene+'.png')
                    else: #calc triplet counts of all sites in gene
                        valid_pos = position_indices[::3] #every 3rd position
                        triplet_counts.append(calc_count_mat(valid_pos))
                        save_count_fig(triplet_counts[-1], 'triplet_counts/'+gene+'.png')
            else: #calc context counts of gene
                context_counts.append(calc_count_mat(position_indices))
                save_count_fig(context_counts[-1], 'context_counts/'+gene+'.png')
        
    #calc total counts
    mat_list_names = ['context_counts','triplet_counts','triplet_fourfolds_left','triplet_fourfolds_center']
    for mat_list_index, mat_list in enumerate([context_counts, triplet_counts, triplet_fourfolds_left, triplet_fourfolds_center]):
        mat_list.append(pd.DataFrame(np.sum([mat_list[i] for i in range(len(mat_list)) if list(genes_dict.keys())[i] in ['S','E','M','N','ORF1a','ORF3a','ORF6','ORF7a','ORF7b','ORF8','ORF10']], axis=0), index=rows_figs, columns=columns_shortened_figs))
        save_count_fig(mat_list[-1], mat_list_names[mat_list_index]+'/total.png')

    #save counts as single csv
    for mat_index, mats in enumerate([context_counts, triplet_counts, triplet_fourfolds_left, triplet_fourfolds_center]):
        mat_df = pd.concat(mats, axis=1)
        mat_df.columns = pd.MultiIndex.from_product([list(genes_dict.keys())+['total'],columns_shortened_figs])
        mat_df.to_csv('simulation_output/final_info/final_tables/supp_tables/'+mat_list_names[mat_index]+'.csv')
    


#compare pirola at timestamps(folders)
def compare_pirola(folders):
    pirola_dict = {folder:torch.load('sim_ref_data/pirola_timestamps/'+folder+'/global_mut_mat.pt') for folder in folders}
    fig, axs = plt.subplots(figsize=(8,7))
    sns.heatmap(pirola_dict[folders[0]], cmap='Greys', ax=axs, xticklabels=columns, yticklabels=rows, vmin=0, vmax=torch.max(pirola_dict[folders[0]])) #0.25
    axs.set_title('global subset rates')
    plt.savefig('simulation_output/pirola_comparisons/10_5_23.png')
    plt.close()
    for index, folder in enumerate(folders[1:]):
        mat_diff = pirola_dict[folder] - pirola_dict[folders[index]]
        mat_diff[mat_diff<0] = 0
        fig, axs = plt.subplots(figsize=(8,7))
        sns.heatmap(mat_diff, cmap='Greys', ax=axs, xticklabels=columns, yticklabels=rows, vmin=0) #0.25 #torch.max(pirola_dict[folders[0]])
        axs.set_title('global subset rates')
        plt.savefig('simulation_output/pirola_comparisons/'+folder+'.png')
        plt.close()
    corr_df = pd.DataFrame(np.zeros([2*len(folders), len(folders)]), index=pd.MultiIndex.from_product([['corr','t'],folders], names=['stat','timestamp']), columns=folders)
    for folder,mat in pirola_dict.items():
        for folder_2, mat_2 in pirola_dict.items():
            corr = torch.corrcoef(torch.stack([mat.flatten(), mat_2.flatten()]))[0,1].item()
            if folder != folder_2:
                t_stat = corr * np.sqrt((mat.flatten().shape[0]-2)/(1-(corr**2)))
            else:
                t_stat = np.inf
            corr_df.loc[('corr',folder), folder_2] = corr
            corr_df.loc[('t',folder), folder_2] = t_stat
    corr_df.to_csv('simulation_output/pirola_comparisons/corr.csv')

#compare mutation lists for pirola timestamps to get new mutations with each update
def compare_pirola_muts(folders):
    pirola_dict = {folder:pd.read_csv('sim_ref_data/pirola_timestamps/'+folder+'/nucleotide-mutations.csv', index_col=0, header=0) for folder in folders}
    for index,folder in enumerate(folders[1:]):
        print(pirola_dict[folder])
        pirola_dict[folder].drop(labels=pirola_dict[folders[index]].isin(pirola_dict[folder].index).index, axis=0, errors='ignore').to_csv('simulation_output/'+folder+'_diff_muts.csv')

#convert context list to context df to be correlated with mut rate mat
def convert_context_list(context_list):
    context_df = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=['T','G','C','A'])
    for triplet in context_list:
        context_df.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] += 1
    return context_df

#loop through fasta and find region with lowest correlation to global_avg_subset_mat
def gen_genome_mutability(global_mat_shortened, genes_avg_mat, window_size, variant_order, fasta):
    context_list = [] #stores each triplet in window
    correlations_dict = {} #index:[corr,variant]

    #establish first window
    for nuc_index in range(1, window_size+1):
        context_list.append(fasta[nuc_index-1:nuc_index+2].values)
    context_df = convert_context_list(context_list)

    #move through genome 1-by-1 maintaining window size
    for nuc_index in range(1+window_size, fasta.shape[0]-1): #
        if nuc_index%1000 == 0:
            print(nuc_index)
        ##remove oldest triplet from df
        triplet = context_list[0]
        context_df.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] -= 1
        context_list = context_list[1:] #drop oldest triplet
        ##add newest triplet to df
        triplet = fasta[nuc_index-1:nuc_index+2].values
        context_list.append(triplet) #append next triplet
        context_df.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] += 1
        ##rename following tensor to not overwrite df
        context_tensor = torch.tensor(context_df.values)[:global_mat_shortened.shape[1],:] #convert to tensor
        #context_tensor = torch.cat([context_tensor[:12,i].unsqueeze(1).expand(12,3) for i in range(4)], dim=1) #reshape to be 16x12
        for var_index in range(global_mat_shortened.shape[0]): #correlation between context df and mut_rate_mat
            #corr = torch.corrcoef(torch.stack([context_tensor.flatten(), global_mat_shortened[var_index].flatten()]))[0,1].item()
            #t_stat = corr * np.sqrt((context_tensor.flatten().shape[0]-2)/(1-(corr**2)))

            #print(torch.stack([context_tensor.flatten(), torch.tensor(np.ones([*context_tensor.shape])).flatten()], dim=1).shape)
            x, residuals, rank, s = np.linalg.lstsq(torch.stack([context_tensor.flatten(), torch.tensor(np.ones([*context_tensor.shape])).flatten()], dim=1), global_mat_shortened[var_index].flatten(), rcond=None)

            if nuc_index in correlations_dict.keys():
                correlations_dict[nuc_index].append([residuals,x[0],x[1],variant_order[var_index]])
            else:
                correlations_dict[nuc_index] = [[residuals,x[0],x[1],variant_order[var_index]]]
    output_df = pd.DataFrame(np.zeros([len(variant_order)*3, len([key for key in correlations_dict.keys()])]), index=pd.MultiIndex.from_product([variant_order,['SoS','b0','b1']], names=['variant', 'stat']), columns=[key for key in correlations_dict.keys()])
    for key,value in correlations_dict.items():
        for pair in value:
            output_df.loc[(pair[3], 'SoS'), key] = pair[0]
            output_df.loc[(pair[3], 'b0'), key] = pair[1]
            output_df.loc[(pair[3], 'b1'), key] = pair[2]
    return output_df

#search through the genome and find the 1k bp region where the context counts are least likely to be mutated
def search_genome(global_avg_subset_mat, genes_avg_mat, window_size=1000, variant_order=[], run_type=0):
    #run_type: 0=ref_genome, 1=var_genome, 2=both

    #append 0-row for A[X>Y]N muts in global_avg_subset_mat
    #global_avg_subset_mat = torch.cat([global_avg_subset_mat, torch.tensor(np.zeros([global_avg_subset_mat.shape[0],4,12]))], dim=1) #shape=[13,16,12]
    #change global_avg_subset_mat to be nx4 instead of nx12 for correlation with context_counts
    global_mat_shortened = torch.tensor(np.zeros([global_avg_subset_mat.shape[0],global_avg_subset_mat.shape[1],4]))
    for var_index in range(global_avg_subset_mat.shape[0]):
        col_index_result = 0
        for col_index in range(0,global_avg_subset_mat.shape[2],3):
            #print(global_avg_subset_mat[var_index,:,col_index:col_index+3])
            #print(torch.sum(global_avg_subset_mat[var_index,:,col_index:col_index+3], dim=1))
            global_mat_shortened[var_index,:,col_index_result] = torch.sum(global_avg_subset_mat[var_index,:,col_index:col_index+3], dim=1)
            col_index_result += 1

    
    #ref genome
    if run_type == 0 or run_type == 2:
        #read in reference fasta
        fasta = pd.Series([*get_fasta()[:-1]], name='ref')
        if not os.path.exists('simulation_output/genome_mutability/ref/genome_mutability.csv'):
            output_df = gen_genome_mutability(global_mat_shortened, genes_avg_mat, window_size, variant_order, fasta)
            output_df.to_csv('simulation_output/genome_mutability/ref/genome_mutability.csv')
        else:
            output_df = pd.read_csv('simulation_output/genome_mutability/ref/genome_mutability.csv', index_col=(0,1), header=0)
        #print(output_df)
        print(output_df.loc[pd.MultiIndex.from_product([variant_order,['SoS']]),:].idxmax(axis=1))
        print(output_df.loc[pd.MultiIndex.from_product([variant_order,['SoS','b0','b1']]), [val for val in output_df.loc[pd.MultiIndex.from_product([variant_order,['SoS']]),:].idxmax(axis=1)]])
        min_regions = output_df.loc[pd.MultiIndex.from_product([variant_order,['SoS']]),:].idxmax(axis=1).astype(int)

    
        for variant_key in min_regions.index:
            context_list = []
            for nuc_index in range(min_regions.loc[variant_key]-window_size, min_regions.loc[variant_key]):
                context_list.append(fasta[nuc_index-1:nuc_index+2].values)
            context_df = convert_context_list(context_list)
            context_df.to_csv('simulation_output/genome_mutability/ref/'+variant_key[0]+'_context_df.csv')
            fig, axs = plt.subplots(figsize=(8,7))
            sns.heatmap(context_df, cmap='Greys', ax=axs, xticklabels=context_df.columns, yticklabels=context_df.index)
            axs.set_title(variant_key[0]+'_context_counts')
            plt.savefig('simulation_output/genome_mutability/ref/'+variant_key[0]+'_context_counts.png')
            plt.close()
    
    if run_type == 1 or run_type == 2:
        if not os.path.exists('simulation_output/genome_mutability/var/genome_mutability.csv'):
            output_df = pd.DataFrame()
            var_folders = [folder for folder in os.listdir('sim_ref_data') if 'full_clade' in folder]
            for variant_folder in var_folders:
                variant = re.search(r'\(\w+\)', variant_folder).group(0)[1:-1]
                #read in var fasta
                with open('sim_ref_data/'+variant_folder+'/nucleotide-consensus.fasta') as f:
                    fasta = pd.Series([*f.readlines()[1][:-1]])
                output_df = pd.concat([output_df, gen_genome_mutability(global_mat_shortened[variant_order.index(variant)].unsqueeze(0), genes_avg_mat, window_size, [variant], fasta)])
                output_df.to_csv('simulation_output/genome_mutability/var/genome_mutability.csv')
        else:
            output_df = pd.read_csv('simulation_output/genome_mutability/var/genome_mutability.csv', index_col=(0,1), header=0)

        print(output_df.loc[pd.MultiIndex.from_product([variant_order,['SoS']]),:].idxmax(axis=1))
        print(output_df.loc[pd.MultiIndex.from_product([variant_order,['SoS','b0','b1']]), [val for val in output_df.loc[pd.MultiIndex.from_product([variant_order,['SoS']]),:].idxmax(axis=1)]])
        min_regions = output_df.loc[pd.MultiIndex.from_product([variant_order,['SoS']]),:].idxmax(axis=1).astype(int)

        for variant_key in min_regions.index:
            context_list = []
            variant_folder = [folder for folder in os.listdir('sim_ref_data/') if 'full_clade' in folder and variant_key[0] in folder][0]
            with open('sim_ref_data/'+variant_folder+'/nucleotide-consensus.fasta') as f:
                fasta = pd.Series([*f.readlines()[1][:-1]])
            for nuc_index in range(min_regions.loc[variant_key]-window_size, min_regions.loc[variant_key]):
                context_list.append(fasta[nuc_index-1:nuc_index+2].values)
            context_df = convert_context_list(context_list)
            context_df.to_csv('simulation_output/genome_mutability/var/'+variant_key[0]+'_context_df.csv')
            fig, axs = plt.subplots(figsize=(8,7))
            sns.heatmap(context_df, cmap='Greys', ax=axs, xticklabels=context_df.columns, yticklabels=context_df.index)
            axs.set_title(variant_key[0]+'_context_counts')
            plt.savefig('simulation_output/genome_mutability/var/'+variant_key[0]+'_context_counts.png')
            plt.close()
    #'''
    
#splice gene sequences from genome for comparison
def gen_gene_sequences():
    fasta = pd.Series([*get_fasta()[:-1]], name='ref')

    gene_positions = {gene:gene_info[gene][1] for gene in gene_info.keys()} 
    for gene,positions in gene_positions.items():
        fasta.loc[positions[0]:positions[1]-1].to_csv('sim_ref_data/genes/'+gene+'_sequence.csv') 

#correlate estimated muts for each variant using data from multiplying variant rate matrices with pfizer meaningful muts
def correlate_estimated_muts(variant_order):
    #variant_order = ['alpha', 'beta', 'delta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu', 'omicron', 'pirola', 'kraken']
    big_df = pd.read_excel('simulation_output/output_spreadsheet.xlsx', sheet_name='Vaccine_3_31_24', skiprows=219, nrows=38)
    df_indices = big_df.iloc[2:18,0]
    df_columns = big_df.iloc[1,1:5]
    
    pfizer_meaningful = big_df.iloc[2:18,1:5]
    pfizer_meaningful.index = df_indices
    pfizer_meaningful.columns = df_columns
    print(pfizer_meaningful)

    dfs_pfizer = {}
    dfs_moderna = {}
    for row_index in range(2,23,18):
        for col_index in range(6,41,5):
            if big_df.iloc[row_index-2, col_index] in variant_order:
                var_df = big_df.iloc[row_index:row_index+16, col_index:col_index+4].astype(float)
                var_df.index = df_indices
                var_df.columns = df_columns
                #print(big_df.iloc[row_index-2, col_index], var_df)
                dfs_pfizer[big_df.iloc[row_index-2, col_index]] = var_df
        for col_index in range(47,78,5):
            if big_df.iloc[row_index-2, col_index] in variant_order:
                var_df = big_df.iloc[row_index:row_index+16, col_index:col_index+4].astype(float)
                var_df.index = df_indices
                var_df.columns = df_columns
                #print(big_df.iloc[row_index-2, col_index], var_df)
                dfs_moderna[big_df.iloc[row_index-2, col_index]] = var_df
    corr_df = pd.DataFrame(np.zeros([len(variant_order)*2, len(variant_order)*2]), index=pd.MultiIndex.from_product([['corr','t'], variant_order]), columns=pd.MultiIndex.from_product([['pfizer', 'moderna'],variant_order]))
    #pfizer
    for var_1 in variant_order:
        for var_2 in variant_order:
            corr = torch.corrcoef(torch.stack([torch.tensor(dfs_pfizer[var_1].values).flatten(), torch.tensor(dfs_pfizer[var_2].values).flatten()]))[0,1].item()
            if var_1 != var_2:
                t_stat = corr * np.sqrt((dfs_pfizer[var_1].size-2)/(1-(corr**2)))
            else:
                t_stat = np.inf
            corr_df.loc[('corr',var_1), ('pfizer',var_2)] = corr
            corr_df.loc[('t',var_1), ('pfizer',var_2)] = t_stat
    #moderna
    for var_1 in variant_order:
        for var_2 in variant_order:
            corr = torch.corrcoef(torch.stack([torch.tensor(dfs_moderna[var_1].values).flatten(), torch.tensor(dfs_moderna[var_2].values).flatten()]))[0,1].item()
            if var_1 != var_2:
                t_stat = corr * np.sqrt((dfs_moderna[var_1].size-2)/(1-(corr**2)))
            else:
                t_stat = np.inf
            corr_df.loc[('corr',var_1), ('moderna',var_2)] = corr
            corr_df.loc[('t',var_1), ('moderna',var_2)] = t_stat
    corr_df.to_csv('simulation_output/estimated_muts_corr.csv')

def correlate_something(variant_order):
    big_df = pd.read_excel('simulation_output/output_spreadsheet.xlsx', sheet_name='Vaccine_3_31_24', skiprows=342, nrows=18)
    print(big_df)
    df_indices = big_df.iloc[1:,7]
    df_columns = big_df.iloc[0,8:20]
    
    spike_meaningful_ext = big_df.iloc[1:,8:20].astype(float)
    spike_meaningful_ext.index = df_indices
    spike_meaningful_ext.columns = df_columns
    print(spike_meaningful_ext)

    alpha = big_df.iloc[1:, 22:34].astype(float)
    alpha.index=df_indices
    alpha.columns=df_columns
    print(alpha)

    corr = torch.corrcoef(torch.stack([torch.tensor(spike_meaningful_ext.values).flatten(), torch.tensor(alpha.values).flatten()]))[0,1].item()
    t_stat = corr * np.sqrt((alpha.size-2)/(1-(corr**2)))
    print(corr,t_stat)

    big_df_2 = pd.read_excel('simulation_output/output_spreadsheet.xlsx', sheet_name='Vaccine_3_31_24', skiprows=372, nrows=36)
    print(big_df_2)
    df_columns_2 = big_df_2.iloc[0,1:5]

    df_dict = {}
    for row_index in range(1,34,18):
        for col_index in range(1,37,5):
            df = big_df_2.iloc[row_index:row_index+16, col_index:col_index+4].astype(float)
            if row_index == 1:
                df_name = big_df_2.columns.values[col_index-1]
            else:
                df_name = big_df_2.iloc[row_index-2, col_index-1]
            print(df_name)
            print(df)
            df.index=df_indices
            df.columns = df_columns_2
            if df_name in variant_order or df_name == 'spike meaningful':
                df_dict[df_name] = df
            #print(df)
    #print(df_dict['spike meaningful'])
    #mats = torch.stack([torch.tensor(mat.values).flatten() for mat in df_dict.values()])
    #print(torch.cov(mats.transpose(0,1)).shape)
    #output_df = pd.DataFrame(torch.cov(mats.transpose(0,1)), index=pd.MultiIndex.from_product([df_indices, df_columns_2]), columns=pd.MultiIndex.from_product([df_indices, df_columns_2]))
    #output_df.to_csv('simulation_output/correlate_something.csv')
    #for col_index in range(mats.shape[1]):
    #    if col_index+1 < mats.shape[1]:
    #        plt.scatter(mats[:,col_index], mats[:,col_index+1])


    #print(mats.shape)
    #whitened = cluster.vq.whiten(mats)
    #kmeans = cluster.vq.kmeans(whitened, 4)
    #print(kmeans)
    #plt.scatter([whitened[:,i] for i in range(whitened.shape[1])])
    #plt.scatter(kmeans[0][:,0], kmeans[0][:,1], c='r')
    #plt.show()
    #output_df = pd.DataFrame(np.ones([len(variant_order)*2, len(variant_order)]), index=pd.MultiIndex.from_product([['corr','t'],variant_order]), columns=variant_order)
    '''for var_1 in variant_order:
        for var_2 in variant_order:
            if var_1 != var_2:
                mat = df_dict[var_1] - df_dict[var_2]
                mat[mat<0] = 0
                print(var_1,var_2,mat)
                corr = torch.corrcoef(torch.stack([torch.tensor(df_dict['spike meaningful'].values).flatten(), torch.tensor(mat.values).flatten()]))[0,1].item()
                t_stat = corr * np.sqrt((mat.size-2)/(1-(corr**2)))
                output_df.loc[('corr',var_1),var_2] = corr
                output_df.loc[('t',var_1),var_2] = t_stat'''
    output_df = pd.DataFrame(np.ones([(len(variant_order)+1)*2, len(variant_order)+1]), index=pd.MultiIndex.from_product([['corr','t'],variant_order+['spike meaningful']]), columns=variant_order+['spike meaningful'])
    for var_1 in variant_order+['spike meaningful']:
        for var_2 in variant_order+['spike meaningful']:
            if var_1 != var_2:
                mat_1 = torch.sum(torch.tensor(df_dict[var_1].values) / torch.sum(torch.tensor(df_dict[var_1].values), dim=(0,1)), dim=0)
                mat_2 = torch.sum(torch.tensor(df_dict[var_2].values) / torch.sum(torch.tensor(df_dict[var_2].values), dim=(0,1)), dim=0)
                corr = torch.corrcoef(torch.stack([mat_1.flatten(), mat_2.flatten()]))[0,1].item()
                t_stat = corr * np.sqrt((mat_1.shape[0]-2)/(1-(corr**2)))
                output_df.loc[('corr',var_1),var_2] = corr
                output_df.loc[('t',var_1),var_2] = t_stat
    '''output_df = pd.DataFrame(np.zeros([len(variant_order)*2, 1]), index=pd.MultiIndex.from_product([['corr','t'],variant_order]), columns=['spike_meaningful'])
    for var in variant_order:
        mat = torch.mean(torch.tensor(df_dict[var].values), dim=1)
        meaningful = torch.mean(torch.tensor(df_dict['spike meaningful'].values), dim=1)
        corr = torch.corrcoef(torch.stack([meaningful.flatten(), mat.flatten()]))[0,1].item()
        t_stat = corr * np.sqrt((mat.shape[0]-2)/(1-(corr**2)))
        output_df.loc[('corr',var),'spike meaningful'] = corr
        output_df.loc[('t',var),'spike meaningful'] = t_stat'''
    output_df.to_csv('simulation_output/correlate_something.csv')

#look at shared muts between variants to try and remove mutations that are redundant for analysis
def calc_shared_muts_between_variants(variant_order, thresholds):
    variant_order = [variant for variant in variant_order if variant not in ['all','transient','persistent']]
    #read in data
    reference_mutations_dict = {}
    for threshold in thresholds:
        for variant in variant_order:
            variant_folder = [folder for folder in os.listdir('sim_ref_data') if 'full_clade' in folder and '('+variant+')' in folder][0]
            reference_mutations_dict[str(threshold)+'_'+variant] = pd.read_csv('sim_ref_data/'+variant_folder+'/reference_mutations/'+str(threshold)+'_reference_mutations.csv', index_col=0, header=0)

    '''#loop through thresholds and compare variant mutation lists to find overlaps
    output_df = pd.DataFrame(np.zeros([len(variant_order), len(thresholds)*2]), index=np.arange(len(variant_order)), columns=pd.MultiIndex.from_product([['count', 'muts'], thresholds]))
    var_count_df = pd.DataFrame(np.zeros([len(variant_order), len(thresholds)]), index=variant_order, columns=thresholds)
    for threshold in thresholds:
        #compile muts into single df
        merged_mutation_df = pd.concat([reference_mutations_dict[str(threshold)+'_'+variant] for variant in variant_order])
        #convert formatting
        merged_mutation_df.loc[:,'position'] = merged_mutation_df.loc[:,'old']+merged_mutation_df.loc[:,'position'].astype(str)+merged_mutation_df.loc[:,'mut']
        #print(merged_mutation_df)
        #pull all unique muts and the number of variants that share them
        positions, counts = np.unique(merged_mutation_df.loc[:,'position'], return_counts=True)
        #print(positions[counts==2])
        for share_count in range(len(variant_order)):
            output_df.loc[share_count, ('count',threshold)] = len(positions[counts==share_count])
            output_df.loc[share_count, ('muts',threshold)] = str(positions[counts==share_count])

        unique_positions = positions[counts==1]
        unique_positions = [int(position[1:-1]) for position in unique_positions]
        #print(unique_positions)
        for variant in variant_order:
            #print(variant, threshold)
            variant_folder = [folder for folder in os.listdir('sim_ref_data') if 'full_clade' in folder and '('+variant+')' in folder][0]
            var_df = reference_mutations_dict[str(threshold)+'_'+variant] 
            unique_var_df = var_df[var_df.loc[:,'position'].isin(unique_positions)]
            var_count_df.loc[variant, threshold] = unique_var_df.shape[0]
            unique_var_df.to_csv('sim_ref_data/'+variant_folder+'/reference_mutations/'+str(threshold)+'_unique_reference_mutations.csv')
    output_df = pd.concat([output_df, var_count_df], axis=1)'''

    if not os.path.exists('simulation_output/final_info/shared_muts'):
        os.mkdir('simulation_output/final_info/shared_muts')
    #method 2
    #merge mutations from each variant's reference
    output_df = pd.DataFrame(np.zeros([len(variant_order),len(thresholds)]), index=range(1,len(variant_order)+1), columns=thresholds)
    #loop through thresholds
    for threshold in thresholds:
        #dummy df to join muts onto
        merged_df = pd.DataFrame([[999999,'A','T']], index=[999999], columns=['position','old','mut'])
        #loop through variants
        for variant in variant_order:
            #merge mutations
            merged_df = merged_df.merge(reference_mutations_dict[threshold+'_'+variant].loc[:,['position','old','mut']], how='outer', on=['position','old','mut'], suffixes=[None, variant], indicator=variant)
            #reformat so variant column has 0 where mutation is not found and 1 where it is
            arr = merged_df.loc[:,variant].to_numpy()
            arr = np.where(arr=='left_only',0,1)
            merged_df[variant] = arr
        #drop dummy row
        merged_df.drop(0, axis=0, inplace=True)
        #count number of variants with mutation
        merged_df['total'] = np.sum(merged_df.loc[:,variant_order], axis=1)
        #get number of mutations shared across n variants
        totals, counts = np.unique(merged_df.loc[:,'total'], return_counts=True)
        #update output_df
        for val, count in zip(totals, counts):
            output_df.loc[val, threshold] = count
            if not os.path.exists('simulation_output/final_info/shared_muts/'+threshold):
                os.mkdir('simulation_output/final_info/shared_muts/'+threshold)
            merged_df.loc[merged_df['total'] == val].to_csv('simulation_output/final_info/shared_muts/'+threshold+'/'+str(val)+'_vars.csv')
        output_df.loc['total',threshold] = merged_df.shape[0]
    print(output_df)
    output_df.to_csv('simulation_output/final_info/shared_muts/muts_shared_across_variants.csv')



#calc t-test between variants and look at contexts of mutations
def analyze_rdrp_and_spike_df(variant_order):
    complete_df = pd.read_excel('simulation_output/output_spreadsheet.xlsx', sheet_name='rdrp and spike', nrows=150)
    rdrp_df, ntd_df, rbd_df, sd_df = 0,0,0,0
    rdrp_df = complete_df.iloc[4:17, 0:20]
    df_columns = complete_df.iloc[3,0:20]
    rdrp_df.columns = df_columns
    ntd_df = complete_df.iloc[24:65, 0:20]
    ntd_df.columns = df_columns
    rbd_df = complete_df.iloc[72:109, 0:20]
    rbd_df.columns = df_columns
    sd_df = complete_df.iloc[116:147, 0:20]
    sd_df.columns = df_columns
    print(rdrp_df)
    print(ntd_df)
    print(rbd_df)
    print(sd_df)

    t_test_df = pd.DataFrame(np.zeros([len(variant_order)*2,len(variant_order)]), index=pd.MultiIndex.from_product([['t','p'],variant_order]), columns=variant_order)
    mut_counts = pd.concat([ntd_df.iloc[-1,3:16],rbd_df.iloc[-1,3:16],sd_df.iloc[-1,3:16]], axis=1).transpose() #rdrp_df.iloc[-1,3:16],
    print(mut_counts)
    for var_1 in variant_order:
        for var_2 in variant_order:
            #print(var_1,var_2)
            if np.mean(mut_counts.loc[:,var_1].to_numpy()) <= np.mean(mut_counts.loc[:,var_2].to_numpy()):
                t,p = stats.ttest_ind(mut_counts.loc[:,var_1], mut_counts.loc[:,var_2], equal_var=False, alternative='less')
            else:
                t,p = stats.ttest_ind(mut_counts.loc[:,var_1], mut_counts.loc[:,var_2], equal_var=False, alternative='greater')
            t_test_df.loc[('t',var_1),var_2] = t
            t_test_df.loc[('p',var_1),var_2] = p
    t_test_df.to_csv('simulation_output/rdrp_and_spike_analysis.csv')

    fasta = pd.Series([*get_fasta()[:-1]], name='ref')
    context_dfs = [[],[]]
    total_context_df_g1 = pd.DataFrame(np.zeros([16,12]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=columns)
    total_context_df_g2 = total_context_df_g1.copy()

    for df in [rdrp_df, ntd_df, rbd_df, sd_df]:
        context_df_g1 = pd.DataFrame(np.zeros([16,12]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=columns)
        context_df_g2 = context_df_g1.copy()
        for index in df.index.to_numpy()[:-1]:
            row = df.loc[index,:]
            triplet = fasta.loc[row['position']-2:row['position']].to_numpy() #center on triplet and account for indexing offset
            if row['kraken'] > 0 or row['omicron'] > 0 or row['pirola'] > 0:
                context_df_g2.loc[triplet[0]+'[X>Y]'+triplet[2], row['original']+'>'+row['mutation']] += 1
            if row['alpha'] > 0 or row['beta'] > 0 or row['delta'] > 0 or row['epsilon'] > 0 or row['eta'] > 0 or row['gamma'] > 0 or row['iota'] > 0 or row['kappa'] > 0 or row['lambda'] > 0 or row['mu'] > 0:
                context_df_g1.loc[triplet[0]+'[X>Y]'+triplet[2], row['original']+'>'+row['mutation']] += 1
        context_dfs[0].append(context_df_g1)
        context_dfs[1].append(context_df_g2)
        total_context_df_g1 += context_df_g1
        total_context_df_g2 += context_df_g2

    for df in context_dfs[0]:
        total_context_df_g1 = pd.concat([total_context_df_g1, df])
    for df in context_dfs[1]:
        total_context_df_g2 = pd.concat([total_context_df_g2, df])
    
    total_context_df = pd.concat([total_context_df_g1, total_context_df_g2], axis=1)
    total_context_df.to_csv('simulation_output/rdrp_and_spike_analysis_2.csv')

#correlate spike meaningful context counts with each variant mut rate at low frequency
def correlate_meaningful_and_low_freqs():
    big_df = pd.read_excel('simulation_output/output_spreadsheet.xlsx', sheet_name='Vaccine_3_31_24', skiprows=578, nrows=38)
    #print(big_df)
    df_indices = big_df.iloc[1:17,0]
    df_columns = big_df.iloc[0,1:5]
    
    spike_meaningful = big_df.iloc[1:17,1:5].astype(float)
    spike_meaningful.index = df_indices
    spike_meaningful.columns = df_columns
    print(spike_meaningful)
    fig, ax = plt.subplots(figsize=(4,10), layout='tight')
    sns.heatmap(spike_meaningful.to_numpy(), yticklabels=rows_figs, xticklabels=columns_shortened_figs, cmap='Greys', linewidth=.5, linecolor='Grey')
    ax.set_yticklabels(rows_figs, rotation='horizontal')
    ax.set_title('Non-Degenerate Contexts in Spike Protein')
    plt.savefig('simulation_output/spike_meaningful_contexts.png')
    plt.close()
    spike_meaningful = torch.tensor(spike_meaningful.to_numpy()[:12,:])

    df_dict = {}
    for row_index in range(1,21,19):
        for col_index in range(7,36,6):
            df = big_df.iloc[row_index:row_index+16, col_index:col_index+4].astype(float)
            if row_index == 1:
                df_name = big_df.columns.values[col_index-1]
            else:
                df_name = big_df.iloc[row_index-2, col_index-1]
            print(df_name)
            print(df)
            df.index=df_indices
            df.columns = df_columns
            df_dict[df_name] = torch.tensor(df.to_numpy()[:12,:])
    grouped_dict = {'g1': torch.stack([df_dict[variant].flatten() for variant in ['alpha', 'delta', 'gamma', 'omicron', 'pirola', 'kraken']]), 
                    'g2': torch.stack([df_dict[variant].flatten() for variant in ['beta', 'epsilon', 'iota', 'aggregate']])}
    print(grouped_dict['g1'].shape)

    output_df = pd.DataFrame(np.ones([2,len(grouped_dict)]), index=['corr','p'], columns=grouped_dict.keys())
    for var_index, var_df in grouped_dict.items():
        #change to stats.pearsonr()
        #corr = torch.corrcoef(torch.stack([var_df.flatten(), spike_meaningful.flatten()]))[0,1].item()
        #print(spike_meaningful.flatten().size())
        #t_stat = corr * np.sqrt((spike_meaningful.flatten().size()[0]-2)/(1-(corr**2)))
        corr = stats.pearsonr(spike_meaningful.flatten().repeat(var_df.shape[0]), var_df.flatten())
        output_df.loc['corr',var_index] = corr[0]
        output_df.loc['p',var_index] = corr[1]
    output_df.to_csv('simulation_output/meaningful_low_freq_correlation.csv')

#perform anova to compare variant mut counts across spike protein for rdrp and spike mutations above .5 threshold
def spike_anova():
    big_df = pd.read_excel('simulation_output/output_spreadsheet.xlsx', sheet_name='rdrp and spike', skiprows=25, nrows=3)
    spike_data = big_df.iloc[:,24:38]
    print(spike_data)

    print(*[spike_data.iloc[:,i].to_numpy() for i in range(spike_data.shape[1])])
    anova = stats.f_oneway(*[spike_data.iloc[:,i].to_numpy() for i in range(spike_data.shape[1])])
    print(anova)

#create histogram of mutations from sim results across variants and iterations
def sim_result_hist(variant_order, threshold, show):
    #show: matches or all
    fasta = pd.Series([*get_fasta()[:-1]], name='ref')
    mut_df = pd.DataFrame(np.zeros([fasta.shape[0], 4]), index=fasta.index, columns=columns_shortened)
    seq_count = 0
    output_path = './simulation_output/global/full_contexts/'
    nuc_codes = {'T':1,'G':2,'C':3,'A':4, 1:'T',2:'G',3:'C',4:'A'}

    #read in all variant mutations
    '''for variant in os.listdir(output_path): #same for aggregate
        for mut_dict in os.listdir(output_path+variant+'/mut_dicts/'+str(threshold)):
            seq_count += 1
            with open(output_path+variant+'/mut_dicts/'+str(threshold)+'/'+mut_dict, 'r') as f:
                lines = f.readlines()
            for mutation_line in lines:
                position = int(re.search(r'^\d+', mutation_line).group(0))
                muts = re.findall(r'[A-Z]+->[A-Z]+', mutation_line)
                for mut in muts:
                    mut_df.loc[position, mut[-2]] += 1'''
    for mut_dict in os.listdir(output_path+'alpha/mut_dicts/'+str(threshold)):
        seq_count += 1
        with open(output_path+'alpha/mut_dicts/'+str(threshold)+'/'+mut_dict, 'r') as f: #LOOKING AT ONLY ALPHA RIGHT NOW :)
            lines = f.readlines()
        for mutation_line in lines:
            position = int(re.search(r'^\d+', mutation_line).group(0))
            muts = re.findall(r'[A-Z]+->[A-Z]+', mutation_line)
            for mut in muts:
                mut_df.loc[position, mut[-2]] += 1
    print('mut_df')
    print(mut_df)
    mut_df.to_csv('simulation_output/001.csv')

    #read in all reference mutations for threshold
    if os.path.exists('sim_ref_data/thresholded_reference_tensors/'+str(threshold)+'.pt'):
        ref_mat = torch.load('sim_ref_data/thresholded_reference_tensors/'+str(threshold)+'.pt')
    else:
        print('compiling reference mutations for '+ str(threshold))
        ref_mat = compile_reference_mutations(variant_order, threshold)
    #ref_mat formatted as [position_index, variant_index, mut] where default shape = [29904,13,3]

    print('number of reference mutations for %d variants: %d ' % (ref_mat.shape[1], torch.count_nonzero(ref_mat).item()))
    
    #find positions where we predict a mut and it matches at least one variant reference
    matching_indices = pd.DataFrame()
    #if show == 'matches':
    for position in mut_df.index.to_numpy():
        temp_df = pd.DataFrame(np.zeros([4,len(variant_order)]), index=pd.MultiIndex.from_product([[position], ['T','G','C','A']]), columns=variant_order)
        for mut in mut_df.columns.to_numpy():
            if mut_df.loc[position,mut] != 0:
                for var_index in range(ref_mat.shape[1]):
                    if nuc_codes[mut] in ref_mat[position,var_index,:]:
                        temp_df.loc[(position,mut),variant_order[var_index]] = mut_df.loc[position,mut]
        
        temp_df = temp_df.loc[~(temp_df==0).all(axis=1)]
        if not temp_df.empty:
            matching_indices = pd.concat([matching_indices,temp_df])
    print('matching_indices shape: ', matching_indices.shape)
    print(matching_indices)

    ref_indices = np.unique(torch.argwhere(ref_mat)[:,0].numpy())
    print(ref_indices, ref_indices.shape)


    fig, ax0 = plt.subplots(figsize=(25,5))

    #pull mutations across the genome in non-overlapping genes
    counts = {} #sim muts that match reference muts for each gene
    ref_genes = {} #ref mut indices for each gene
    #using subset_genes
    '''for gene, position in subset_genes.items():
        if len(position) == 2:
            matches = matching_indices.loc[position[0]:position[1]]
        else:
            matches = matching_indices.loc[position[0][0]:position[0][1]]
            for sub_position in position[1:]:
                matches = pd.concat([matches, matching_indices.loc[sub_position[0]:sub_position[1]]])
        counts[gene] = pd.Series(np.max(matches.to_numpy(), axis=1), index=matches.index.get_level_values(0), name=gene) / seq_count
    '''
    if show == 'matches':
        #not using subset_genes
        for gene, value in gene_info.items():
            position = value[1]
            matches = matching_indices.loc[position[0]:position[1]]
            counts[gene] = pd.Series(np.max(matches.to_numpy(), axis=1), index=matches.index.get_level_values(0), name=gene) / seq_count
            print(gene + ' refs: ', torch.count_nonzero(ref_mat[position[0]:position[1]]).item())
            ref_genes[gene] = np.intersect1d(ref_indices, np.arange(position[0],position[1]))
    
    elif show == 'all':
        for gene, value in gene_info.items():
            position = value[1]
            matches = mut_df.loc[position[0]:position[1]] #subset matches to current gene
            matches = matches.loc[~(matches==0).all(axis=1)] #remove 'matches' of 0.0 frequency that appear since we aren't comparing against ref list
            counts[gene] = pd.Series(np.max(matches.to_numpy(), axis=1), index=matches.index.get_level_values(0), name=gene) / seq_count
            print(gene + ' refs: ', torch.count_nonzero(ref_mat[position[0]:position[1]]).item())
            print(gene + ' matching muts: ', matching_indices.loc[position[0]:position[1]].shape[0])

    
    count = pd.Series()
    for gene, df in counts.items():
        if gene in ['S','E','M','N','ORF1ab','ORF3a','ORF6','ORF7a','ORF7b','ORF8','ORF10']:
            if gene in ['ORF7a','ORF7b']:
                df = df.loc[df.index.to_numpy()[(df.index.to_numpy() < subset_genes[gene][1]) & (df.index.to_numpy() > subset_genes[gene][0])]]
            #df.to_csv('simulation_output/'+gene+'_count_2.csv')
            count = pd.concat([count, df], axis=0)
    #count = count / seq_count
    print('genome count: ', count.shape, count)
    count.to_csv('simulation_output/002.csv')

    '''ref_indices = np.array([])
    for gene, arr in ref_only.items():
        ref_only_indices = np.append(ref_only_indices, arr)'''
    

    high_freq_indices = count.loc[count >= count.mean() + (3*count.std())].index
    low_freq_indices = count.loc[count <= count.mean() - (3*count.std())].index
    #low_freq_indices = np.setdiff1d(ref_indices, count.index.to_numpy()) #ref_only_indices[ref_only_indices != matching_indices.index.to_numpy()]
    print('genome freq avg: %f, std: %f, num high freq: %d, num low freq: %d' % (count.mean(), count.std(), high_freq_indices.shape[0], low_freq_indices.shape[0]))
    #print(high_freq_indices)
    #print(count.loc[high_freq_indices])
    print('genome high freq: ', high_freq_indices.shape, ' low freq: ', low_freq_indices.shape, 'all ref: ', ref_indices.shape)
    #print(np.sort(high_freq_indices.to_numpy()))
    #print(np.sort(low_freq_indices))
    #print(np.sort(ref_indices))

    empty_indices = [pos for pos in range(29904) if pos not in count.index.to_numpy()]
    empty_series = pd.Series(np.zeros([len(empty_indices)]),index=empty_indices)
    count = pd.concat([count, empty_series])
    count = count.sort_index()
    colors=['r' if index in high_freq_indices.to_numpy() else 'g' if index in low_freq_indices else 'b' for index in count.index.to_numpy()]
    ax0.bar(x=count.index.to_numpy(), width=1.5, height=count.to_numpy(), color=colors) #/seq_count
    #ax.invert_yaxis()
    ax0.set_yticks(ticks=np.arange(0,.3,.05), labels=['{:,.0%}'.format(val) for val in np.arange(0,.3,.05)])
    ax0.set_xticks(ticks=np.arange(0,30000,1000), labels=np.arange(0,30000,1000), minor=False)
    
    #label high freq indices along the top
    #ax0.tick_params(axis='x', which='minor', top=False, labeltop=True, bottom=False, labelbottom=False)
    #ax0.set_xticks(ticks=high_freq_indices, labels=high_freq_indices, minor=True, rotation=45)

    plt.savefig('simulation_output/sim_result_hist.png')
    plt.close()

    #find region of size n where mut frequencies are lowest
    window_size = 25384 - 21562
    genome_count = pd.Series(np.max(mut_df.to_numpy(), axis=1), index=mut_df.index.get_level_values(0), name=gene) / seq_count
    #print(genome_count, genome_count.shape)
    genome_count.to_csv('simulation_output/002.csv')
    genome_count = genome_count.to_numpy()
    avgs = [np.mean(genome_count[index:index+window_size]) for index in range(genome_count.shape[0]-window_size)]
    #print('avgs: ', len(avgs), avgs[:10])
    min_index = np.argmin(avgs)
    min_value = avgs[min_index]
    print('min region: ', min_index, ', ', min_value)
    print('avgs avg: ', np.mean(avgs), ', std: ', np.std(avgs))
    print('spike avg: ', avgs[21562])

    #grid

    fig = plt.figure(layout='constrained', dpi=100, figsize=(35,10))
    grid = gridspec.GridSpec(3,1, figure=fig, wspace=.15, hspace=.1, height_ratios=[1/3,1/3,1/3])
    row_1 = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=grid[1,0]) #, hspace=.2
    row_2 = gridspec.GridSpecFromSubplotSpec(1,7,subplot_spec=grid[2,0]) #, hspace=.2
    ax0 = fig.add_subplot(grid[0,0])
    ax1_0,ax1_1,ax1_2 = [fig.add_subplot(row_1[0,i]) for i in range(0,3)]
    ax2_0,ax2_1,ax2_2,ax2_3,ax2_4,ax2_5,ax2_6 = [fig.add_subplot(row_2[0,i]) for i in range(0,7)]
    axs = [ax0, ax1_0,ax1_1,ax1_2, ax2_0,ax2_1,ax2_2,ax2_3,ax2_4,ax2_5,ax2_6]

    gene_order = ['ORF1ab','S','E','M','N','ORF3a','ORF6','ORF7a','ORF7b','ORF8','ORF10']
    for gene_index, gene in enumerate(gene_order):
        ax = axs[gene_index]
    
        count = counts[gene]
        high_freq_indices = count.loc[count >= count.mean() + (3*count.std())].index
        low_freq_indices = count.loc[count <= count.mean() - (3*count.std())].index
        #low_freq_indices = np.setdiff1d(ref_genes[gene], count_gene.index.to_numpy())
        
        
        #print(high_freq_indices)
        #print(count_gene.loc[high_freq_indices])
        print(gene +' count: ', count.shape)
        print(gene+' freq avg: %f, std: %f, num high freq: %d, num low freq: %d' % (count.mean(), count.std(), high_freq_indices.shape[0], low_freq_indices.shape[0]))
        #count.to_csv('simulation_output/'+gene+'_count.csv')

        empty_indices = [pos for pos in range(gene_info[gene][1][0],gene_info[gene][1][1]) if pos not in count.index.to_numpy()]
        empty_series = pd.Series(np.zeros([len(empty_indices)]),index=empty_indices)
        count = pd.concat([count, empty_series])
        count = count.sort_index()
        #count = pd.Series(np.zeros([gene_info[gene][1][1] - gene_info[gene][1][0]]), index=np.arange(gene_info[gene][1][0], gene_info[gene][1][1])) + count
        colors=['r' if index in high_freq_indices.to_numpy() else 'g' if index in low_freq_indices else 'b' for index in count.index.to_numpy()]
        ax.bar(x=count.index.to_numpy(), width=1.5, height=count.to_numpy(), color=colors) # /seq_count
        #ax.invert_yaxis()
        ax.set_yticks(ticks=np.arange(0,0.25,.05), labels=['{:,.0%}'.format(val) for val in np.arange(0,.25,.05)])
        ax.set_xticks(ticks=np.linspace(count.index.to_numpy()[0],count.index.to_numpy()[-1],5,True,dtype=int), labels=np.linspace(count.index.to_numpy()[0],count.index.to_numpy()[-1],5,True,dtype=int), minor=False)
        ax.tick_params(axis='x', which='minor', top=True, labeltop=True, bottom=False, labelbottom=False, rotation=45)
        ax.set_xticks(ticks=high_freq_indices, labels=high_freq_indices, minor=True)
        ax.set_title(gene)


    plt.savefig('simulation_output/sim_result_hist_2.png')


    #spike

    fig, ax0 = plt.subplots(figsize=(16,4), dpi=250)
    #position = subset_genes['S']
    #matches_spike = matching_indices.loc[position[0]:position[1]]
    #count = pd.Series(np.max(matches_spike.to_numpy(), axis=1), index=matches_spike.index.get_level_values(0))
    #count = count / seq_count
    #count = count.drop_level(1)
    #print('count', count, count.shape, count.index)
    count = counts['S']
    #print(count)
    #print(count.loc[count>.5])
    
    #mut = pd.Series(np.argmax(mut_df.to_numpy(), axis=1), index=mut_df.index)
    high_freq_indices = count.loc[count >= (count.mean() + (3*count.std()))].index.get_level_values(0) #count.loc[count >= .9].index #
    low_freq_indices = count.loc[count <= (count.mean() - (3*count.std()))].index.get_level_values(0)
    #low_freq_indices = np.setdiff1d(ref_genes['S'], count.index.to_numpy())
    #count = pd.concat([count, pd.Series(np.zeros([position[1]-position[0]]), index=np.arange(position[0],position[1])).drop(count.index)]).reset_index()
    #count.columns = ['position','value']
    #print(count)
    #print(np.mean(count.to_numpy()))
    #print(high_freq_indices)
    count.to_csv('simulation_output/dist.csv')
    empty_indices = [pos for pos in range(gene_info['S'][1][0],gene_info['S'][1][1]) if pos not in count.index.to_numpy()]
    empty_series = pd.Series(np.zeros([len(empty_indices)]),index=empty_indices)
    count = pd.concat([count, empty_series])
    count = count.sort_index()
    #count = pd.Series(np.zeros([gene_info['S'][1][1] - gene_info['S'][1][0]]), index=np.arange(gene_info['S'][1][0],gene_info['S'][1][1])) + count
    colors=['r' if index in high_freq_indices.to_numpy() else 'g' if index in low_freq_indices else 'b' for index in count.index.to_numpy()]
    ax0.bar(x=count.index.get_level_values(0), height=count.to_numpy(), color=colors, width=2) #/seq_count
    #sns.barplot(data=count, x='position', y='value', ax=ax0)
    #ax.invert_yaxis()
    ax0.set_yticks(ticks=np.arange(0,0.25,.05), labels=['{:,.0%}'.format(val) for val in np.arange(0,.25,.05)])
    ax0.set_xticks(ticks=np.append(np.arange(subset_genes['S'][0],subset_genes['S'][1],500),subset_genes['S'][1]), labels=np.append(np.arange(subset_genes['S'][0],subset_genes['S'][1],500),subset_genes['S'][1]), minor=False)
    
    #label high freq indices along the top
    ax0.tick_params(axis='x', which='minor', top=False, labeltop=True, bottom=False, labelbottom=False)
    ax0.set_xticks(ticks=high_freq_indices, labels=high_freq_indices, minor=True, rotation=45)

    plt.savefig('simulation_output/sim_result_hist_3.png')
    plt.close()

def compile_reference_mutations(variant_order, threshold):
    #total_reference_df = pd.DataFrame(index = np.arange(0,29904))
    ref_mat = torch.tensor(np.zeros([29904,len(variant_order),3]))
    folders = [folder for folder in os.listdir('sim_ref_data') if '_full_clade' in folder]
    for var_index, var in enumerate(variant_order):
        folder = [folder for folder in folders if '('+var+')' in folder][0]
        var_ref_df = pd.read_csv('sim_ref_data/'+folder+'/reference_mutations/'+str(threshold)+'_reference_mutations.csv', index_col=1, header=0).sort_index()
        #var_ref_series = var_ref_df.loc[:,'mut']
        #var_ref_series.name = variant
        #total_reference_df = total_reference_df.join(var_ref_series)
        #print(total_reference_df)
        nuc_codes={'T':1,'G':2,'C':3,'A':4}
        for index in var_ref_df.index.to_numpy():
            position_entry = var_ref_df.loc[index,:]
            #if index == 31:
            #    print(position_entry, position_entry.shape)
            if len(position_entry.shape) > 1:
                for row in range(position_entry.shape[0]):
                    ref_mat[index,var_index,row] = nuc_codes[position_entry.iloc[row,-1]]
            else:
                ref_mat[index,var_index,0] = nuc_codes[position_entry.loc['mut']]
        #print(ref_mat[31,var_index,:])
    if not os.path.exists('sim_ref_data/thresholded_reference_tensors'):
        os.mkdir('sim_ref_data/thresholded_reference_tensors')
    torch.save(ref_mat, 'sim_ref_data/thresholded_reference_tensors/'+str(threshold)+'.pt')
    with open('sim_ref_data/thresholded_reference_tensors/info.txt', 'a') as f:
        f.write(str(threshold) + ': ' + ','.join(variant_order))
    return ref_mat

#plot total gwtc as heatmap
def plot_triplet_count(size=(12,12)):
    triplet_counts = pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/total.csv', index_col=0, header=0)
    if size == (12,12):
        fig, ax = plt.subplots(figsize=(8,6), dpi=250)
        sns.heatmap(np.repeat(triplet_counts.to_numpy(), 3, axis=1)[:-4,:], cmap='Greys', linewidths=.5, linecolor='Black', xticklabels=columns_figs, yticklabels=rows_figs[:-4])
        ax.set_title('Total Gene Triplet Counts', fontsize=20)
        plt.savefig('simulation_output/genome_triplet_counts.png')
        plt.close()
    elif size == (12,4):
        #fig, ax = plt.subplots(figsize=(4,6), dpi=250, layout='tight')
        #sns.heatmap(triplet_counts.to_numpy()[:12,:], cmap='Greys', linewidths=.5, linecolor='Black', annot=True, annot_kws={'fontsize':14}, fmt='.3g')
        fig, axs = plt.subplots(figsize=(3,4), layout='tight', dpi=200)
        sns.heatmap(triplet_counts, ax=axs, annot=False, cmap='Greys', linewidth=.5, linecolor='gray')
        axs.set_yticklabels([row.replace('--','') for row in rows_figs][:12], fontsize=10, rotation='horizontal')
        axs.set_xticklabels(columns_shortened_figs, fontsize=10)
        axs.set_title('Total Gene Triplet Counts', fontsize=14)
        plt.savefig('simulation_output/genome_triplet_counts_thin.png')
        plt.close()

#i have too many correlations to keep track of
def real_vaccine_correlation(mats, variant_orders):
    big_df = pd.read_excel('simulation_output/final_info/output_spreadsheet.xlsx', sheet_name='Vaccine_3_31_24', skiprows=578, nrows=38)
    df_indices = big_df.iloc[1:17,0]
    df_columns = big_df.iloc[0,1:5]
    
    spike_meaningful = big_df.iloc[1:17,1:5].astype(float)
    spike_meaningful.index = df_indices
    spike_meaningful.columns = df_columns
    print(spike_meaningful)
    fig, ax = plt.subplots(figsize=(4,10), layout='tight')
    sns.heatmap(spike_meaningful.to_numpy(), yticklabels=rows_figs, xticklabels=columns_shortened_figs, cmap='Greys', linewidth=.5, linecolor='Grey')
    ax.set_yticklabels(rows_figs, rotation='horizontal')
    ax.set_title('Non-Degenerate Contexts in Spike Protein')
    plt.savefig('simulation_output/spike_meaningful_contexts.png')
    plt.close()
    spike_meaningful = spike_meaningful.to_numpy()[:12,:]

    

    thresholds = ['1e-05+', '1e-05 : 5e-05', '5e-05+']
    num_variants = len(variant_orders[0])
    output_df = pd.DataFrame(np.zeros([2*3, num_variants+2]), index=pd.MultiIndex.from_product([['corr', 'p'],thresholds]), columns=variant_orders[0]+['g1','g2'])
    g1 = [0,1,2,3,4,5]
    g2 = [num_variants-1]
    
    #old and bad
    '''for mat_index, mat in enumerate(mats):
        print(mat.shape)
        for variant_index, variant in enumerate(variant_orders[mat_index]):
            
            #curr_mat = torch.stack([torch.sum(mat[variant_index][:,i:i+3], dim=(1)) for i in range(0,12,3)], axis=1)
            #print(spike_meaningful.flatten().shape, curr_mat.flatten().shape)
            #print(spike_meaningful, curr_mat)
            reconfiguered_mats[mat_index,variant_index] = curr_mat
            #print(curr_mat, curr_mat.flatten(), spike_meaningful.flatten())
            corr = stats.pearsonr(spike_meaningful.flatten(), curr_mat.flatten())
            #output_df.loc['corr',var_index] = corr[0]
            #output_df.loc['p',var_index] = corr[1]
            #print(mat_index, variant)
            #print(corr)
            #print(mat[variant_index])
            #print(curr_mat)
            output_df.loc[('corr',thresholds[mat_index]), variant] = corr[0]
            output_df.loc[('p',thresholds[mat_index]), variant] = corr[1]
        #print(variant_orders[mat_index])
        for i, group in enumerate([g1,g2]):
            mat_subset = reconfiguered_mats[mat_index,group,:,:]
            corr = stats.pearsonr(np.tile(spike_meaningful.flatten(), mat_subset.shape[0]), mat_subset.flatten())
            print(group, corr)
            print(np.tile(spike_meaningful.flatten(), mat_subset.shape[0]))
            print(mat_subset.flatten())
            output_df.loc[('corr',thresholds[mat_index]), ['g1','g2'][i]] = corr[0]
            output_df.loc[('p',thresholds[mat_index]), ['g1','g2'][i]] = corr[1]
    output_df.to_csv('simulation_output/final_vaccine_corr.csv')'''


    '''
    TTT for spike meaningful is # of contexts that can change resulting protein
    TTT for mut 'rates' has T[T->Y]T which has n base contexts with 3 potential results
        so if n = 10 and we have 5 muts for T[T->G]T, 1 for T[T->C]T, and 7 T[T->A]T for alpha
        T[T]T (alpha) = (5+1+7) / (10+10+10) = 13/30
        if beta is 4,3,2:
        T[T]T (alpha,beta) = (5+1+7+4+3+2) / (10+10+10+10+10+10); denom = 3*triplet_count*num_variants
    '''
    #reconfigure mut_rates to be 12x4 and aggregate across groups
    reconfigured_mats = torch.tensor(np.zeros([3,num_variants+2,12,4])) #(agg,g1,g2),(var,g1,g2),(row),(col)
    triplet_counts = pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/total.csv', index_col=0, header=0)
    for mat_index, mat in enumerate(mats):
        print(thresholds[mat_index])
        for var_index in range(mat.shape[0]):
            if var_index == 0:
                print('alpha: ', mat[var_index])
            for col_index in range(0,mat[var_index].shape[1],3): #0,3,6,9
                temp_cols = np.array([])
                for col in mat[var_index,:,col_index:col_index+3].transpose(0,1):
                    #print(col, triplet_counts.iloc[:,int(col_index/3)])
                    temp_cols = np.append(temp_cols, col * triplet_counts.iloc[:-4,int(col_index/3)]) #un-normalize by triplet count (5/10 -> 5)
                temp_cols = temp_cols.reshape(3,12).T
                if var_index == 0:
                    print('temp cols: ', temp_cols[0])
                reconfigured_mats[mat_index,var_index,:,int(np.floor(col_index/3))] = torch.tensor(np.sum(temp_cols, axis=1)) #append sum of same mutation columns (T[T->G,C,A]T -> TTT)
            if var_index == 0:
                print('reconfigured_mat: ', reconfigured_mats[mat_index,var_index])
        '''for group_index, group in enumerate([g1,g2]):
            if group_index == 0:
                print("g1 NTN's: ", reconfigured_mats[mat_index,group,0,0])
            reconfigured_mats[0-mat_index,num_variants+group_index] = torch.tensor(np.sum(reconfigured_mats[mat_index,group].numpy(), axis=0))
            if group_index == 0:
                print("g1 final: ", reconfigured_mats[mat_index,num_variants+group_index,0,0])

            #then normalize
            reconfigured_mats[0-mat_index,num_variants+group_index] /= len(group) * 3 * triplet_counts.to_numpy()[:-4]
            if group_index == 0:
                print("g1 final normalized: ", reconfigured_mats[mat_index,num_variants+group_index,0,0])

            #generate heatmap of g1 and g2
            fig, ax = plt.subplots(figsize=(4,7), layout='tight')
            sns.heatmap(reconfigured_mats[mat_index,num_variants+group_index], yticklabels=rows_figs[:-4], xticklabels=columns_shortened_figs, cmap='Greys', linewidth=.5, linecolor='Grey')
            ax.set_yticklabels(rows_figs[:-4], rotation='horizontal')
            group_label = ['G1','G2'][group_index]
            threshold_label = thresholds[mat_index]
            threshold_file = thresholds[mat_index]
            if mat_index == 1:
                threshold_label = '1e-05 : 5e-05'
                threshold_file = 'low_freqs'
            ax.set_title('Aggregate Mutation Matrix for '+group_label+' at \n'+threshold_label)
            plt.savefig('simulation_output/'+['g1','g2'][group_index]+'_'+threshold_file+'.png')
            plt.close()'''
        #for var_index in range(mat.shape[0]):
        #    reconfigured_mats[mat_index,var_index] /= 3 * triplet_counts.to_numpy()[:-4]
            #if var_index == 6:
            #    print('pirola final: ', reconfigured_mats[mat_index,var_index])

        
        #loop through each variant (including g1 and g2) and correlate to spike_meaningful
        for var_index in range(reconfigured_mats.shape[1]):
            if var_index == 0:
                print('--------')
                print(spike_meaningful, reconfigured_mats[mat_index, var_index])
            corr = stats.pearsonr(spike_meaningful.flatten(), reconfigured_mats[mat_index,var_index].flatten())
            output_df.loc[('corr',thresholds[mat_index]), (variant_orders[mat_index]+['g1','g2'])[var_index]] = corr[0]
            output_df.loc[('p',thresholds[mat_index]), (variant_orders[mat_index]+['g1','g2'])[var_index]] = corr[1]
    #if not max_threshold:
    output_df.to_csv('simulation_output/final_vaccine_corr_2.csv')
    #else:
    #    output_df.to_csv('simulation_output/final_vaccine_corr_'+max_threshold+'.csv')                

#correlate randomly generated variant matrices based on n mutations from a threshold normalized by gwtc
def randomize_correlation_matrices(variant_order):
    #normal case is 1e-5:5e-05
    mut_counts = {'alpha':1963, 'beta':428, 'delta':2403, 'epsilon':617, 'gamma':1741, 'iota':285, 'omicron':2811, 'pirola':2360, 'kraken':2091, 'aggregate':519}

    triplet_counts = pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/total.csv', index_col=0, header=0) #read in triplet counts for all encoding genes
    #probability of each cell in matrix is split to work with np method, so options is [0:144) n number of times and probs is 1 n number of times
    options = np.array([])
    probs = np.array([])
    triplet_counts_flattened = np.repeat(triplet_counts.to_numpy(), 3, axis=1).flatten()
    print(triplet_counts_flattened)
    #loop through gwtc count each triplet, then normalize probability based on n
    for index, count in enumerate(triplet_counts_flattened):
        for iter in range(int(count)):
            options = np.append(options, index)
            probs = np.append(probs, 1)
    probs = probs / np.sum(probs)

    print(options)
    print(probs)
    print(np.sum(probs))

    total_output_df = pd.DataFrame()
    #generate random dists 100 times
    for iter in range(100):
        var_dict = {}
        for variant, mut_count in mut_counts.items():
            if variant in variant_order:
                dist =  np.random.choice(options, size=mut_count, replace=False, p=probs) #gen random dist of mut_counts[variant] number of muts
                #print(dist)
                var_mat = np.zeros([144])
                for val in dist:
                    var_mat[int(val)] += 1
                var_mat = pd.DataFrame(var_mat.reshape([12,12]), index=triplet_counts.index.to_numpy()[:-4], columns=columns) #convert dist into 12x12 matrix
                #var_mat = var_mat / mut_count
                var_mat = var_mat / np.repeat(triplet_counts.to_numpy()[:12,:],3,axis=1) #normalize by triplet counts to match our 'mut_rates'
                #print(np.repeat(triplet_counts.to_numpy()[:12,:],3,axis=1))
                #print('>')
                #print(var_mat)
                var_dict[variant] = var_mat

        output_df = pd.DataFrame(np.zeros([2*len(variant_order),len(variant_order)]), index=pd.MultiIndex.from_product([['corr','p'], variant_order]), columns=variant_order)
        #loop through each combination of variants
        for var_1, mat_1 in var_dict.items():
            for var_2, mat_2 in var_dict.items():
                #correlate random mats
                if var_1 != var_2:
                    corr = stats.pearsonr(mat_1.to_numpy().flatten(), mat_2.to_numpy().flatten())
                    output_df.loc[('corr', var_1), var_2] = corr[0]
                    output_df.loc[('p', var_1), var_2] = corr[1]
                else: 
                    output_df.loc[('corr', var_1), var_2] = np.nan
                    output_df.loc[('p', var_1), var_2] = np.nan
        #print(output_df)
        total_output_df = pd.concat([total_output_df, output_df])
        #total_output_df.loc[:,(iter,list(mut_counts.keys()))] = output_df
    #
    #read in correlation of observed mutations for variants
    expected_means = pd.read_csv('simulation_output/000.csv', index_col=0, header=0)

    print(total_output_df.loc[('corr','alpha'),:])
    #loop through each variant and t-test between randomized dist corr and observed corr
    t_test_output_df = pd.DataFrame(np.zeros([2*len(variant_order),len(variant_order)]), index=pd.MultiIndex.from_product([['t','p'], variant_order]), columns=variant_order)
    for var_1 in var_dict.keys():
        for var_2 in var_dict.keys():
            #t_result = stats.ttest_ind(total_output_df.loc[('corr',var_1),:].to_numpy().flatten(), total_output_df.loc[('corr',var_2),:].to_numpy().flatten(), equal_var=False, nan_policy='omit')
            t_result = stats.ttest_1samp(total_output_df.loc[('corr',var_1),var_2].to_numpy().flatten(), expected_means.loc[var_1, var_2], alternative='less')
            print(t_result.statistic, t_result.pvalue)
            t_test_output_df.loc[('t',var_1),var_2] = t_result.statistic
            t_test_output_df.loc[('p',var_1),var_2] = t_result.pvalue
    total_output_df = pd.concat([total_output_df, t_test_output_df])
    '''for var_0 in var_dict.keys():
        t_test_output_df = pd.DataFrame(np.zeros([2*len(mut_counts.keys()),len(mut_counts.keys())]), index=pd.MultiIndex.from_product([['t','p'], mut_counts.keys()]), columns=mut_counts.keys())
        for var_1 in var_dict.keys():
            for var_2 in var_dict.keys():
                t_result = stats.ttest_ind(total_output_df.loc[('corr',var_0),var_1].to_numpy().flatten(), total_output_df.loc[('corr',var_0),var_2].to_numpy().flatten(), equal_var=False, nan_policy='omit')
                print(t_result.statistic, t_result.pvalue)
                t_test_output_df.loc[('t',var_1),var_2] = t_result.statistic
                t_test_output_df.loc[('p',var_1),var_2] = t_result.pvalue
        total_output_df = pd.concat([total_output_df, t_test_output_df])'''
    total_output_df.to_csv('simulation_output/randomized_low_freq_corr.csv')
    
#generate random matrices and analyze columns vs observed variant matrices
def randomize_correlation_matrices_columns(variant_order):

    triplet_counts = pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/total.csv', index_col=0, header=0).iloc[:12,:] #read in triplet counts for all encoding genes
    #gen probability dist from triplet_counts
    options = {}
    probs = {}
    for col in triplet_counts.columns.to_numpy():
        options[col] = np.concatenate([np.repeat(index, triplet_counts.loc[:,col].iloc[index]) for index in range(triplet_counts.shape[0])]) #convert triplet count column to m indices n times
        probs[col] = [1] * len(options[col]) / np.sum([1] * len(options[col]))
        #print(len(options[col]), len(probs[col]))
        print(np.unique(options[col], return_counts=True), np.unique(probs[col], return_counts=True))

    avg_mats, std_mats = {},{}
    output_df = pd.DataFrame(np.zeros([2*len(variant_order),12]), index=pd.MultiIndex.from_product([['corr','p'], variant_order]), columns=columns)
    output_df_2 = pd.DataFrame()
    
    #loop through each variant
    threshold='low_freq'
    for variant in variant_order:
        #read in observed mut matrix
        var_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder and '_full_clade' in folder][0]
        obs_mat = pd.read_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/'+threshold+'_mat.csv', index_col=0, header=0)
        mut_counts = {}
        for col in obs_mat.columns.to_numpy():
            col_counts = obs_mat.loc[:,col] * triplet_counts.loc[:,col[0]] #convert obs rate to mut counts
            mut_counts[col] = round(col_counts.sum()) #total mut count for column
        
        var_mat = np.zeros([100,12,12])
        #perform 100 iterations
        for iter in range(100):

            for col_index, col in enumerate(obs_mat.columns.to_numpy()):
                dist = np.random.choice(options[col[0]], size=mut_counts[col], replace=False, p=probs[col[0]]) #gen random dist of mut_counts[variant] number of muts
                col_mat = np.zeros([12])
                for val in dist:
                    col_mat[int(val)] += 1
                col_mat = col_mat / triplet_counts.loc[:,col[0]] #normalize by triplet counts to match our 'mut_rates'
                var_mat[iter,:,col_index] = col_mat
        
        avg_mats[variant] = np.mean(var_mat, 0)
        std_mats[variant] = np.std(var_mat, 0)

        output_df_2 = pd.concat([output_df_2, pd.concat([obs_mat, pd.DataFrame(avg_mats[variant], index=obs_mat.index, columns=obs_mat.columns), pd.DataFrame(std_mats[variant], index=obs_mat.index, columns=obs_mat.columns)], axis=1)])

        for col_index in range(avg_mats[variant].shape[1]):
            col_mean = np.mean(avg_mats[variant][:,col_index])
            col_std = np.std(avg_mats[variant][:,col_index])
            corr = stats.pearsonr(avg_mats[variant][:,col_index], obs_mat.iloc[:,col_index].to_numpy())
            output_df.loc[('corr', variant), obs_mat.columns.to_numpy()[col_index]] = corr[0]
            output_df.loc[('p', variant), obs_mat.columns.to_numpy()[col_index]] = corr[1]
    output_df.to_csv('simulation_output/randomized_low_freq_col_corr.csv')
    output_df_2.index = pd.MultiIndex.from_product([variant_order, obs_mat.index.to_numpy()])
    output_df_2.columns = pd.MultiIndex.from_product([['observed', 'rand_mean', 'rand_std'], obs_mat.columns.to_numpy()])
    output_df_2.to_csv('simulation_output/randomized_low_freq_col_mats.csv')

#perform t-test between spike match distributions
def spike_hist_t_tests():
    agg = pd.read_csv('simulation_output/spike_hists/all_variants/1e-20/agg/dist.csv', index_col=0, header=0)
    g1 = pd.read_csv('simulation_output/spike_hists/all_variants/1e-20/g1_low/dist.csv', index_col=0, header=0)
    g2 = pd.read_csv('simulation_output/spike_hists/all_variants/1e-20/g2_low/dist.csv', index_col=0, header=0)

    sim_names=['agg','g1','g2']
    output_df = pd.DataFrame(np.zeros([2*3,3]), index=pd.MultiIndex.from_product([['t','p'],sim_names]), columns=sim_names)
    for index_1, dist_1 in enumerate([agg, g1, g2]):
        for index_2, dist_2 in enumerate([agg, g1, g2]):
            t_result = stats.ttest_ind(dist_1, dist_2)
            output_df.loc[('t',sim_names[index_1]),sim_names[index_2]] = t_result.statistic
            output_df.loc[('p',sim_names[index_1]),sim_names[index_2]] = t_result.pvalue
    output_df.to_csv('simulation_output/spike_hists/all_variants/1e-20/dist_comparisons.csv')

def corr_this():

    jean_folders = [folder for folder in os.listdir('sim_ref_data') if 'jean' in folder]
    our_folders = ['B.1.1.7(alpha)_full_clade','B.1.351(beta)_full_clade','B.1.617.2(delta)_full_clade','P.1(gamma)_full_clade','B.1.1.529(omicron)_full_clade']
    print(jean_folders, our_folders)
    output_df = pd.DataFrame(np.zeros([7,10]), index=jean_folders, columns=pd.MultiIndex.from_product([['corr','p'],our_folders]))

    for variant_folder in jean_folders:
        for variant_folder_2 in our_folders:
            df_1 = pd.read_csv('sim_ref_data/'+variant_folder+'/thresholded_mutations/1e-05_mat.csv', index_col=0, header=0)
            df_2 = pd.read_csv('sim_ref_data/'+variant_folder_2+'/thresholded_mutations/5e-05_mat.csv', index_col=0, header=0)

            corr = stats.pearsonr(df_1.to_numpy().flatten(), df_2.to_numpy().flatten())
            output_df.loc[variant_folder,('corr',variant_folder_2)] = corr[0]
            output_df.loc[variant_folder,('p',variant_folder_2)] = corr[1]
    output_df.to_csv('simulation_output/jean_figs/dataset_correlations.csv')

#looking into jean rates
def jean_testing():
    #assuming fig 3 is based on total
    '''total_df = pd.read_csv('simulation_output/jean_figs/tensors/jean_total_1e-05.csv', index_col=0, header=0)
    for index in total_df.index:
        for column in total_df.columns:
            print(index, column, np.sum(total_df.loc[index,column]))'''
    total_tensor = torch.read('simulation_output/jean_figs/tensors/jean_total_1e-05.pt')
    for index in total_tensor.shape[0]:
        for column in total_tensor.shape[1]:
            print(rows[index], columns[column], torch.sum(total_tensor[index,column]))

#looking at overlap in muts between variants for thresholds
def threshold_testing(variant_order, thresholds):

    output_df = pd.DataFrame(np.zeros([len(variant_order)*2,len(thresholds)*len(variant_order)]),index=pd.MultiIndex.from_product([variant_order, ['matches', 'total']]), columns=pd.MultiIndex.from_product([thresholds, variant_order]))

    for threshold in thresholds:
        for variant in variant_order:
            var_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder and 'full_clade' in folder][0]
            mut_df = pd.read_csv('sim_ref_data/'+var_folder+'/reference_mutations/'+str(threshold)+'_reference_mutations.csv')
            for variant_2 in variant_order:
                if variant_2 != variant:
                    var_folder_2 = [folder for folder in os.listdir('sim_ref_data') if '('+variant_2+')' in folder and 'full_clade' in folder][0]
                    mut_df_2 = pd.read_csv('sim_ref_data/'+var_folder_2+'/reference_mutations/'+str(threshold)+'_reference_mutations.csv')

                    overlapping_df = pd.merge(mut_df, mut_df_2, how='inner', on=['position','old','mut'])
                    total_df = pd.merge(mut_df, mut_df_2, how='outer', on=['position','old','mut'])
                    print(f'{variant} vs {variant_2} has {overlapping_df.shape[0]} matches out of {total_df.shape[0]} total muts')
                    #print(overlapping_df)
                    output_df.loc[(variant, 'matches'), (threshold, variant_2)] = overlapping_df.shape[0]
                    output_df.loc[(variant, 'total'), (threshold, variant_2)] = total_df.shape[0]
    output_df.to_csv('simulation_output/threshold_testing.csv')

#correlate 5e-05 population rates with 5e-05 unweighted sample rates
def correlate_jf_with_our_data(weighted=False):
    pop_folders = [folder for folder in os.listdir('sim_ref_data') if '_full_clade' in folder and 'jean' not in folder]
    pop_vars = [re.search(r'\(\w+\)', folder)[0][1:-1] for folder in pop_folders]
    vivo_folders = [folder for folder in os.listdir('sim_ref_data') if '_full_clade' in folder and 'jean' in folder]
    vivo_vars = [re.search(r'\(\w+\)', folder)[0][1:-1] for folder in vivo_folders]
    #want full:both, full:lethals, full:non-lethals, tstv:tstv, naive:naive
    output_df = pd.DataFrame(np.zeros([len(vivo_vars)*2*4*2, len(pop_folders)*3]), index=pd.MultiIndex.from_product([['normal','log'],['full','reduced','tstv','naive'],['corr','p'],vivo_vars],names=['base','size','stat','jf_variants']), columns=pd.MultiIndex.from_product([['both','lethals','non-lethals'],pop_vars],names=['jf_mut_type','pop_variants']))
    total_mat_df = pd.DataFrame()
    variants_used = []
    print(pop_vars, vivo_vars)
    var_matches = {'alpha':'jean_alpha', 'beta':'jean_beta', 'delta':'jean_delta', 'gamma':'jean_gamma', 'omicron':'jean_omicron', 'aggregate':'jean_total'}
    #var_matches = {'alpha':'jean_total', 'beta':'jean_total', 'delta':'jean_total', 'gamma':'jean_total', 'omicron':'jean_total', 'aggregate':'jean_total'}
    #do both/lethal/non-lethal
    for var_folder_1, var_1 in zip(pop_folders, pop_vars): #for each population variant
        mat_1 = pd.read_csv('sim_ref_data/'+var_folder_1+'/thresholded_mutations/5e-05_mat.csv', index_col=0, header=0)
        for var_folder_2, var_2 in zip(vivo_folders, vivo_vars): #for each in vivo variant
            for mut_type in ['both','lethals','non-lethals']: #loop through dataset mutation types
                #read in jf dataframe
                mat_2 = pd.read_csv('sim_ref_data/'+var_folder_2+'/thresholded_mutations/'+mut_type+'_mut_rate_mat.csv', index_col=0, header=0)
                mat_2.fillna(value=0.0, inplace=True) #fill na values that may exist
                mat_2 = mat_2.iloc[:12,:12] #drop A[X>Y]N contexts that we can't generate
                #print(mat_1, mat_2)
                
                #correlate full context df
                corr = stats.pearsonr(mat_1.to_numpy().flatten(), mat_2.to_numpy().flatten())
                output_df.loc[('normal','full','corr',var_2), (mut_type,var_1)] = corr[0]
                output_df.loc[('normal','full','p',var_2), (mut_type,var_1)] = corr[1]

                #correlate context df without c->T column
                corr = stats.pearsonr(mat_1.drop(['C>T'],axis=1).to_numpy().flatten(), mat_2.drop(['C>T'],axis=1).to_numpy().flatten())
                output_df.loc[('normal','reduced','corr',var_2), (mut_type,var_1)] = corr[0]
                output_df.loc[('normal','reduced','p',var_2), (mut_type,var_1)] = corr[1]

                #print(mat_1, mat_2)
                #correlate log full context df
                corr = stats.pearsonr(np.log10(mat_1.to_numpy().flatten()+float('1e-16')), np.log10(mat_2.to_numpy().flatten()+float('1e-16')))
                output_df.loc[('log','full','corr',var_2), (mut_type,var_1)] = corr[0]
                output_df.loc[('log','full','p',var_2), (mut_type,var_1)] = corr[1]

                #correlate log context df without c->T column
                corr = stats.pearsonr(np.log10(mat_1.drop(['C>T'],axis=1).to_numpy().flatten()+float('1e-16')), np.log10(mat_2.drop(['C>T'],axis=1).to_numpy().flatten()+float('1e-16')))
                output_df.loc[('log','reduced','corr',var_2), (mut_type,var_1)] = corr[0]
                output_df.loc[('log','reduced','p',var_2), (mut_type,var_1)] = corr[1]    

                if var_1 in var_matches.keys():
                    if var_2 == var_matches[var_1]:
                        total_mat_df = pd.concat([total_mat_df, pd.concat([mat_1, mat_2], axis=1)], axis=0)
                        variants_used.append(var_1)
    total_mat_df.index = pd.MultiIndex.from_product([variants_used, rows])
    total_mat_df.columns = pd.MultiIndex.from_product([['population','vivo'],columns])

    #do tstv and naive
    pop_global, pop_tstv, pop_blind = read_thresholded_global_mat('5e-05', pop_vars) #pull mutation matrices
    vivo_global, vivo_tstv, vivo_blind = read_thresholded_global_mat('5e-05', vivo_vars, jf_flag=1) #pull jf matrices

    for context_type in ['tstv','naive']:
        for var_1 in pop_vars:
            for var_2 in vivo_vars:
                if context_type == 'tstv':
                    mat_1 = pop_tstv[pop_vars.index(var_1)].numpy().flatten()
                    mat_2 = vivo_tstv[vivo_vars.index(var_2)].numpy().flatten()
                else:
                    mat_1 = pop_blind[pop_vars.index(var_1)].numpy().flatten()
                    mat_2 = vivo_blind[vivo_vars.index(var_2)].numpy().flatten()
                #correlate full context df
                corr = stats.pearsonr(mat_1, mat_2)
                output_df.loc[('normal',context_type,'corr',var_2), ('both',var_1)] = corr[0]
                output_df.loc[('normal',context_type,'p',var_2), ('both',var_1)] = corr[1]

                #print(mat_1, mat_2)
                #correlate log full context df
                corr = stats.pearsonr(np.log10(mat_1+float('1e-16')), np.log10(mat_2+float('1e-16')))
                output_df.loc[('log',context_type,'corr',var_2), ('both',var_1)] = corr[0]
                output_df.loc[('log',context_type,'p',var_2), ('both',var_1)] = corr[1]

                
                #if var_1 in var_matches.keys():
                #    if var_2 == var_matches[var_1]:
                #        total_mat_df = pd.concat([total_mat_df, pd.concat([mat_1, mat_2], axis=1)], axis=0)
                #        variants_used.append(var_1)

    for mat_type in ['full','tstv','naive','reduced']:
        fig, axs = plt.subplots(figsize=(12,12))
        jf_vars = ['jean_alpha','jean_beta','jean_delta','jean_gamma','jean_omicron','jean_usa','jean_total']
        pop_vars = ['alpha','beta','delta','gamma','omicron','all','jf_vars']
        corr_df = output_df.loc[('normal',mat_type,'corr',jf_vars),('both',pop_vars)]
        p_value_df = output_df.loc[('normal',mat_type,'p',jf_vars),('both',pop_vars)]
        annotations = corr_df.round(2) #round df for formatting
        sns.heatmap(corr_df.to_numpy(), cmap='Greys', ax=axs, annot=annotations, fmt='', annot_kws={'fontsize':22}, vmin=0, vmax=1, linewidth=.5, linecolor='grey')
        axs.set_yticklabels([var.capitalize() for var in jf_vars], rotation=45, fontsize=22, wrap=True)
        axs.set_xticklabels([var.capitalize() for var in pop_vars], rotation=45, fontsize=22)
        axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        axs.set_title('Dataset Correlations', fontsize=28)
        plt.tight_layout()

        plt.savefig('simulation_output/jean_figs/'+mat_type+'_correlation.png')
        plt.close()

        fig, axs = plt.subplots(figsize=(12,12))
        sns.heatmap(p_value_df.to_numpy(), cmap='Greys', ax=axs, annot=True, annot_kws={'fontsize':22}, linewidth=.5, linecolor='grey')
        axs.set_yticklabels([var.capitalize() for var in jf_vars], rotation=45, fontsize=22, wrap=True)
        axs.set_xticklabels([var.capitalize() for var in pop_vars], rotation=45, fontsize=22)
        axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        axs.set_title('Dataset Correlations P-values', fontsize=28)
        plt.tight_layout()

        plt.savefig('simulation_output/jean_figs/'+mat_type+'_p_values.png')
        plt.close()
        print(p_value_df)
    
    if weighted:
        output_df.to_csv('simulation_output/jean_figs/weighted_vs_5e-05_pop.csv')
        total_mat_df.to_csv('simulation_output/jean_figs/weighted_mats.csv')
    else:
        output_df.to_csv('simulation_output/jean_figs/unweighted_vs_5e-05_pop.csv')
        total_mat_df.to_csv('simulation_output/jean_figs/unweighted_mats.csv')

    

#count number of mutations in each gene >0.5 for each variant
#changed to be >= threshold
def count_gene_muts_thresholded(threshold):
    #grab variant folder paths
    var_folders = [folder for folder in os.listdir('sim_ref_data') if 'full_clade' in folder and 'jean' not in folder]
    #grab variants
    variants = [re.search(r'\(\w+\)', folder)[0][1:-1] for folder in var_folders]
    #combine gene dicts for all genes and nsps and rdrp sub-domains
    gene_info_dict = subset_genes | nsp_positions | rdrp_sub_domains | spike_sub_domains
    
    #create output
    output_df = pd.DataFrame(np.zeros([len(var_folders), len(gene_info_dict)]), index=variants, columns=[key for key in gene_info_dict.keys()])
    
    #loop through variants
    for variant_folder, variant in zip(var_folders, variants):
        #read in reference mutations for threshold
        df = pd.read_csv('sim_ref_data/'+variant_folder+'/reference_mutations/'+threshold+'_reference_mutations.csv', header=0)
        print(variant)
        #loop through mutations
        #for mut_index in df.index:
        #    for gene, positions in gene_info_dict.items():
        #        positions = positions[1]
        #        if df.loc[mut_index, 'position'] >= positions[0] and df.loc[mut_index, 'position'] <= positions[1]:
        #            output_df.loc[variant, gene] += 1
        for gene, positions in gene_info_dict.items():
            print(gene, positions)
            if len(positions) == 2:
                output_df.loc[variant,gene] = df.loc[df['position'].isin(range(positions[0],positions[1]))].shape[0]
            else:
                for sub_positions in positions:
                    output_df.loc[variant,gene] += df.loc[df['position'].isin(range(sub_positions[0],sub_positions[1]))].shape[0]
    output_df = output_df.loc[('all','persistent','transient','alpha','delta','kraken','omicron','pirola','beta','epsilon','eta','gamma','hongkong','iota','kappa','lambda','mu'),('S','E','M','N','ORF1a','ORF3a','ORF6','ORF7a','ORF7b','ORF8','ORF10','nsp1','nsp2','nsp3','nsp4','nsp5','nsp6','nsp7','nsp8','nsp9','nsp10','nsp11','rdrp','nsp13','nsp14','nsp15','nsp16','NTD','RBD','SD1_2','NiRAN','interface','hand')]
    output_df.to_csv('simulation_output/high_freq_mut_dist_'+str(threshold)+'.csv')
    
#looping through each of the simulations run (that I renamed) and refactoring the output
def post_sims_analysis(analysis_variants, thresholds):
    # want
    # muts correct / muts placed
    # % of correct at each threshold (non-inclusive at lower frequencies)
    # easy graph for drop-off of muts returned by # placed
    if not os.path.exists('simulation_output/final_info/final_sim_analysis'):
        os.mkdir('simulation_output/final_info/final_sim_analysis')

    #averaging across variants, thresholds
    #so we have mut_count, context_type, positional/contextual, sim_type
    #graph_df.loc[(mut_type,num_muts),context_type]
    #graph_df = pd.DataFrame(np.zeros([2*11, 3]), index=pd.MultiIndex.from_product([['positional_matches','contextual_matches'],np.arange(1000,12000,1000)]), columns=['blind_contexts','naive_contexts','full_contexts'])
    output_df = pd.DataFrame()
    mut_counts = np.array([])
    datasets = np.array([])
    matches = np.array([])
    
    #read in number of possible positional and contextual matches for each variant at each threshold
    possible_matches = pd.DataFrame(np.zeros([2*len(analysis_variants), len(thresholds)]), index=pd.MultiIndex.from_product([analysis_variants,['positional','contextual']]), columns=thresholds)
    for variant in analysis_variants:
        for threshold in thresholds:
            variant_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder and 'full_clade' in folder][0]
            ref_muts = pd.read_csv('sim_ref_data/'+variant_folder+'/reference_mutations/'+str(threshold)+'_reference_mutations.csv', index_col=0, header=0)
            possible_matches.loc[(variant,'contextual'),threshold] = ref_muts.shape[0]
            possible_matches.loc[(variant,'positional'),threshold] = len(np.unique(ref_muts.loc[:,'position']))
    print(possible_matches.index, possible_matches.columns)
    print(possible_matches)
    possible_matches.to_csv('simulation_output/final_info/final_sim_analysis/possible_matches.csv')

    #loop through analysis folders
    for analysis_folder in [folder for folder in os.listdir('simulation_output/global/') if 'analysis' in folder]:
        num_muts = int(re.search(r'\d+', analysis_folder).group(0))
        if 'unweighted' in analysis_folder:
            dataset_type = 'unweighted'
        elif 'weighted' in analysis_folder:
            dataset_type = 'weighted'
        else:
            dataset_type = 'population'
        
        #read in thresholded_mat
        analysis_mat = pd.read_csv('simulation_output/global/'+analysis_folder+'/thresholded_mat.csv', index_col=[0,1], header=[0,1])

        print(analysis_mat)

        #adding 'all' to output df
        output_df = pd.concat([output_df, analysis_mat.loc[('all')]], axis=0)
        mut_counts= np.append(mut_counts, [num_muts,num_muts])
        datasets = np.append(datasets, [dataset_type,dataset_type])
        matches = np.append(matches, ['positional_matches','contextual_matches'])

        #creating heatmap of matching muts for 'all'
        '''for match_type in ['positional','contextual']:
            match_tensor = torch.load('simulation_output/global/'+analysis_folder+'/'+match_type+'_tensor.pt')
            #print(positional_tensor.shape)
            #[5,3,100,17,16,12]
            #[thresholds,contexts,num_sims,num_variants,row,col]
            #variants follow variant_order
            tensor_shape = match_tensor.shape
            fig, axs = plt.subplots(figsize=(30,20), nrows=tensor_shape[1], ncols=tensor_shape[0])
            
            for threshold_index in range(tensor_shape[0]):
                for context_index in range(tensor_shape[1]):
                    #print(analysis_variants)
                    #print(torch.mean(match_tensor[threshold_index,context_index,:,analysis_variants.index('all')], dim=(0)).shape)
                    match_df = pd.DataFrame(torch.mean(match_tensor[threshold_index,context_index,:,analysis_variants.index('all')], dim=(0)), index=rows_figs, columns=columns_figs)
                    sns.heatmap(match_df, ax=axs[context_index,threshold_index], cmap='Greys', linecolor='gray', linewidth=.5, annot=True, fmt='.2G')
                    axs[context_index,threshold_index].set_title(str(thresholds[threshold_index])+'_'+['naive','tstv','full'][context_index])
                    axs[context_index,threshold_index].set_xticklabels(columns_figs, rotation='horizontal')
                    #axs[context_index,threshold_index].set_yticklabels(rows_figs)
            plt.savefig('simulation_output/global/'+analysis_folder+'/'+match_type+'_matches.png')
            plt.close()'''


    output_df.index = pd.MultiIndex.from_tuples(zip(mut_counts.flatten(),datasets.flatten(),matches.flatten()))
    
    #output_df = output_df.loc[]
    output_df.sort_index(level=[0,1], inplace=True)
    #output_df.drop(labels=['unweighted'], axis=0, level=1, inplace=True)
    print(output_df)
    output_df.to_csv('simulation_output/final_info/final_sim_analysis/all_df.csv')

    '''#normalizing positional and contextual matches by the number of mutations placed or the maximum mutations possible at a threshold if it is lower
        possible_matches_mat = np.reshape(np.repeat(possible_matches.to_numpy(),3), [-1, len(thresholds*3)])
        possible_matches_mat = np.where(possible_matches_mat > num_muts, num_muts, possible_matches_mat)
        normalized_mat = analysis_mat.to_numpy() / possible_matches_mat
        normalized_mat = pd.DataFrame(normalized_mat, index=analysis_mat.index, columns=analysis_mat.columns)
        #print(normalized_mat)
        normalized_mat.to_csv('simulation_output/global/'+analysis_folder+'/thresholded_mat_normalized.csv')

        #getting % of mutation matches per threshold (not inclusive of higher thresholds)
        proportion_mat = analysis_mat.copy()
        for context_type in ['blind_contexts','naive_contexts','full_contexts']:
            temp_thresholds = thresholds[::-1]
            for threshold_index, threshold in enumerate(temp_thresholds):
                if threshold_index != 0:
                    if threshold_index != 1:
                        previous_vals = np.sum(proportion_mat.loc[:,(temp_thresholds[:threshold_index],context_type)], axis=1)
                    else:
                        previous_vals = proportion_mat.loc[:,(temp_thresholds[threshold_index-1],context_type)]
                    proportion_mat.loc[:,(threshold,context_type)] = proportion_mat.loc[:,(threshold,context_type)] - previous_vals
            proportion_mat.loc[:,(thresholds,context_type)] = proportion_mat.loc[:,(thresholds,context_type)] / np.reshape(np.repeat(analysis_mat.loc[:,(temp_thresholds[-1], context_type)].to_numpy(), len(thresholds)), (-1, len(thresholds)))
        #print(proportion_mat)
        proportion_mat.to_csv('simulation_output/global/'+analysis_folder+'/thresholded_mat_proportions.csv')
                
        #adding values for graph
        for mut_type in ['positional_matches','contextual_matches']:
            for context_type in ['blind_contexts','naive_contexts','full_contexts']:
                #print(np.mean(normalized_mat.loc[(analysis_variants,mut_type),(thresholds,context_type)].to_numpy()))
                #graph_df.loc[(mut_type,num_muts),context_type] = np.mean(normalized_mat.loc[(analysis_variants,mut_type),(thresholds,context_type)].to_numpy())
                graph_df[dataset_type+mut_type+context_type+str(num_muts)] = pd.Series([dataset_type, mut_type, num_muts, context_type, np.mean(normalized_mat.loc[(analysis_variants,mut_type),(thresholds,context_type)].to_numpy())], index=['dataset_type','mut_type','mut_count','context_type','value'])'''

    '''#graph_df_copy = graph_df.copy()
    print(graph_df)
    #graph_df.reindex(index=pd.MultiIndex.from_tuples(graph_df.loc['mut_type',:],graph_df.loc['mut_count',:]), columns=graph_df.loc['context_type',:])
    graph_index = pd.MultiIndex.from_product([np.unique(graph_df.loc['mut_type']), np.unique(graph_df.loc['mut_count'])])
    #graph_columns = ['blind_contexts','naive_contexts','full_contexts']#np.unique(graph_df.loc['context_type'])
    graph_columns = pd.MultiIndex.from_product([['population','weighted','unweighted'],['blind_contexts','naive_contexts','full_contexts']])
    graph_df_temp = pd.DataFrame(np.zeros([len(graph_index),len(graph_columns)]), index=graph_index, columns=graph_columns)
    for col in graph_df.columns:
        row = graph_df[col]
        graph_df_temp.loc[(row['mut_type'],row['mut_count']),(row['dataset_type'],row['context_type'])] = row['value']
    graph_df = graph_df_temp
    print(graph_df)
    graph_df.to_csv('simulation_output/final_info/final_sim_analysis/graph_df.csv')'''

#check the count/proportion of mutations placed in each region of the genome during simulation
def analyze_sim_muts_per_gene(sim_folder, analysis_variant, gene_regions_dict, sub_dict_cols):
    output_df = pd.DataFrame(np.zeros([12,len(gene_regions_dict)]), index=pd.MultiIndex.from_product([['full','tstv','naive'],['count','gene_size','proportion_local','proportion_global']]), columns=[gene for gene in gene_regions_dict.keys()])

    sim_type_names = {'full_contexts':'full', 'naive_contexts':'tstv', 'blind_contexts':'naive'} #renaming sim types
    for sim_type in ['full_contexts','naive_contexts','blind_contexts']:
        print(sim_type)
        num_files = 0
        for sim_file in os.listdir(sim_folder+'/'+sim_type+'/'+analysis_variant+'/mut_dicts/5e-05/'):
            mut_df = pd.read_csv(sim_folder+'/'+sim_type+'/'+analysis_variant+'/mut_dicts/5e-05/'+sim_file, index_col=0, header=0, dtype={'position':np.float64})
            mut_positions = mut_df.loc[:,'position'].to_numpy()
            #for region_dict in gene_regions_dicts:
            for gene, positions in gene_regions_dict.items():
                output_df.loc[(sim_type_names[sim_type],'count'),gene] += mut_positions[(mut_positions>positions[0]) & (mut_positions<positions[1])].shape[0]
            num_files += 1
        print(output_df)
        output_df.loc[(sim_type_names[sim_type],'count'),:] = output_df.loc[((sim_type_names[sim_type]),'count'),:] / num_files
        for gene, positions in gene_regions_dict.items():
            output_df.loc[(sim_type_names[sim_type],'gene_size'),gene] = positions[1]-positions[0]
        for cols in sub_dict_cols:
            output_df.loc[(sim_type_names[sim_type],'proportion_local'),cols] = output_df.loc[(sim_type_names[sim_type],'count'),cols] / output_df.loc[(sim_type_names[sim_type],'count'),cols].sum(axis=None)
        output_df.loc[(sim_type_names[sim_type],'proportion_global'),:] = output_df.loc[(sim_type_names[sim_type],'count'),:] / output_df.loc[(sim_type_names[sim_type],'count'),:].sum(axis=None)
    print(output_df)
    output_df.to_csv('simulation_output/final_info/final_sim_analysis/mutation_distribution.csv')



def main():
    #setup info
    variant_order = ['alpha','delta','kraken','omicron','pirola', 'beta','epsilon','eta','gamma','hongkong','iota','kappa','lambda','mu']
    persistent_variants = ['alpha', 'delta', 'kraken', 'omicron', 'pirola']
    transient_variants = ['beta','epsilon','eta','gamma','hongkong','iota','kappa','lambda','mu']
    pooled_variants = ['all','transient','persistent','mut_vars']
    mut_variants = ['mut_total','mut_alpha','mut_beta','mut_delta','mut_gamma','mut_omicron','mut_usa']
    sim_variants = ['all','mut_total']
    num_sims = 100
    thresholds = ['5e-05', '0.0005', '0.005', '0.05', '0.5']
    simulation_type = ['global'] #gene_specific is deprecated
    contexts = ['blind_contexts', 'naive_contexts', 'full_contexts'] #blind = no contextual or resultant mutation info, naive = no contextual info, full = contextual and resultant mutation info
    #matching number of unique context mutations in 'all'
    num_muts = {variant:10 for variant in sim_variants} #number of mutations to place per simulation
    scalers = {'blind_contexts':.015, 'naive_contexts':.015, 'full_contexts':.25} #weighting of variant mutation matrices to increase or decrease number of runs per simulation needed to achieve x-number of mutations placed

    #create file structure
    if not os.path.exists('simulation_output'):
        os.mkdir('simulation_output')
    if not os.path.exists('simulation_output/final_info'):
        os.mkdir('simulation_output/final_info')
    if not os.path.exists('simulation_output/global'):
        os.mkdir('simulation_output/global')

    '''generate variant reference data'''
    #gen_context_dependent_info(dataset='population', variant_order=variant_order, gene_dict=(subset_genes | nsp_positions | rdrp_sub_domains | spike_sub_domains | overlapping_genes), fourfold_positions=fourfold_positions, thresholds=thresholds)
    #convert_mutation_dataset(lethals=True, exclude_c_to_t=False)

    #simulate polymorphisms in reference sequence based on cdm or cdp info
    #uses 'all' for population polymorphism data, 'mut_total' for in vivo mutation data
    dataset = 'all'
    output_info = str(num_muts[dataset])

    if dataset == 'all':
        global_avg_subset_mat, global_naive_subset_mat, global_blind_subset_mat = read_thresholded_global_mat('5e-05', persistent_variants+pooled_variants[:-1]) #pull mutation matrices for cdp data
    elif dataset == 'mut_total':
        global_avg_subset_mat, global_naive_subset_mat, global_blind_subset_mat = read_thresholded_global_mat('5e-05', mut_variants, jf_flag=1) #pull mutation matrices for cdm data

    for context_type in contexts:
        if context_type == 'full_contexts':
            init_sim(variant=dataset, num_sims=num_sims, num_muts=num_muts[dataset],
                            mut_mat=global_avg_subset_mat[(persistent_variants+pooled_variants[:-1]).index(dataset)],
                            contexts=context_type, threshold=thresholds[0], scaler=scalers[context_type])
        elif context_type == 'naive_contexts':
            init_sim(variant=dataset, num_sims=num_sims, num_muts=num_muts[dataset],
                            mut_mat=global_naive_subset_mat[(persistent_variants+pooled_variants[:-1]).index(dataset)],
                            contexts=context_type, threshold=thresholds[0], scaler=scalers[context_type])
        elif context_type == 'blind_contexts':
            init_sim(variant=dataset, num_sims=num_sims, num_muts=num_muts[dataset],
                            mut_mat=global_blind_subset_mat[(persistent_variants+pooled_variants[:-1]).index(dataset)],
                            contexts=context_type, threshold=thresholds[0], scaler=scalers[context_type])
    
    #simulation analysis
    #analyze_mutations_using_mut_dicts(['alpha','delta','all'], thresholds, '5e-05', 'alpha', contexts)
    analyze_sims_from_server_3(variant_order=variant_order+pooled_variants[:-1], thresholds=thresholds, contexts=contexts, analysis_threshold=thresholds[0], analysis_variant=dataset, output_info=output_info)
    analyze_sims_genes(thresholds, contexts, analysis_threshold=thresholds[0], analysis_variant=dataset, gene_dict=(subset_genes | nsp_positions | rdrp_sub_domains | spike_sub_domains), sim_folder='analysis_'+output_info, sim_output_flag=True)
    #gen_sim_hist_2(gene='total', context_type='full_contexts', sim_folder='analysis_'+output_info)
    #gen_sim_dft(gene='S', context_type='full_contexts', sim_folder='analysis_'+output_info)
    #post_sims_analysis(analysis_variants=[re.search(r'\(\w+\)',variant_folder).group(0)[1:-1] for variant_folder in os.listdir('sim_ref_data') if 'clade' in variant_folder and 'jean' not in variant_folder and 'jf' not in variant_folder], thresholds=thresholds)

    #wrangle analysis data into two dfs
    #analyze_sims_from_server_2(variant_order=variant_order_aggregate, thresholds=thresholds, num_sims=num_sims, contexts=contexts, threshold_mat=True, t_test_mat=False, output_mut_mat=grouped_global_full, anova=True, reduced_output=False) #grouped_global_full or global_avg_subset_mat
    #variant_mut_rates_2(global_avg_subset_mat, genes_avg_mat, variant_order, gene_order)
    #viz_gwtc()
    #calc_triplet_counts_post_sim(contexts, variant_order, thresholds, num_sims)
    #compare_pirola(['BA.2.86(pirola)_full_clade_10_5_23', 'BA.2.86(pirola)_full_clade_1_17_24', 'BA.2.86(pirola)_full_clade_3_19_24'])
    #compare_pirola_muts(['BA.2.86(pirola)_full_clade_10_5_23', 'BA.2.86(pirola)_full_clade_1_17_24', 'BA.2.86(pirola)_full_clade_3_19_24'])
    #search_genome(global_avg_subset_mat, genes_avg_mat, 1000, variant_order, 2)
    #gen_gene_sequences()
    #plot_triplet_count(size=(12,4))
    #spike_anova()
    
    #sim_result_hist([variant_order_aggregate[i] for i in unweighted_indices_lf1], '1e-20', show='all') #[variant_order_aggregate[i] for i in unweighted_indices_lf1]
    #randomize_correlation_matrices(variant_order_aggregate)
    #randomize_correlation_matrices_columns(variant_order_aggregate)
    #spike_hist_t_tests()
    
    
    '''analyze variants and aggregates'''
    #analyze_low_freq_muts(variant_order=variant_order_aggregate, t_1='1.5e-05', t_2='5e-05')
    #global_avg_subset_mat, global_naive_subset_mat, global_blind_subset_mat = read_thresholded_global_mat('0.5', variant_order_aggregate)
    #global_avg_subset_mat, global_naive_subset_mat = read_thresholded_global_mat('low_freq', variant_order_aggregate)
    #'omicron','kraken','pirola','delta','epsilon','gamma','aggregate','iota','alpha','beta' #'pirola','gamma','kraken','delta','omicron','alpha','beta','aggregate','epsilon','iota' #'omicron','delta','alpha','gamma','aggregate','beta','pirola','iota','epsilon','kraken'
    #corr_df, pvalue_df = correlate_variants(global_avg_subset_mat, variant_order=variant_order_aggregate, extra_variants=False, var_corr_order=['all','persistent','delta','alpha','pirola','omicron','transient','kraken'], vmin=.001, vmax=1) #'alpha','omicron','aggregate','delta','pirola','kraken'
    #variant_order_transients = ['beta','epsilon','eta','gamma','iota','kappa','lambda','mu']
    #global_avg_subset_mat_transients, global_naive_subset_mat_transients, global_blind_subset_mat_transients = read_thresholded_global_mat('5e-05', variant_order_transients)
    #corr_df, pvalue_df = correlate_variants(global_avg_subset_mat_transients, variant_order=variant_order_transients, extra_variants=False, var_corr_order=['beta','gamma','lambda','mu','epsilon','iota','kappa','eta'], vmin=.001, vmax=1)
    #corr_df.to_csv('simulation_output/final_info/final_tables/supp_tables/5e-5_full_corr_transients.csv')
    #pvalue_df.to_csv('simulation_output/final_info/final_tables/supp_tables/5e-5_full_p_values_transients.csv')
    #plot_variants_grid(global_avg_subset_mat, global_naive_subset_mat, variant_order_aggregate, shape='3_vars', threshold='5e-5') #8_vars or dynamic_aggregate
    
    #real_vaccine_correlation([read_thresholded_global_mat('1.5e-05', variant_order_aggregate)[0],read_thresholded_global_mat('low_freq', variant_order_aggregate)[0],read_thresholded_global_mat('5e-05', variant_order_aggregate)[0]], [variant_order_aggregate,variant_order_aggregate,variant_order_aggregate])

    #convert_jean_dataset()
    #variant_order_aggregate = ['jean_alpha','jean_beta','jean_delta','jean_gamma','jean_omicron','jean_usa','jean_total']
    #gen_gwtc_jean(pd.Series([*get_fasta()[:-1]], name='ref'), rows, columns_shortened)
    #variant_order_aggregate = ['jean_alpha']
      
    #parse_reference_mutations(thresholds=['5e-05', '5e-04', .005, .05, .5], variant_order=variant_order_aggregate, jean_toggle='weighted')
    #analyze_low_freq_muts(variant_order=variant_order_aggregate, t_1='5e-05', t_2='5e-04')
    #global_avg_subset_mat, global_naive_subset_mat = read_thresholded_global_mat('0', variant_order_aggregate)
    #correlate_variants(global_avg_subset_mat, variant_order=[label.split('_')[-1] for label in variant_order_aggregate], extra_variants=False, var_corr_order=['alpha','beta','delta','gamma','omicron','usa','total'], vmin=.3, vmax=1) # #['alpha','aggregate','gamma','omicron','pirola','delta','kraken']
    #plot_variants_grid(global_avg_subset_mat, global_naive_subset_mat, variant_order_aggregate, shape='aggregate', threshold='jean_weighted')
    






    '''call 1-off functions'''
    #compare_genes_thresholded(['alpha','beta','delta','epsilon','eta','gamma','hongkong','iota','kappa','kraken','lambda','mu','omicron','pirola'], nsp_positions, .3)
    #threshold_testing(variant_order_aggregate, thresholds)
    #correlate_jf_with_our_data(weighted=True)
    #gene_context_counts()
    #count_gene_muts_thresholded('5e-05') #5e-05 and .5
    #gene_context_count_analysis()
    #expected_vs_obs((subset_genes | nsp_positions | rdrp_sub_domains | spike_sub_domains))
    #analyze_contextual_influence('all', '5e-05')
    #meaningful_triplet_counts(nsp_positions | spike_sub_domains | {key:gene_info[key][1] for key in gene_info.keys()})
    #correlate_all_meaningful_contexts()
    #analyze_fourfold_thresholds(thresholds)
    #gen_aggregate_plots(global_avg_subset_mat[variant_order_aggregate.index('all')])
    #testing_alpha()
    #collate_mat_rates_and_variances(global_avg_subset_mat, global_naive_subset_mat, variant_order_aggregate)
    #compare_two_variants(['all','jean_total'])
    #gen_genome_context_counts()
    #calc_shared_muts_between_variants(variant_order, thresholds)
    #search_population_mutations((subset_genes | nsp_positions | rdrp_sub_domains | spike_sub_domains), window_sizes=[1,100,250,500,1000,1500,2000,2500,3000])
    #prep_sim_fasta_for_blast(context_type='full_contexts', analysis_variant='jean_total', analysis_threshold='5e-05', num_muts=100)
    #gen_all_vs_total_output_figure()
    #format_sim_fastas('jean_total', '5e-05', 1000, 'vivo')                                                                                                                                                 [5e-5,5e-4],[5e-4,5e-3],[5e-3,5e-2],[5e-2,.5],[.5,1]
    #low_and_high_figure(output_types=['sim','pop'], sim_folder_path='simulation_output/global/analysis_1000/genes/full_contexts/total_hist.csv', window_size=500, thresholds=[[5e-5,1]], regions_dict=(subset_genes | nsp_positions | rdrp_sub_domains)) #250,500,1000,1500,2000,2500,3000
    #convert_shared_mutations_list('4(unique_mutations)_full_clade', 5e-05, 1)
    #analyze_genome_context_counts((subset_genes | nsp_positions | rdrp_sub_domains | spike_sub_domains))
    #rdrp_table_fig()
    #analyze_sim_muts_per_gene('simulation_output/global/analysis_1000', 'all', (temp_gene_dict | nsp_positions | rdrp_sub_domains | spike_sub_domains), [[gene for gene in temp_gene_dict.keys()], [gene for gene in nsp_positions.keys()], [gene for gene in rdrp_sub_domains.keys()]])
    #plot_mut_mat(global_naive_subset_mat[variant_order_aggregate.index('all')], 'sim_ref_data/0(all)_full_clade/thresholded_mutations/5e-05_tstv_mat.png', None, 'global', 'naive_contexts', include_a=False, vmin=0, vmax=1, annot=True)
    #plot_mut_mat(global_blind_subset_mat[variant_order_aggregate.index('all')], 'sim_ref_data/0(all)_full_clade/thresholded_mutations/5e-05_naive_mat.png', None, 'global', 'super_naive_contexts', include_a=False, vmin=0, vmax=1, annot=True)
    #vaccine_corr_updated(global_avg_subset_mat, variant_order_aggregate)
    #xpehh_testing((subset_genes | nsp_positions | rdrp_sub_domains | spike_sub_domains))
    #analyze_gene_mutation_frequencies(variant_order_aggregate+['beta','epsilon','eta','gamma','iota','kappa','lambda','mu','hongkong'], (subset_genes | nsp_positions | rdrp_sub_domains | spike_sub_domains))
    #analyze_introns()
    #calc_contexts_of_regions({1:[1,1299],2:[3476,4129],3:[4656,7092],4:[13014,13549],5:[16394,17014],6:[23528,24139],7:[25114,25799],8:[26342,27249],9:[27016,27686],10:[28059,29319]})
    #calc_contexts_of_regions({1:[1,1081],2:[769,1299],3:[3476,4129],4:[4656,5427],5:[5385,6373],6:[5927,6447],7:[6027,6680],8:[6233,7092],9:[13014,13549],10:[16394,17014],11:[23528,24050],12:[23621,24139],13:[25144,25659],14:[25180,25760],15:[25276,25799],16:[26342,26862],17:[26401,27249],18:[27016,27516],19:[27052,27516],20:[27181,27634],21:[28059,28576],22:[28309,28874],23:[28407,28990],24:[28502,29058],25:[28597,29157],26:[28703,29319]})
    #gen_mutated_fasta(['RBD'], collapse=False, gene_dict=(subset_genes | nsp_positions | rdrp_sub_domains | spike_sub_domains), variants=['pirola'])
    #analyze_who_data()
    #gen_gisaid_ids()
    #reformat_pos_con_fig()
    #plot_cdm_types()
    #gen_subset_gene_info((subset_genes | nsp_positions | rdrp_sub_domains | overlapping_genes))
    #gene_site_analysis((subset_genes | nsp_positions | spike_sub_domains | n_sub_domains), gene_info)
    #gen_variant_seq_dist_fig()
    #gen_sim_mut_count_fig()
    #gen_sim_mut_thresholding_fig()
    #compare_vivo_rate_dist()
    #analyze_snps_in_population((subset_genes | rdrp_sub_domains | spike_sub_domains | overlapping_genes))


if __name__ == '__main__':
    main()