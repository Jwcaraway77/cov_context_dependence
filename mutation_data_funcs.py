import numpy as np
import pandas as pd
import torch, os
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec
import seaborn as sns
from scipy import stats
import functools
import time
import re

'''file holding functions to try and replicate jf's matrix from dataset'''

rows = ["T[X->Y]T","T[X->Y]G","T[X->Y]C","T[X->Y]A","G[X->Y]T","G[X->Y]G","G[X->Y]C","G[X->Y]A","C[X->Y]T","C[X->Y]G","C[X->Y]C","C[X->Y]A"]
rows_figs = ["U[X->Y]U","U[X->Y]G","U[X->Y]C","U[X->Y]A","G[X->Y]U","G[X->Y]G","G[X->Y]C","G[X->Y]A","C[X->Y]U","C[X->Y]G","C[X->Y]C","C[X->Y]A","A[X->Y]U","A[X->Y]G","A[X->Y]C","A[X->Y]A"]
columns = ["T>G","T>C","T>A","G>T","G>C","G>A","C>T","C>G","C>A","A>T","A>G","A>C"]
columns_figs = ["U>G","U>C","U>A","G>U","G>C","G>A","C>U","C>G","C>A","A>U","A>G","A>C"]
columns_shortened = ['T','G','C','A']
columns_shortened_figs = ['U','G','C','A']

'''read in reference fasta'''
def get_fasta():
    with open('EPI_ISL_402124.fasta', 'r') as f:
        lines = f.readlines()
    return lines[1]

'''plot a mutation matrix to path'''
def plot_mut_mat(variant_mat, path, gene_order, data_type, contexts, include_a=False, vmin=0, vmax=1, annot=False):
    #potential shapes: 15x16x12 = genes full mat, 15x12 = genes naive mat, 16x12 = global full mat, 12 = global naive mat
    print('generating ' + path)
    print(variant_mat.shape)
    if 'full' in contexts:
        rows_temp = rows_figs
    else:
        rows_temp = ['N[X-->Y]N']
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
            fig, axs = plt.subplots(figsize=(6,2))
            sns.heatmap(torch.unsqueeze(variant_mat,0), cmap='Greys', ax=axs, vmin=vmin, vmax=vmax, annot=annot, linewidth=.5, linecolor='gray')
            axs.set_xticklabels(labels=columns_shortened_figs)
            axs.set_yticklabels(labels=rows_temp, rotation='horizontal')
        elif contexts == 'naive_contexts':
            fig, axs = plt.subplots(figsize=(8,2))
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

#calculate mutation spectrum based on supplied mutations
def compute_spectrum_from_muts(df, sample_names, l_min_cov, l_max_rate, nl_min_cov, nl_max_rate, max_before_hs, fasta, data_subset):

    #initialize empty spectrum matrices
    mut_rate_mat = pd.DataFrame(np.zeros([16,12]), index=rows+['A[X->Y]T','A[X->Y]G','A[X->Y]C','A[X->Y]A'], columns=columns)
    tstv_mut_rate_mat = pd.DataFrame(np.zeros([1,12]), index=['N[X->Y]N'], columns=columns)
    naive_mut_rate_mat = pd.DataFrame(np.zeros([1,4]), index=['N[X->Y]N'], columns=['T','G','C','A'])

    #loop through each sample
    for sample in sample_names:
        print('computing spectrum for ' + sample)
        #subset df to only include current sample
        df_sample = df.loc[:,['position','ref','bTo',sample+'_muts', sample+'_cov','upstream_nuc','downstream_nuc','lethal']]
        
        if data_subset in ['lethals', 'both']:
            #drop sites where coverage does not reach threshold
            df_sample_lethals = df_sample.loc[(df_sample[sample+'_cov'] >= l_min_cov) & (df_sample['lethal']==True)].copy()
            #drop sites where frequency > max_rate
            df_sample_lethals['frequency'] = df_sample_lethals.loc[:,sample+'_muts'] / df_sample_lethals.loc[:,sample+'_cov']
            df_sample_lethals = df_sample_lethals.loc[df_sample_lethals['frequency'] <= l_max_rate]
        if data_subset in ['non-lethals', 'both']:
            #drop sites where coverage does not reach threshold
            df_sample_nonlethals = df_sample.loc[(df_sample[sample+'_cov'] >= nl_min_cov) & (df_sample['lethal']==False)].copy()
            #drop sites where frequency > max_rate
            df_sample_nonlethals['frequency'] = df_sample_nonlethals.loc[:,sample+'_muts'] / df_sample_nonlethals.loc[:,sample+'_cov']
            df_sample_nonlethals = df_sample_nonlethals.loc[df_sample_nonlethals['frequency'] <= nl_max_rate]
        if data_subset == 'lethals':
            df_sample = df_sample_lethals
        elif data_subset == 'non-lethals':
            df_sample = df_sample_nonlethals
        else:
            df_sample = pd.concat([df_sample_lethals, df_sample_nonlethals])
        
    
        #loop through each context
        for upstream_nuc in columns_shortened:
            for downstream_nuc in columns_shortened:
                #loop through each mutation type
                for ref_nuc in columns_shortened:
                    for mut_nuc in columns_shortened:
                        if ref_nuc != mut_nuc:
                            #subset df_sample
                            df_sample_muts = df_sample.loc[(df_sample['upstream_nuc']==upstream_nuc)&(df_sample['downstream_nuc']==downstream_nuc)&(df_sample['ref']==ref_nuc)&(df_sample['bTo']==mut_nuc)]
                            #store avg frequency
                            mut_rate_mat.loc[upstream_nuc+'[X->Y]'+downstream_nuc, ref_nuc+'>'+mut_nuc] = df_sample_muts['frequency'].mean(axis=None)
                            #store average frequency for contextless matrix
                            tstv_mut_rate_mat.loc['N[X->Y]N', ref_nuc+'>'+mut_nuc] = df_sample.loc[(df_sample['ref']==ref_nuc)&(df_sample['bTo']==mut_nuc)]['frequency'].mean(axis=None)
                            #store average frequency for contextless and resultantless matrix
                            naive_mut_rate_mat.loc['N[X->Y]N', ref_nuc] = df_sample.loc[df_sample['ref']==ref_nuc]['frequency'].mean(axis=None)

        var_num = sample_names.index(sample)
        if not os.path.exists('sim_ref_data/m'+str(var_num)+'(mut_'+sample+')_full_clade'):
            os.mkdir('sim_ref_data/m'+str(var_num)+'(mut_'+sample+')_full_clade')
        mut_rate_mat.to_csv('sim_ref_data/m'+str(var_num)+'(mut_'+sample+')_full_clade/'+data_subset+'_mut_rate_mat.csv')
        tstv_mut_rate_mat.to_csv('sim_ref_data/m'+str(var_num)+'(mut_'+sample+')_full_clade/'+data_subset+'_tstv_mut_rate_mat.csv')
        naive_mut_rate_mat.to_csv('sim_ref_data/m'+str(var_num)+'(mut_'+sample+')_full_clade/'+data_subset+'_naive_mut_rate_mat.csv')

#https://github.com/jfgout/SARS-CoV-2/tree/main/data
#new method to more closely resemble their scripts
def convert_mutation_dataset(lethals=True, exclude_c_to_t=False):
    min_cov = 1000
    max_rate = 1/min_cov
    max_before_hs = 1000
    fasta = np.array([*get_fasta()[:-1]])


    #read in dataset
    df = pd.read_csv('sim_ref_data/mutation_data.csv', index_col=0, header=0)
    #append columns for 5' and 3' neighboring nucleotides for easier manipulation
    df['upstream_nuc'] = np.repeat(['', *fasta][:-1], 3)
    df['downstream_nuc'] = np.repeat([*fasta,''][1:], 3)

    #lines 30:38
    #remove C->T mutations if flag
    if exclude_c_to_t:
        df = df.loc[(df['ref'] != 'C') & (df['bTo'] != 'T')]
    
    #lines 40:47
    #update lethal flag to False for mutations with a frequency higher than max_rate
    #don't know if this is per sample, going to try with using 'total'
    df.loc[(df.loc[:,'total_muts']/df.loc[:,'total_cov'])>max_rate, 'lethal'] = False
    #lines 49:53
    #subsetting into 2 dfs, lethal and non-lethal muts
    print('total df size', df.shape)
    df_lethals = df.loc[df['lethal']==True]
    df_non_lethals = df.loc[df['lethal']==False]
    print(f'lethals size: {df_lethals.shape} and non-lethals: {df_non_lethals.shape}')
    
    #lines 55:60
    #pulling column names, and sample names
    column_names = df.columns
    mutation_columns = [column for column in column_names if 'muts' in column]
    sample_names = np.unique([column[:-5] for column in mutation_columns])
    #print(mutation_columns, sample_names)

    #convert all mutations into mutation spectrum with separate thresholds for lethals and non-lethals

    compute_spectrum_from_muts(
        df = df_lethals,
        sample_names = sample_names,
        l_min_cov = 1,
        l_max_rate = .1,
        nl_min_cov = min_cov,
        nl_max_rate = max_rate,
        max_before_hs = max_before_hs,
        fasta = fasta,
        data_subset = 'lethals')

    compute_spectrum_from_muts(
        df = df_non_lethals,
        sample_names = sample_names,
        l_min_cov = 1,
        l_max_rate = .1,
        nl_min_cov = min_cov,
        nl_max_rate = max_rate,
        max_before_hs = max_before_hs,
        fasta = fasta,
        data_subset = 'non-lethals')

    compute_spectrum_from_muts(
        df = df,
        sample_names = sample_names,
        l_min_cov = 1,
        l_max_rate = .1,
        nl_min_cov = min_cov,
        nl_max_rate = max_rate,
        max_before_hs = max_before_hs,
        fasta = fasta,
        data_subset = 'both')




























    '''#col_labels = [prefix + suffix for prefix in ['total','USA','ALPHA','DELTA', 'BETA_A_1', 'GAMMA_A_1', 'OMICRON_A_1'] for suffix in ['_muts', '_cov']]
    #df = df.loc[:, (['position', 'uid'] + col_labels + ['lethal'])] #subset dataset

    #print(df)

    #rows = 29903 * 3, each position has 3 potential mutations
    #cols = position, id, ref, mut, [count, depth]*#variant lines, lethality

    #labels of variant lines
    var_labels = [(prefix+suffix) for prefix in ['ALPHA','BETA_A_1','DELTA','GAMMA_A_1','OMICRON_A_1','USA', 'total'] for suffix in ['_muts','_cov']]
    output_df = np.empty([15])

    #loop through each position,mutation combo in dataset
    for row_index in df.index.to_numpy():
        #look at total read count for mutation
        total = df.loc[row_index, 'total_muts']
        #check if mutation exists
        if total != 0:
            #reorder uid to match pipeline
            mutation = df.loc[row_index, 'uid'].split('_')
            mutation = ''.join([mutation[1],mutation[0],mutation[2]])

            #add new uid, and read counts for each variant line
            temp_row = np.append(np.array(mutation), df.loc[row_index,(var_label for var_label in var_labels)].to_numpy())
            output_df = np.append(output_df, temp_row)
            
    #convert np array to pd dataframe
    output_df = pd.DataFrame(output_df[15:].reshape(-1,15), columns=['mutation']+var_labels)

    #print('--output_df', output_df)
    
    #loop through variant lines
    for var_label in var_labels[::2]:
        #subset df to mutations that variant has
        var_df = output_df.loc[output_df[var_label] != 0, ('mutation',var_label,'_'.join(var_label.split('_')[:-1]+['cov']))]
        #print(var_df)
        #append dummy columns to match pipeline
        #print('--output_df subset', output_df.loc[output_df.loc[output_df[var_label] != 0].index])
        var_df['proportion'] = (var_df.loc[:,var_label] / var_df.loc[:,'_'.join(var_label.split('_')[:-1]+['cov'])]).astype(float)
        #var_df['proportion'] = np.round(var_df['proportion'].to_numpy(), 10)
        var_df['jaccard'] = np.zeros([var_df.shape[0]])
        #print(var_df)
        #reorder df
        var_df.drop(var_df.loc[var_df[['_'.join(var_label.split('_')[:-1]+['cov'])][0]] < 1000].index, inplace=True) #remove sites with coverage lower than 1000
        var_df.drop(['_'.join(var_label.split('_')[:-1]+['cov'])], axis=1, inplace=True)
        var_df.columns = ['mutation','count','proportion','jaccard']
        #save variant mutation data to csv in reference folder
        var_folder = [folder for folder in os.listdir('sim_ref_data') if '(jean_'+var_label.split('_')[0].lower()+')' in folder][0]
        var_df.loc[:,('mutation','proportion','count','jaccard')].to_csv('sim_ref_data/'+var_folder+'/nucleotide-mutations.csv', index=False, float_format='%.15f')
        #count max number of reads at any site for variant
        #print(var_label, df.loc[:,'_'.join(var_label.split('_')[:-1])+'_cov'].max())'''

#collapse mutations at each position so A->T, A->G, A->C is all one row
def collapse_mutation_list(variant):
    variant_folder_name = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder][0]
    variant_mutations = pd.read_csv('sim_ref_data/'+variant_folder_name+'/nucleotide-mutations.csv', header=0)
    
    #change formatting of variant mutations
    formatting_timer = time.perf_counter()
    
    variant_mutations['position'] = [int(variant_mutations.loc[i,'mutation'][1:-1]) for i in range(variant_mutations.shape[0])]
    variant_mutations['original'] = [variant_mutations.loc[i,'mutation'][0] for i in range(variant_mutations.shape[0])]
    variant_mutations['mutation'] = [variant_mutations.loc[i,'mutation'][-1] for i in range(variant_mutations.shape[0])]
    #drop indels
    variant_mutations.drop(variant_mutations.loc[variant_mutations['mutation']=='-'].index, inplace=True)
    #variant_mutations.drop(variant_mutations.loc[variant_mutations[]])
    #sort by position
    variant_mutations.sort_values(by='position', inplace=True)
    #reindex
    variant_mutations.index = np.arange(variant_mutations.shape[0])
    positions = np.unique(variant_mutations.loc[:,'position'].to_numpy())
    #output_mat
    variant_mutations_collapsed = pd.DataFrame(np.zeros([len(positions),10]), index=positions, columns = ['position', 'original', 'T', 'G', 'C', 'A', 'T_freq', 'G_freq', 'C_freq', 'A_freq'])
    
    #loop through mutated positions
    for position in positions:
        #update position and original nucleotide
        position_df = variant_mutations.loc[variant_mutations['position']==position]
        variant_mutations_collapsed.loc[position, 'position'] = position
        variant_mutations_collapsed.loc[position, 'original'] = position_df.iloc[0, -1]
        #loop through mutations at position
        for index in position_df.index:
            variant_mutations_collapsed.loc[position, position_df.loc[index,'mutation']] = position_df.loc[index, 'count']
            variant_mutations_collapsed.loc[position, position_df.loc[index,'mutation']+'_freq'] = position_df.loc[index, 'proportion']
    collapsing_timer = time.perf_counter()
    print(f'reformatting timer {collapsing_timer - formatting_timer}')
    
    variant_mutations_collapsed.to_csv('sim_ref_data/'+variant_folder_name+'/collapsed_mutation_list.csv')

#updated to use collapsed mutation list instead (saves time)
#added in 'count' column to reference_mutations output df to recalc depth at site
def threshold_reference_mutations(variant, threshold):
    reference_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder][0]

    #create reference mutation dictionary for threshold
    if not os.path.exists('sim_ref_data/'+reference_folder+'/reference_mutations/'):
        os.mkdir('sim_ref_data/'+reference_folder+'/reference_mutations/')
    #if not os.path.exists('sim_ref_data/'+reference_folder+'/reference_mutations/'+str(threshold)+'_reference_mutations.csv'):
    print('generating reference_mutations for threshold ', str(threshold))
    
    #read in total nucleotide mutations for variant
    reference_mutations_temp = pd.read_csv('sim_ref_data/'+reference_folder+'/collapsed_mutation_list.csv', header=0)
    #print(reference_mutations_temp.shape)

    #drop mutations with frequency < threshold
    reference_mutations = pd.DataFrame()
    #loop through base substitutions
    for mut in ['T','G','C','A']:
        subset_df = reference_mutations_temp.loc[reference_mutations_temp[mut+'_freq'].astype(float) >= float(threshold)]
        subset_df = subset_df.loc[:,['position', mut, mut+'_freq', 'original']]
        subset_df = pd.concat([subset_df, pd.Series(np.array([mut]*subset_df.shape[0]), index=subset_df.index)], axis=1)
        subset_df.columns = ['position','count','proportion','old','mut']
        #print(subset_df.shape)
        reference_mutations = pd.concat([reference_mutations, subset_df], axis=0)
    #print(reference_mutations.shape)

    #convert datatype
    #print(reference_mutations['position'])
    reference_mutations['position'] = np.asarray(reference_mutations['position'].values, dtype=int)
    reference_mutations.sort_values(by='position', inplace=True)
    reference_mutations.reset_index(drop=True, inplace=True)
    reference_mutations.to_csv('sim_ref_data/'+reference_folder+'/reference_mutations/'+str(threshold)+'_reference_mutations.csv')

#building a new one
def convert_reference_mutation_list_to_figure(variant, threshold, jean_toggle='weighted'):
    #read in ref fasta
    fasta = pd.Series([*get_fasta()[:-1]], name='ref')

    #read in reference mutation list
    var_folder = [folder for folder in os.listdir('sim_ref_data/') if '('+variant+')' in folder][0]
    ref_mut_list = pd.read_csv('sim_ref_data/'+var_folder+'/reference_mutations/'+str(threshold)+'_reference_mutations.csv', index_col=0, header=0)

    if not 'jean' in variant:
        #read in total genome triplet counts
        total_triplet_counts = pd.read_csv('sim_ref_data/4fold/gwtc/triplets/total.csv', index_col=0, header=0)

        #read in valid 4fold positions
        valid_fourfold_positions = pd.read_csv('sim_ref_data/4fold/gwtc/valid_fourfold_positions/total.csv', index_col=0, header=0).to_numpy().flatten()

        #drop mutations not at valid 4fold sites
        ref_mut_list = ref_mut_list.loc[ref_mut_list['position'].isin(valid_fourfold_positions+1)].dropna(how='all')
        
    else:
        #read in total genome triplet counts
        total_triplet_counts = pd.read_csv('sim_ref_data/4fold/contexts/jean_total.csv', index_col=0, header=0)
        if total_triplet_counts.shape[0]>12:
            total_triplet_counts.drop(['A[X-->Y]T','A[X-->Y]G','A[X-->Y]C','A[X-->Y]A'], axis=0, inplace=True)
        #df storing the number of mutations that were analyzed at each context
        mut_counts = pd.DataFrame(np.zeros([12,12]), index=rows, columns=columns)
        #df storing the number of sequences analyzed at each context
        coverage = pd.DataFrame(np.zeros([12,12]), index=rows, columns=columns)

    var_matrix = pd.DataFrame(np.zeros([12,12]), index=rows, columns=columns)
    mut_count = 0
    #jean testing dict
    #testing_dict = {row:{column:[] for column in columns} for row in rows}
    for index in ref_mut_list.index.values:
        #position,old,mut,combined
        row = ref_mut_list.loc[index,:]
        #print(row)
        triplet = fasta.loc[row.loc['position']-2:row.loc['position']].values #indexed -1 to match fasta
        #default to counting each mutation as 1 individual mutation (don't have consecutive sequence info)
        if not 'jean' in variant:
            var_matrix.loc[triplet[0]+'[X->Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += 1
        else:
            #check if valid mutation for jean dataset (we can't use A[X->Y]N)
            if triplet[0]+'[X->Y]'+triplet[2] in var_matrix.index.to_numpy():
                #calc based on frequencies
                if jean_toggle == 'weighted':
                    #var_matrix.loc[triplet[0]+'[X->Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += row.loc['proportion']
                    #testing_dict[triplet[0]+'[X->Y]'+triplet[2]][row.loc['old']+'>'+row.loc['mut']].append(float(row.loc['proportion']))
                    
                    
                    var_matrix.loc[triplet[0]+'[X->Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += row.loc['count']
                    mut_counts.loc[triplet[0]+'[X->Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += 1
                    coverage.loc[triplet[0]+'[X->Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += row.loc['count'] / row.loc['proportion']
                #calc based on unique mutations
                else:
                    var_matrix.loc[triplet[0]+'[X->Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += 1
        mut_count+=1
    #jean testing stuff
    '''max_length = np.max([[len(testing_dict[row][column]) for column in testing_dict[row].keys()] for row in testing_dict.keys()])
    empty_arr = np.empty([144,max_length])
    empty_arr.fill(np.nan)
    testing_df = pd.DataFrame(empty_arr, index=pd.MultiIndex.from_product([rows,columns]))
    #print(testing_df)
    for key, row in testing_dict.items():
        for key_2, column in testing_dict[key].items():
            #print(column)
            #print(testing_df.loc[(key,key_2),:])
            for val_index in range(len(column)):
                testing_df.loc[(key,key_2),val_index] = column[val_index]
    sum_col = pd.Series(testing_df.sum(axis=1), index=testing_df.index, name='sum')
    #print(sum_col, sum_col.shape)
    for row_index in sum_col.index:
        sum_col.loc[row_index] /= total_triplet_counts.loc[row_index[0],row_index[1][0]]
    #normalize sum col by triplet count
    testing_df = pd.concat([sum_col, testing_df], axis=1)
    testing_df.to_csv('simulation_output/jean_figs/tensors/'+variant+'_'+str(threshold)+'.csv')'''

    if not os.path.exists('simulation_output/final_info/variant_mut_count_mats_4fold'):
        os.mkdir('simulation_output/final_info/variant_mut_count_mats_4fold')
    if not os.path.exists('simulation_output/final_info/variant_mut_count_mats_4fold/'+str(threshold)):
        os.mkdir('simulation_output/final_info/variant_mut_count_mats_4fold/'+str(threshold))
    var_matrix.to_csv('simulation_output/final_info/variant_mut_count_mats_4fold/'+str(threshold)+'/'+variant+'_mut_counts.csv')
    
    #normalize by 4fold triplet counts
    #if jean_toggle == 'unweighted' or 'jean' not in variant:
    if jean_toggle == 'weighted':
        var_matrix.to_csv('simulation_output/jf_testing/'+variant+'_counts.csv')
        var_matrix /= coverage #num_muts / num_seqs = num_freq
        var_matrix.to_csv('simulation_output/jf_testing/'+variant+'_freqs.csv')
        var_matrix /= mut_counts #num_freq / num_muts = avg_num_freq
        var_matrix.to_csv('simulation_output/jf_testing/'+variant+'_avg_freqs.csv')
        coverage.to_csv('simulation_output/jf_testing/'+variant+'_coverage.csv')
    for column in var_matrix.columns.values:
        var_matrix.loc[:,column] = var_matrix.loc[:,column].to_numpy() / total_triplet_counts.loc[:,column[0]].to_numpy()
    return var_matrix, mut_count

#convert_reference_mutation_list for each threshold,variant combo and normalize results by mut_count
def parse_reference_mutations(thresholds, variant_order, jean_toggle):

    #loop through thresholds
    for threshold in thresholds:
        var_data = []
        max_rate = 0
        #convert_reference_mutation_list
        for variant in variant_order:
            var_data.append([*convert_reference_mutation_list_to_figure(variant, threshold, jean_toggle)])
        counts = np.asarray([var_data[i][1] for i in range(len(var_data))])
        print(counts)    
        
        #save var_matrix and normalize heatmap by max of variant rates
        for variant_index, variant in enumerate(variant_order):
            var_matrix = var_data[variant_index][0]
            #save and plot var_matrix
            var_folder = [var_folder for var_folder in os.listdir('sim_ref_data') if '('+variant+')' in var_folder and 'full_clade' in var_folder][0]
            if not os.path.exists('sim_ref_data/'+var_folder+'/thresholded_mutations'):
                os.mkdir('sim_ref_data/'+var_folder+'/thresholded_mutations')
            #cols_temp = var_matrix.columns
            #index_temp = var_matrix.index
            #var_matrix += .0000000000000001
            #print(var_matrix)
            #var_matrix = np.log10(var_matrix)
            #print(var_matrix)
            #var_matrix = np.where(var_matrix==float('-inf'), float('1e-10'), var_matrix)
            #var_matrix = pd.DataFrame(var_matrix, index=index_temp, columns=cols_temp)
            #var_matrix /= var_matrix.max(axis=None)
            var_matrix.to_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/'+str(threshold)+'_mat.csv')
            #print(max_rate)
            #if 'jean' in var_folder:
            #    var_matrix = var_matrix / var_matrix.sum(axis=None) #normalize
            #    var_matrix = (var_matrix - var_matrix.min(None)) / (var_matrix.max(None)-var_matrix.min(None))
            
            #var_matrix /= var_matrix.sum(axis=None)
            #var_matrix /= var_matrix.max(axis=None)
            max_rate = var_matrix.max(axis=None)
            if len(max_rate) > 1:
                max_rate = max_rate[0]
            min_rate = var_matrix.min(axis=None)
            if len(min_rate) > 1:
                min_rate = min_rate[0]
            plot_mut_mat(var_matrix.to_numpy(), 'sim_ref_data/'+var_folder+'/thresholded_mutations/'+str(threshold)+'_full_contexts_global.png', [], 'global', 'full_contexts', vmin=min_rate, vmax=max_rate, annot=True)

#look at mutability of each gene and variant
def analyze_genes():
    #each gene and its starting and ending positions
    gene_dict = {'orf1a':[266,13483],'orf1b':[13483,21555],'s':[21563,25384],'orf3a':[25393,26220],'orf3b':[25765,26220],'e':[26245,26472],'m':[26523,27191],'orf6':[27202,27387],'orf7a':[27394,27759],'orf7b':[27756,27887],'orf8':[27894,28259],'n':[28274,29533],'orf9b':[28284,28577],'orf9c':[28734,28955],'orf10':[29558,29674],
                 'nsp1':[266,806],'nsp2':[806,2720],'nsp3':[2720,8555],'nsp4':[8555,10055],'nsp5':[10055,10973],'nsp6':[10973,11843],'nsp7':[11843,12092],'nsp8':[12092,12686],'nsp9':[12686,13025],'nsp10':[13025,13442],'nsp11':[13442,13481],'nsp12':[13442,16238],'nsp13':[16238,18041],'nsp14':[18041,19622],'nsp15':[19622,20660],'nsp16':[20660,21554],
                 'NiRAN':[13441,14191],'interface':[14191,14536],'hand':[14542,16201]}
    #variants analyzed
    variants = ['ALPHA','BETA_A_1','DELTA','GAMMA_A_1','OMICRON_A_1','USA','total']
    #thresholds for mutation frequency
    thresholds = [0,5e-5,5e-4,5e-3,5e-2,5e-1,1]
    #output dataframe
    output_df = pd.DataFrame(np.zeros([len(gene_dict),(len(thresholds)-1)*len(variants)]), index=gene_dict.keys(), columns=pd.MultiIndex.from_product([thresholds[:-1],variants]))

    #loop through variants
    for variant in variants:
        #read in muts
        mut_df = pd.read_csv('simulation_output/jf_testing/'+variant+'/both_subset_dataframe.csv', index_col=0, header=0)
        #loop through genes
        for gene, positions in gene_dict.items():
            #subset muts to only those in the gene
            subset_muts = mut_df.loc[(mut_df['position']>=positions[0])&(mut_df['position']<=positions[1])]
            for threshold_index, threshold in enumerate(thresholds[:-1]):
                thresholded_muts = subset_muts.loc[(subset_muts['frequency']>=threshold)&(subset_muts['frequency']<=thresholds[threshold_index+1])]
                output_df.loc[gene,(threshold,variant)] = thresholded_muts.shape[0]
    output_df['length'] = [gene_dict[gene][1]-gene_dict[gene][0] for gene in gene_dict.keys()]
    output_df.to_csv('simulation_output/final_info/jf_gene_analysis.csv')












def main():
    '''#setup info
    #variant_order = ['alpha','delta','kraken','omicron','pirola', 'all','dominant','transient', 'beta','epsilon','eta','gamma','hongkong','iota','kappa','lambda','mu']
    #variant_order_aggregate = ['alpha', 'delta', 'kraken', 'omicron', 'pirola', 'all', 'dominant', 'transient']
    variant_order_aggregate = ['jean_total','jean_alpha','jean_beta','jean_delta','jean_gamma','jean_omicron','jean_usa']
    num_sims = 100
    thresholds = ['1e-30']
    simulation_type = ['global'] #gene_specific is deprecated
    contexts = ['blind_contexts', 'naive_contexts', 'full_contexts'] #blind = no contextual or resultant mutation info, naive = no contextual info, full = contextual and resultant mutation info
    num_muts = {variant:4000 for variant in variant_order_aggregate} #number of mutations to place per simulation
    scalers = {'blind_contexts':.03, 'naive_contexts':.03, 'full_contexts':.5} #weighting of variant mutation matrices to increase or decrease number of runs per simulation needed to achieve x-number of mutations placed
    #.03, .03, .5
    #.015, .015, .3

    #''generate variant reference data'' #(subset_genes | nsp_positions | rdrp_sub_domains) 
    #calc_fourfold_positions_for_subset_genes((subset_genes | nsp_positions | rdrp_sub_domains), fourfold_positions) #calc fourfold site info
    #''reformat and subset mutation data for variants''
    for variant in ['jean_total']:#variant_order_aggregate: #variant_order
        collapse_mutation_list(variant)
        for threshold in thresholds:
            threshold_reference_mutations(variant, threshold)
    parse_reference_mutations(thresholds=thresholds, variant_order=['jean_total'], jean_toggle='weighted')
    #gen_thresholded_mut_counts_for_analysis(thresholds)
    #global_avg_subset_mat, global_naive_subset_mat, global_blind_subset_mat = read_thresholded_global_mat('5e-05', variant_order_aggregate) #pull mutation matrices
    #append_neighboring_nucs_to_collapsed_mutation_list(['jean_total'])'''
    #use this one
    #convert_mutation_dataset()
    #checking mutability of each gene/variant
    #`analyze_genes()

if __name__ == '__main__':
    main()