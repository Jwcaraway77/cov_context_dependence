import numpy as np
import pandas as pd
import torch, os
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec, cm, ticker
import seaborn as sns
from scipy import stats
import functools
import time
import re
#import scikit-allel as allel

#print(os.path.exists('C:\\Users\\John\\anaconda3\\envs\\cmdap_env\\lib\\R\\bin\\x64\\R.dll'))
#os.environ['R_HOME'] = 'C:\\Users\\John\\anaconda3\\envs\\cmdap_env\\lib\\R'
#import rpy2.robjects as robjects

rows = ["T[X>Y]T","T[X>Y]G","T[X>Y]C","T[X>Y]A","G[X>Y]T","G[X>Y]G","G[X>Y]C","G[X>Y]A","C[X>Y]T","C[X>Y]G","C[X>Y]C","C[X>Y]A"]
rows_figs = ["U[X>Y]U","U[X>Y]G","U[X>Y]C","U[X>Y]A","G[X>Y]U","G[X>Y]G","G[X>Y]C","G[X>Y]A","C[X>Y]U","C[X>Y]G","C[X>Y]C","C[X>Y]A","A[X>Y]U","A[X>Y]G","A[X>Y]C","A[X>Y]A"]
columns = ["T>G","T>C","T>A","G>T","G>C","G>A","C>T","C>G","C>A","A>T","A>G","A>C"]
columns_figs = ["U>G","U>C","U>A","G>U","G>C","G>A","C>U","C>G","C>A","A>U","A>G","A>C"]
columns_shortened = ['T','G','C','A']
columns_shortened_figs = ['U','G','C','A']

'''read in reference fasta'''
def get_fasta():
    with open('EPI_ISL_402124.fasta', 'r') as f:
        lines = f.readlines()
    return lines[1]

'''save fasta after simulation run'''
def save_fasta(fasta_results, sim_run, num_muts, num_iters, sim_type, variant, contexts, threshold):
    '''if contexts:
        path = 'simulation_output/'+sim_type+'/full_contexts/' + variant + '/fastas/'+str(threshold)+'/simulated_fasta_' + str(sim_run)+'.fasta'
    else:
        path = 'simulation_output/'+sim_type+'/naive_contexts/' + variant + '/fastas/'+str(threshold)+'/simulated_fasta_' + str(sim_run)+'.fasta'
    '''
    path = 'simulation_output/'+sim_type+'/'+contexts+'/' + variant + '/fastas/'+str(threshold)+'/simulated_fasta_' + str(sim_run)+'.fasta'
    with open(path, 'w') as f:
        f.write('>simulation_fasta_'+str(sim_run)+'_with_'+str(num_muts)+'_mutation(s)_completed_in_'+str(num_iters)+'_iterations\n')
        f.write(fasta_results)

'''save mutation dictionary after mutation run'''
def save_mutation_dict(mutation_dict_results, sim_run, sim_type, variant, contexts, threshold):
    meta_data = str(mutation_dict_results[99999999])
    del mutation_dict_results[99999999]
    mut_dict = pd.DataFrame.from_dict(mutation_dict_results, orient='index')
    mut_dict.columns = ['original','position','mutation']
    mut_dict.loc[99999999,'original'] = meta_data
    mut_dict.to_csv('simulation_output/'+sim_type+'/'+contexts+'/'+variant+'/mut_dicts/'+str(threshold)+'/mut_dicts_'+str(sim_run)+'.csv')
    
#read global mat for a specific threshold
def read_thresholded_global_mat(threshold, variant_order, rate_type='re-calced', jf_flag=0):
    #rate_type determines if naive and blind mats are averaged full-rates or re-calced from mut counts and triplet counts
    global_avg_subset_mat = []
    global_naive_subset_mat = []
    global_blind_subset_mat = []
    for variant in variant_order: #loop through variants
        var_folder = [var_folder for var_folder in os.listdir('sim_ref_data') if '('+variant+')' in var_folder and 'full_clade' in var_folder][0]
        if jf_flag == 1:
            global_avg_subset_mat.append(torch.tensor(pd.read_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/both_mut_rate_mat.csv', index_col=0, header=0).values[:12,:12]))
            global_naive_subset_mat.append(torch.tensor(pd.read_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/both_tstv_mut_rate_mat.csv', index_col=0, header=0).to_numpy()))
            global_blind_subset_mat.append(torch.tensor(pd.read_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/both_naive_mut_rate_mat.csv', index_col=0, header=0).to_numpy()))
        else:
            global_avg_subset_mat.append(torch.tensor(pd.read_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/'+str(threshold)+'_mat.csv', index_col=0, header=0).values))
    global_avg_subset_mat = torch.stack(global_avg_subset_mat) #stack variant mats into tensor
    print(global_avg_subset_mat.shape)
    if jf_flag == 1:
        global_naive_subset_mat = torch.stack(global_naive_subset_mat)
        global_blind_subset_mat = torch.stack(global_blind_subset_mat)
        return global_avg_subset_mat, global_naive_subset_mat, global_blind_subset_mat
    
    if rate_type == 'averaged':
        #naive mat
        global_naive_subset_mat = torch.mean(global_avg_subset_mat, dim=(1))
        #blind mat
        global_blind_subset_mat = torch.stack([torch.mean(global_avg_subset_mat[:,:,i:i+3], dim=(1,2)) for i in range(0,12,3)]).T
    elif rate_type == 're-calced':
        #read in triplet counts
        triplet_counts = np.repeat(pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/total.csv', index_col=0, header=0).to_numpy(), 3).reshape(12,12)
        full_muts = global_avg_subset_mat * triplet_counts #un-normalize mutation counts
        #naive_mat
        naive_muts = torch.sum(full_muts, dim=(1)) #sum muts across rows
        naive_triplets = np.sum(triplet_counts, axis=0) #sum triplets across rows
        global_naive_subset_mat = naive_muts / naive_triplets #calc naive `rates`
        #blind_mat
        blind_muts = torch.stack([torch.sum(full_muts[:,:,i:i+3], dim=(1,2)) for i in range(0,12,3)]).T #sum muts across rows and T,G,C,A columns
        blind_triplets = np.array([np.sum(triplet_counts[:,i:i+3]) for i in range(0,12,3)]) #sum triplets across rows and T,G,C,A columns
        global_blind_subset_mat = blind_muts / blind_triplets #calc blind `rates`

    return global_avg_subset_mat, global_naive_subset_mat, global_blind_subset_mat

'''compare reference and mutated fastas'''
def compare_fastas_2(fasta_1, fasta_2, path):
    fasta_2 = pd.concat([fasta_1[:len(fasta_2)], pd.Series([*fasta_2], name='sim_fasta')], axis=1)
    fasta_2.columns=['mutated_fasta', 'sim_fasta']
    comp = ''
    diff = 0
    '''loop through genome'''
    for index in fasta_2.index:
        #check that simulated fasta has info at index
        if fasta_2.loc[index, 'sim_fasta'] != np.nan:
            #check if simulated nucleotide matches either original or known mutation in reference
            if fasta_2.loc[index, 'sim_fasta'] in fasta_2.loc[index, 'mutated_fasta']:
                #match
                comp += '*'
            else:
                #difference
                comp += fasta_2.loc[index, 'sim_fasta']
                diff+=1
    #save comparison fasta
    with open(path+'_comp_fasta.txt', 'w') as f:
        f.write('len fasta_1: '+ str(fasta_1.shape[0])+ ' \n')
        f.write('len fasta_2: '+ str(fasta_2.shape[0])+ ' \n')
        f.write('total differences: '+ str(diff)+ ' \n')
        f.write(comp)
        #print(path+'comp_fasta.txt')

#further analyzing context counts for genes
def gene_context_count_analysis():
    output_df = np.array(np.zeros([8,4]))
    genes = []
    nucs = columns_shortened
    for file in [file for file in os.listdir('simulation_output/context_counts/genes') if 'correlations' not in file]:
        genes.append(file.split('_')[0])
        mat = pd.read_csv('simulation_output/context_counts/genes/'+file, index_col=0, header=0)
        mat.index = pd.MultiIndex.from_product([nucs,nucs])

        gene_array = np.array([])
        for nuc_2 in nucs:
            nuc_array = np.array([])
            #C->X analysis
            for nuc_1 in nucs:
                #print(mat.loc[(nuc_1,nucs),'C'])
                #print(mat.loc[(nuc_1,nucs),nucs])
                nuc_array = np.append(nuc_array, np.sum(mat.loc[(nuc_1,nucs),nuc_2].to_numpy())/np.sum(mat.loc[(nuc_1,nucs),nucs].to_numpy()))
            #print(gene_array)
            #N[C]X analysis
            for nuc_3 in nucs:
                #print(mat.loc[(nucs,nuc_3),'C'])
                #print(mat.loc[(nucs,nuc_3),nucs])
                nuc_array = np.append(nuc_array, np.sum(mat.loc[(nucs,nuc_3),nuc_2].to_numpy())/np.sum(mat.loc[(nucs,nuc_3),nucs].to_numpy()))
            gene_array = np.append(gene_array, nuc_array)
            #gene_array = gene_array.reshape(-1,8).T
            #print(gene_array)
        output_df = np.append(output_df, gene_array.reshape(-1,8).T, axis=1)
    print(output_df.shape)
    output_df = pd.DataFrame(output_df[:,4:], index=nucs+['N[X]'+nuc for nuc in nucs], columns=pd.MultiIndex.from_product([genes,nucs]))
    output_df.to_csv('simulation_output/gene_contexts_analysis.csv')

#chi-square test for each variant between expected muts >=.5 and observed
def expected_vs_obs(subset_genes):
    '''#read in expected mutations
    expected_df = pd.read_excel('simulation_output/final_info/muts_and_time_plots.xlsx', sheet_name='expected_muts', skiprows=1, nrows=20)
    exp_df = expected_df.iloc[4:20, 3:38]
    exp_df.index = expected_df.iloc[4:20,1]
    exp_df.columns = expected_df.iloc[0, 3:38]
    exp_df.drop(['aggregate'], inplace=True)
    print(exp_df)
    del expected_df
    

    #read in observed mutations
    observed_df = pd.read_excel('simulation_output/final_info/muts_and_time_plots.xlsx', sheet_name='expected_muts', skiprows=22, nrows=16)
    obs_df = observed_df.iloc[1:, 3:38]
    obs_df.index = observed_df.iloc[1:,2]
    obs_df.columns = observed_df.iloc[0, 3:38]
    obs_df.drop(['aggregate'], inplace=True)
    print(obs_df)
    del observed_df
    '''
    triplet_counts = pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/total.csv', index_col=0, header=0)
    fasta = pd.Series([*get_fasta()[:-1]], name='ref')

    #now generating expected mutations from multiplying normalized variant mut matrix with gene 4fold triplet counts
    '''exp_df = pd.DataFrame(np.zeros(obs_df.shape), index=obs_df.index, columns=obs_df.columns)
    for gene in exp_df.columns:
        #gene_df = pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/'+gene+'.csv', index_col=0, header=0)
        if len(subset_genes[gene]) > 2:
            positions = []
            for sub_positions in subset_genes[gene]:
                [positions.append(val) for val in range(sub_positions[0],sub_positions[1])]
        else:
            positions = range(subset_genes[gene][0],subset_genes[gene][1])
        print(gene, positions)
        gene_df = []
        for sim in os.listdir('simulation_output/global/full_contexts/aggregate/mut_dicts/5e-05'):
            mut_dict = pd.read_csv('simulation_output/global/full_contexts/aggregate/mut_dicts/5e-05/'+sim, index_col=0, header=0)
            mut_dict.drop(99999999, inplace=True)
            vals = [mut_dict.loc[index].to_numpy()[0].split('|') for index in mut_dict.index]
            mut_dict['position'] = [int(val[1]) for val in vals]
            mut_dict['old'] = [val[0][1] for val in vals]
            mut_dict['mut'] = [val[2][1] for val in vals]
            mut_dict['triplet'] = [val[0] for val in vals]
            mut_dict = mut_dict.loc[mut_dict['position'].isin(positions)]
            
            triplet_counts = pd.DataFrame(np.zeros([12,12]), index=rows, columns=columns)
            #loop through mutations
            for index in mut_dict.index:
                if mut_dict.loc[index,'triplet'][0] != 'A':
                    triplet_counts.loc[mut_dict.loc[index,'triplet'][0]+'[X>Y]'+mut_dict.loc[index,'triplet'][-1], mut_dict.loc[index,'old']+'>'+mut_dict.loc[index,'mut']] += 1
            gene_df.append(triplet_counts.to_numpy())
        gene_df = np.mean(gene_df, axis=0).flatten()
        #gene_df = np.repeat(np.mean(gene_df, axis=0), [3])
        #gene_df = np.repeat(gene_df.to_numpy()[:12,:], [3])
        mats = []
        for variant in obs_df.index:
            var_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder][0]
            mut_df = pd.read_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/5e-05_mat.csv', index_col=0, header=0).to_numpy().flatten()
            #tried normalizing indiviually and as a group
            exp_df.loc[variant, gene] = np.sum(gene_df*mut_df)
    print(exp_df)
    exp_df.to_csv('simulation_output/final_info/expected_muts.csv')'''

    #expected obs are now taken from simulation (currently using 'all' to simulate)
    '''exp_df = pd.DataFrame(np.zeros(obs_df.shape), index=obs_df.index, columns=obs_df.columns)
    for gene in exp_df.columns:
        gene_hist = pd.read_csv('simulation_output/global/analysis/genes/full_contexts/'+gene+'_hist.csv', index_col=0, header=0)
        exp_df.loc[:,gene] = np.repeat(np.sum(gene_hist.loc[:,['T','G','C','A']].to_numpy()), exp_df.shape[0])
    exp_df.to_csv('simulation_output/final_info/expected_muts_2.csv')'''

    #now generating expected muts by taking 5e-05 and .5 mut count and multiplying by size of gene
    exp_dfs = {}
    obs_dfs = {}
    mut_counts_list = []
    gene_sizes_list = []
    testing_thresholds = ['5e-05','0.5']
    for threshold in testing_thresholds:
        #read in observed muts
        obs_df = pd.read_csv('simulation_output/high_freq_mut_dist_'+str(threshold)+'.csv', index_col=0, header=0)
        obs_dfs[threshold] = obs_df
        exp_df = pd.DataFrame(np.zeros(obs_df.shape), index=obs_df.index, columns=obs_df.columns)
        gene_sizes = {}
        for gene in exp_df.columns:
            gene_bounds = subset_genes[gene]
            if len(gene_bounds) > 2:
                gene_size = 0
                for sub_region in gene_bounds:
                    gene_size += sub_region[1]-sub_region[0]
                gene_sizes[gene] = gene_size
            else:
                gene_sizes[gene] = gene_bounds[1]-gene_bounds[0]
            print(gene, gene_sizes[gene])
        gene_sizes_list.append(gene_sizes)
        mut_counts = {}
        for variant in exp_df.index:
            var_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder and 'clade' in folder and 'jean' not in folder][0]
            mut_count = pd.read_csv('sim_ref_data/'+var_folder+'/reference_mutations/'+threshold+'_reference_mutations.csv', index_col=0, header=0).shape[0]
            mut_counts[variant] = mut_count
            print(variant, mut_count)
            for gene in exp_df.columns:
                exp_df.loc[variant,gene] = mut_count * gene_sizes[gene] / 29903
        mut_counts_list.append(mut_counts)
        exp_df.to_csv('simulation_output/final_info/expected_muts_'+threshold+'.csv')
        exp_dfs[threshold] = exp_df
        


    #create output_mat for comparing variant obs vs variant exp
    '''output_df = pd.DataFrame(np.zeros([2,exp_df.shape[0]]), index=['chi-squared', 'p-value'], columns=exp_df.index)

    #loop through variants
    for var in exp_df.index:
        chi_res = stats.chisquare(f_obs=obs_df.loc[var,:], f_exp=exp_df.loc[var,:])
        output_df.loc['chi-squared', var] = chi_res[0]
        output_df.loc['p-value', var] = chi_res[1]
    
    #save output df
    output_df.to_csv('simulation_output/final_info/exp_vs_obs_variants.csv')'''

    #create output_mat for comparing gene obs vs gene exp
    '''want:
        0.5 chi-square where values > 10
        5e-05 ks-test 
    '''
    output_df = pd.DataFrame(np.zeros([10,exp_df.shape[1]*2]), index=['chi-squared', 'p-value', 't-stat', 'p_value', 'ks-stat', 'p', 'less-ks-stat','less-p','greater-ks-stat','greater-p'], columns=pd.MultiIndex.from_product([testing_thresholds,exp_df.columns]))

    #loop through thresholds
    for threshold in testing_thresholds:
        exp_df = exp_dfs[threshold]
        obs_df = obs_dfs[threshold]
        #loop through genes
        for gene in exp_df.columns:
            #try:
            #chi_res = stats.chisquare(f_obs=obs_df.loc[:,gene], f_exp=exp_df.loc[:,gene], sum_check=False)
            #chi_res = robjects.chisq(obs_df.loc[:,gene], f_exp=exp_df.loc[:,gene])
            #print(chi_res)
            #output_df.loc['chi-squared', (threshold,gene)] = chi_res[0]
            #output_df.loc['p-value', (threshold,gene)] = chi_res[1]
            output_df.loc[['chi-squared','p-value'], (threshold,gene)] = stats.chisquare(f_obs=obs_df.loc[:,gene], f_exp=exp_df.loc[:,gene], sum_check=False)
            '''except:
                output_df.loc['chi-squared', (threshold,gene)] = np.nan
                output_df.loc['p-value', (threshold,gene)] = np.nan'''
            obs_var = stats.variation(obs_df.loc[:,gene])
            exp_var = stats.variation(exp_df.loc[:,gene])
            if obs_var >= exp_var*.95 or obs_var <= exp_var*1.05:
                ttest_res = stats.ttest_ind(obs_df.loc[:,gene], exp_df.loc[:,gene], equal_var=True)
            else:
                ttest_res = stats.ttest_ind(obs_df.loc[:,gene], exp_df.loc[:,gene], equal_var=False)
            output_df.loc['t-stat',(threshold,gene)] = ttest_res[0]
            output_df.loc['p_value',(threshold,gene)] = ttest_res[1]
            output_df.loc[['ks-stat','p'],(threshold,gene)] = stats.kstest(rvs=obs_df.loc[:,gene], cdf=exp_df.loc[:,gene])
            output_df.loc[['less-ks-stat','less-p'],(threshold,gene)] = stats.kstest(rvs=obs_df.loc[:,gene], cdf=exp_df.loc[:,gene], alternative='less')
            output_df.loc[['greater-ks-stat','greater-p'],(threshold,gene)] = stats.kstest(rvs=obs_df.loc[:,gene], cdf=exp_df.loc[:,gene], alternative='greater')
    
    #binomial test
    binom_df = pd.DataFrame(np.zeros([2*len(mut_counts),exp_df.shape[1]*2]), index=pd.MultiIndex.from_product([[key for key in mut_counts.keys()],['binom','binom_p']]), columns=pd.MultiIndex.from_product([testing_thresholds,exp_df.columns]))
    for threshold_index, threshold in enumerate(testing_thresholds):
        for variant, mut_count in mut_counts_list[threshold_index].items():
            for gene in exp_df.columns:
                if threshold == '5e-05':
                    print(f'{variant}, {gene},  k={obs_dfs[threshold].loc[variant,gene]}, n={mut_count}, p={gene_sizes_list[threshold_index][gene]}/{29903}')
                binom_result = stats.binomtest(k=int(np.round(obs_dfs[threshold].loc[variant,gene])), n=int(np.round(mut_count)), p=gene_sizes_list[threshold_index][gene]/29903)
                binom_df.loc[variant,(threshold,gene)] = [binom_result.proportion_estimate,binom_result.pvalue]
    #save output df
    #output_df.to_csv('simulation_output/final_info/exp_vs_obs_genes_2.csv')
    binom_df.to_csv('simulation_output/final_info/exp_vs_obs_genes_binom_tests.csv')

#look at 5' and 3' influence on 'rates'
def analyze_contextual_influence(variant, threshold):
    five_prime_mat = np.zeros([4,12])
    three_prime_mat = np.zeros([3,12])
    stats_df = pd.DataFrame(np.zeros([4,len(columns_figs)]), index=pd.MultiIndex.from_product([['3_prime','5_prime'],['x^2','p']]), columns=columns_figs)
    tpm_labels = ['U[X>Y]N','G[X>Y]N','C[X>Y]N']
    fpm_labels = ['N[X>Y]U','N[X>Y]G','N[X>Y]C','N[X>Y]A']
    expected_muts_df = pd.DataFrame(np.zeros([7,len(columns_figs)]), index=tpm_labels+fpm_labels, columns=columns_figs)

    if not os.path.exists('simulation_output/final_info/contextual_influence'):
        os.mkdir('simulation_output/final_info/contextual_influence')
    
    #read in mut matrix for threshold
    folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder][0]
    var_mat = pd.read_csv('sim_ref_data/'+folder+'/thresholded_mutations/'+threshold+'_mat.csv', index_col=0, header=0)

    #recalc rates
    triplet_counts = pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/total.csv', index_col=0, header=0) #read in triplet counts
    triplet_counts = np.repeat(triplet_counts.to_numpy(), 3).reshape(12,12) #convert to 12x12 matrix
    print(var_mat.shape, triplet_counts.shape)

    var_mat = var_mat.to_numpy() * triplet_counts #convert rates into counts for each context


    #N3' collapse
    #gives (3,12) shape
    #which should be U[X>Y]N, G[X>Y]N, C[X>Y]N
    #three_prime_mat = np.array([np.mean(var_mat.iloc[i:i+3,:].to_numpy(), axis=0) for i in range(0,11,4)])
    three_prime_mat = np.array([np.sum(var_mat[i:i+4,:], axis=0) for i in range(0,11,4)])
    three_prime_triplets = np.array([np.sum(triplet_counts[i:i+4,:], axis=0) for i in range(0,11,4)])
    pd.DataFrame(three_prime_mat, index=tpm_labels, columns=columns_figs).to_csv('simulation_output/final_info/contextual_influence/tpm_counts.csv')
    pd.DataFrame(three_prime_triplets, index=tpm_labels, columns=columns_figs).to_csv('simulation_output/final_info/contextual_influence/tpm_triplets.csv')
    tpm_binom_df = pd.DataFrame(np.zeros([2*len(tpm_labels),len(columns_figs)]), index=pd.MultiIndex.from_product([['exp','p'],tpm_labels]), columns=columns_figs)
    for col in range(three_prime_mat.shape[1]):
        #exp_col = np.repeat(np.sum(three_prime_mat[:,col]) / 4, 3) / three_prime_triplets[:,col]
        print(three_prime_mat[:,col], three_prime_triplets[:,col])
        
        #sum mutation count and triplet count
        mut_sum = np.sum(three_prime_mat[:,col])
        triplet_sum = np.sum(three_prime_triplets[:,col])

        #distribute x mutations to each row in col, naively assuming each context is equally likely
        exp_muts = mut_sum * (three_prime_triplets[:,col]/triplet_sum) #trying first without rounding
        expected_muts_df.loc[tpm_labels,columns_figs[col]] = exp_muts

        #chisquare between observed muts and expected muts
        chi_square_result = stats.chisquare(three_prime_mat[:,col],exp_muts)
        
        stats_df.loc[('3_prime',['x^2','p']),columns_figs[col]] = chi_square_result
        
        #binomial test
        #test if numer of mutations in [row,col] is different than placing mut_sum number of mutations across column based on triplet weighting
        for row in range(three_prime_mat.shape[0]):
            binom_result = stats.binomtest(k=int(np.round(three_prime_mat[row,col])), n=int(np.round(mut_sum)), p=three_prime_triplets[row,col]/triplet_sum)
            tpm_binom_df.loc[(['exp','p'],tpm_labels[row]),columns_figs[col]] = [binom_result.proportion_estimate,binom_result.pvalue]
        
    three_prime_mat = three_prime_mat / three_prime_triplets
    #print(three_prime_triplets)

    #N5' collapse
    #gives (4,12) shape
    #which should be N[X>Y]U, N[X>Y]G, N[X>Y]C, N[X>Y]A
    indices = [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
    #five_prime_mat = np.array([np.mean(var_mat.iloc[index,:].to_numpy(), axis=0) for index in indices])
    five_prime_mat = np.array([np.sum(var_mat[index,:], axis=0) for index in indices])
    five_prime_triplets = np.array([np.sum(triplet_counts[index,:], axis=0) for index in indices])
    pd.DataFrame(five_prime_mat, index=fpm_labels, columns=columns_figs).to_csv('simulation_output/final_info/contextual_influence/fpm_counts.csv')
    pd.DataFrame(five_prime_triplets, index=fpm_labels, columns=columns_figs).to_csv('simulation_output/final_info/contextual_influence/fpm_triplets.csv')
    fpm_binom_df = pd.DataFrame(np.zeros([2*len(fpm_labels),len(columns_figs)]), index=pd.MultiIndex.from_product([['exp','p'],fpm_labels]), columns=columns_figs)

    for col in range(five_prime_mat.shape[1]):
        #exp_col = np.repeat(np.sum(five_prime_mat[:,col]) / 3, 4) / five_prime_triplets[:,col]
        print(five_prime_mat[:,col], five_prime_triplets[:,col])
        
        #sum mutation count and triplet count
        mut_sum = np.sum(five_prime_mat[:,col])
        triplet_sum = np.sum(five_prime_triplets[:,col])

        #distribute x mutations to each row in col, naively assuming each context is equally likely
        exp_muts = mut_sum * (five_prime_triplets[:,col]/triplet_sum) #trying first without rounding
        expected_muts_df.loc[fpm_labels,columns_figs[col]] = exp_muts

        #chisquare between observed muts and expected muts
        chi_square_result = stats.chisquare(five_prime_mat[:,col],exp_muts)
        
        stats_df.loc[('5_prime',['x^2','p']),columns_figs[col]] = chi_square_result
    
        #binomial test
        #test if numer of mutations in [row,col] is different than placing mut_sum number of mutations across column based on triplet weighting
        for row in range(five_prime_mat.shape[0]):
            binom_result = stats.binomtest(k=int(np.round(five_prime_mat[row,col])), n=int(np.round(mut_sum)), p=five_prime_triplets[row,col]/triplet_sum)
            fpm_binom_df.loc[(['exp','p'],fpm_labels[row]),columns_figs[col]] = [binom_result.proportion_estimate,binom_result.pvalue]
    five_prime_mat = five_prime_mat / five_prime_triplets
    #print(five_prime_triplets)
    

    #plot results
    fig = plt.figure(layout='constrained', dpi=200, figsize=(16,10))
    grid = gridspec.GridSpec(1,3, figure=fig, width_ratios=[.95,.02,.03])
    main_grid = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=grid[0,0], height_ratios=[.4,.05,.55])
    three_prime_ax = fig.add_subplot(main_grid[0,0])
    five_prime_ax = fig.add_subplot(main_grid[2,0])
    cbar_ax = fig.add_subplot(grid[0,2])
    fontsize_1 = 22
    fontsize_2 = 28
    norm = colors.Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.Greys), cax=cbar_ax, orientation='vertical')
    cbar.set_ticklabels([i/10 for i in range(0,11,2)], fontsize=20)
    convert_val = lambda x: str(int(x)) if x in [0,1] else str(np.round(x,2))
    tpm_annot = [[convert_val(three_prime_mat[index,col]) for col in range(three_prime_mat.shape[1])] for index in range(three_prime_mat.shape[0])]
    fpm_annot = [[convert_val(five_prime_mat[index,col]) for col in range(five_prime_mat.shape[1])] for index in range(five_prime_mat.shape[0])]
    sns.heatmap(three_prime_mat, ax=three_prime_ax, xticklabels=False, cbar=False, linewidth=2, linecolor='grey', cmap='Greys', annot=tpm_annot, fmt='s', annot_kws={'fontsize':fontsize_1})
    sns.heatmap(five_prime_mat, ax=five_prime_ax, cbar=False, linewidth=2, linecolor='grey', cmap='Greys', annot=fpm_annot, fmt='s', annot_kws={'fontsize':fontsize_1})
    #three_prime_ax.set_xticklabels(columns_figs, fontsize=20)
    three_prime_ax.set_yticklabels(tpm_labels, rotation='horizontal', fontsize=fontsize_1)
    three_prime_ax.set_title("Effects of 5' Nucleotides", fontsize=fontsize_2)
    five_prime_ax.set_xticklabels(columns_figs, fontsize=fontsize_1)
    five_prime_ax.set_yticklabels(fpm_labels, rotation='horizontal', fontsize=fontsize_1)
    five_prime_ax.set_title("Effects of 3' Nucleotides", fontsize=fontsize_2)
    plt.savefig('simulation_output/final_info/final_figs/fig_2_contextual_influence.png')
    plt.close()

    #stats testing
    #want to look at null expectation of columns being evenly distributed
    #output_df = pd.DataFrame()
    #for col in range(three_prime_mat.shape[1]):
    #    exp_col = np.repeat(np.sum(var_mat[:,col])/np.sum(triplet_counts[:,col]), 3)
    #    print(stats.chisquare(three_prime_mat[:,col], exp_col))
    stats_df.to_csv('simulation_output/final_info/contextual_influence/contextual_influence_chi_square_tests.csv')
    pd.concat([pd.DataFrame(three_prime_mat,index=tpm_labels,columns=columns_figs), pd.DataFrame(five_prime_mat, index=fpm_labels, columns=columns_figs)], axis=0).to_csv('simulation_output/final_info/contextual_influence/contextual_influence_fig_mat.csv')
    expected_muts_df.to_csv('simulation_output/final_info/contextual_influence/contextual_influence_expected_muts.csv')
    tpm_binom_df.to_csv('simulation_output/final_info/contextual_influence/tpm_binom_tests.csv')
    fpm_binom_df.to_csv('simulation_output/final_info/contextual_influence/fpm_binom_tests.csv')

#calculate context counts of meaningful nucleotides (results in amino acid change)
def meaningful_triplet_counts(gene_dict):
    codon_map = {'TTT':[3,3,2],'TTG':[2,3,2],'TTC':[3,3,2],'TTA':[2,3,2], 'TGT':[3,3,2],'TGG':[3,3,3],'TGC':[3,3,2],'TGA':[3,2,3], 'TCT':[3,3,0],'TCG':[3,3,0],'TCC':[3,3,0],'TCA':[3,3,0], 'TAT':[3,3,2],'TAG':[3,3,2],'TAC':[3,3,2],'TAA':[3,2,2],
            'GTT':[3,3,0],'GTG':[3,3,0],'GTC':[3,3,0],'GTA':[3,3,0], 'GGT':[3,3,0],'GGG':[3,3,0],'GGC':[3,3,0],'GGA':[3,3,0], 'GCT':[3,3,0],'GCG':[3,3,0],'GCC':[3,3,0],'GCA':[3,3,0], 'GAT':[3,3,2],'GAG':[3,3,2],'GAC':[3,3,2],'GAA':[3,3,2],
            'CTT':[3,3,0],'CTG':[2,3,0],'CTC':[3,3,0],'CTA':[2,3,0], 'CGT':[3,3,0],'CGG':[2,3,0],'CGC':[3,3,0],'CGA':[2,3,0], 'CCT':[3,3,0],'CCG':[3,3,0],'CCC':[3,3,0],'CCA':[3,3,0], 'CAT':[3,3,2],'CAG':[3,3,2],'CAC':[3,3,2],'CAA':[3,3,2],
            'ATT':[3,3,1],'ATG':[3,3,3],'ATC':[3,3,1],'ATA':[3,3,1], 'AGT':[3,3,2],'AGG':[2,3,2],'AGC':[3,3,2],'AGA':[2,3,2], 'ACT':[3,3,0],'ACG':[3,3,0],'ACC':[3,3,0],'ACA':[3,3,0], 'AAT':[3,3,2],'AAG':[3,3,2],'AAC':[3,3,2],'AAA':[3,3,2]}
    empty_df = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=['T','G','C','A'])
    fasta = pd.Series([*get_fasta()])
    context_counts = {}
    if not os.path.exists('simulation_output/context_counts/genes_meaningful'):
        os.mkdir('simulation_output/context_counts/genes_meaningful')
    if not os.path.exists('simulation_output/context_counts/triplet_counts'):
        os.mkdir('simulation_output/context_counts/triplet_counts')
    for gene, gene_positions in gene_dict.items():
        print(gene, gene_positions)
        seq = fasta[gene_positions[0]-1:gene_positions[1]+1]
        triplet_counts = empty_df.copy()
        meaningful_contexts = empty_df.copy()

        #normal triplet count
        for i in range(1,len(seq)-1,3):
            triplet = ''.join(seq[i:i+3])
            print(len(seq))
            print(triplet)
            if len(triplet) == 3:
                triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] += 1

        #all context count
        #for i in range(1, len(seq)-1):
        #    triplet = seq[i-1:i+2]
        #    if len(triplet) == 3: #shouldnt be necessary
        #        all_contexts.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] += 1

        #meaningful context count
        #each triplet has a maximum of 9 mutations, of which x are degenerate so 9-x / 9
        for triplet_index in range(1,len(seq)-3,3): #loop through each triplet
            triplet = seq[triplet_index-1:triplet_index+4].to_numpy() #including nucs upstream and downstream of triplet
            #print(triplet)
            #pulling TGCAT
            #so contexts are TGC, GCA, CAT
            #and weighted by Gca, gCa, gcA
            for nuc_index in range(1,4):
                #1=TGC, 2=GCA, 3=CAT
                nuc_triplet = triplet[nuc_index-1:nuc_index+2]
                #print(nuc_index, nuc_triplet)
                #print(nuc_triplet[0]+'[X>Y]'+nuc_triplet[2], nuc_triplet[1])
                #print(codon_map[''.join(triplet[1:-1])][nuc_index-1])
                meaningful_contexts.loc[nuc_triplet[0]+'[X>Y]'+nuc_triplet[2], nuc_triplet[1]] += codon_map[''.join(triplet[1:-1])][nuc_index-1]
        meaningful_contexts = meaningful_contexts / 3
                

        triplet_counts.to_csv('simulation_output/context_counts/triplet_counts/'+gene+'_triplet_counts.csv')
        #all_contexts.to_csv('simulation_output/'+vaccine_file.split('_')[0]+'_all_contexts.csv')
        meaningful_contexts.to_csv('simulation_output/context_counts/genes_meaningful/'+gene+'_meaningful_contexts.csv')
        #context_counts[vaccine_file.split('_')[0]] = all_contexts
        #context_counts[vaccine_file.split('_')[0]] = meaningful_contexts
        #print(all_contexts)
    #return context_counts

#correlate aggregate 5e-05 obs with meaningful contexts for each gene
def correlate_all_meaningful_contexts():
    aggregate_mat = pd.read_csv('sim_ref_data/0(aggregate)_full_clade/thresholded_mutations/5e-05_mat.csv', index_col=0, header=0)
    context_mats = {file.split('_')[0]:pd.read_csv('simulation_output/context_counts/genes_meaningful/'+file, index_col=0, header=0) for file in os.listdir('simulation_output/context_counts/genes_meaningful')}
    triplet_counts = pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/total.csv', index_col=0, header=0)
    #triplet_counts = {file.split('_')[0]:pd.read_csv('simulation_output/context_counts/triplet_counts/'+file, index_col=0, header=0) for file in os.listdir('simulation_output/context_counts/triplet_counts')}
    output_df = pd.DataFrame(np.zeros([4,len(context_mats.keys())]), index=pd.MultiIndex.from_product([['(12,12)','(12,4)'],['corr', 'p']]), columns=list(context_mats.keys()))
    #there are 2 ways to correlate, 12x12 or 12x4
    for gene, context_mat in context_mats.items():
        #12x12 expands meaningful context counts
        context_mat = context_mat.iloc[:12,:12]
        context_mat_expanded = pd.DataFrame(np.zeros([12,12]), index=aggregate_mat.index, columns=aggregate_mat.columns)
        for i in range(0,12):
            context_mat_expanded.iloc[:,i] = context_mat.iloc[:,int(np.floor(i/3))]
        corr = stats.pearsonr(aggregate_mat.to_numpy().flatten(), context_mat_expanded.to_numpy().flatten())
        output_df.loc[('(12,12)','corr'),gene] = corr[0]
        output_df.loc[('(12,12)','p'),gene] = corr[1]

        #12x4 shrinks aggregate mat
        aggregate_mat_shrunk = aggregate_mat.copy()
        #un-normalize the columns by triplet count
        for col in aggregate_mat_shrunk.columns:
            #print(aggregate_mat_shrunk.loc[:,col])
            #print(triplet_counts.loc[:,col[0]])
            aggregate_mat_shrunk.loc[:,col] *= triplet_counts.loc[:,col[0]]
        #collapse columns from T->G, T->G, T->A to T
        for i in range(0,12,3):
            aggregate_mat_shrunk.iloc[:,i] = aggregate_mat_shrunk.iloc[:,i:i+3].sum(axis=1)
        aggregate_mat_shrunk = aggregate_mat_shrunk.iloc[:,[0,3,6,9]]
        #renormalize by gwtc
        aggregate_mat_shrunk.columns = triplet_counts.columns
        aggregate_mat_shrunk /= triplet_counts
        if gene == 'S':
            print(gene)
            print(aggregate_mat_shrunk)
            print(context_mat)
        corr = stats.pearsonr(aggregate_mat_shrunk.to_numpy().flatten(), context_mat.to_numpy().flatten())
        output_df.loc[('(12,4)','corr'),gene] = corr[0]
        output_df.loc[('(12,4)','p'),gene] = corr[1]
    output_df.loc[:,['S','E','M','N','ORF1ab','ORF1a','ORF3a','ORF3b','ORF6','ORF7a','ORF7b','ORF8','ORF9b','ORF9c','ORF10','nsp1','nsp2','nsp3','nsp4','nsp5','nsp6','nsp7','nsp8','nsp9','nsp10','nsp11','rdrp','nsp13','nsp14','nsp15','nsp16','NTD','RBD','SD1']].to_csv('simulation_output/final_info/meaningful_correlations.csv')

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

#threshold reference mutations for comparison
#updated to use collapsed mutation list instead (saves time)
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
        subset_df = subset_df.loc[:,['position', mut+'_freq', 'original']]
        subset_df = pd.concat([subset_df, pd.Series(np.array([mut]*subset_df.shape[0]), index=subset_df.index)], axis=1)
        subset_df.columns = ['position','proportion','old','mut']
        #print(subset_df.shape)
        reference_mutations = pd.concat([reference_mutations, subset_df], axis=0)
    #print(reference_mutations.shape)

    #convert datatype
    #print(reference_mutations['position'])
    reference_mutations['position'] = np.asarray(reference_mutations['position'].values, dtype=int)
    reference_mutations.sort_values(by='position', inplace=True)
    reference_mutations.reset_index(drop=True, inplace=True)
    reference_mutations.to_csv('sim_ref_data/'+reference_folder+'/reference_mutations/'+str(threshold)+'_reference_mutations.csv')

'''def convert_to_aa(x, aa_codes, aa_abbrevs, nuc_order):
    output = []
    for i in range(len(x)):
        if i+3 <= len(x):
            print(nuc_order.index(x[i]), nuc_order.index(x[i+1]), nuc_order.index(x[i+2]))
            output.append(aa_abbrevs[aa_codes[nuc_order.index(x[i]),nuc_order.index(x[i+1]),nuc_order.index(x[i+2])]])
    return ''.join(output)'''


#pull mutations from nsps and spike where at least 1 variant has a frequency >=threshold
def compare_genes_thresholded(variant_order, nsp_positions, threshold):
    #read in ref fasta
    fasta = np.array([*get_fasta()[:-1]])

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

    #function to convert nucleotide sequence to amino acid sequence easily
    convert_to_aa = lambda x : ''.join([aa_abbrevs[aa_codes[nuc_order.index(x[i]),nuc_order.index(x[i+1]),nuc_order.index(x[i+2])]] for i in range(0,len(x),3) if i+3 <= len(x)])

    #array storing mutation output
    output_mat = np.array([[[],[],[],[]]])

    #folder storing sequences for each gene
    if not os.path.exists('simulation_output/rdrp_and_spike'):
        os.mkdir('simulation_output/rdrp_and_spike')

    #loop through regions
    for gene,positions in nsp_positions.items():
        subset_fasta = fasta[positions[0]:positions[1]] #splice fasta at gene
        if gene == 'rdrp': #add C frameshift into rdrp
            subset_fasta = np.insert(fasta[positions[0]:positions[1]], 26, ['C'])
        elif gene in ['nsp13', 'nsp14', 'nsp15', 'nsp16']: #adjust starting position to account for frameshift in rdrp
            positions = [positions[0]+2,positions[1]]
            subset_fasta = fasta[positions[0]:positions[1]]
        print(gene)
        #save nucleotides in each gene
        pd.Series(subset_fasta).reindex(np.arange(positions[0],positions[0]+len(subset_fasta))).to_csv('simulation_output/rdrp_and_spike/'+gene+'_nucs.csv')
        
        variant_mut_csvs = []
        #loop through variants
        for variant in variant_order:
            #read in variant mutation data
            variant_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder][0]
            variant_muts = pd.read_csv('sim_ref_data/'+variant_folder+'/collapsed_mutation_list.csv', index_col=0, header=0)

            #drop mutations not in region
            variant_muts = variant_muts.loc[(variant_muts['position'].isin(np.arange(positions[0],positions[1])))]
            variant_muts.index = variant_muts.loc[:,'position']
            variant_muts.drop(['position','T','G','C','A'], axis=1, inplace=True)
            variant_muts.columns = [variant+'_'+column for column in variant_muts.columns]
            #print(variant_muts)
            variant_mut_csvs.append(variant_muts)
        
        complete_mut_df = variant_mut_csvs[0].join(variant_mut_csvs[1:], how='outer')
        #index is position and each variant has variant_mut_freq columns
        complete_mut_df.columns = pd.MultiIndex.from_product([variant_order, ['original','T_freq','G_freq','C_freq','A_freq']])
        #print(complete_mut_df)
        complete_mut_df.to_csv('simulation_output/rdrp_and_spike/complete_mut_df.csv')

        #loop through mutations and remove any that are less than threshold% frequency
        for index in complete_mut_df.index:
            for mut in ['T','G','C','A']:
                #check if any variant has mut at position >= threshold
                if np.any(complete_mut_df.loc[index, (variant_order,mut+'_freq')].to_numpy()>=threshold): 
                    for variant in variant_order:
                        #triplet isn't centered on mutation
                        position = int(index) #position of mutation
                        #print(variant, position, complete_mut_df.loc[index, (variant, mut+'_freq')])
                        triplet_position = (position - positions[0]) % 3
                        #position should by default be offset by 1
                        gene_position = position-positions[0]
                        
                        '''nsp16 example:
                            A20724G = LLEK = L[TTA>TTG]EK
                            where position +=1 and triplet_position=2
                            triplet_position = (20724-20656+2) % 3 = 0
                        '''


                        if triplet_position == 0:  #third position in triplet is mutated
                            nuc_triplet = subset_fasta[gene_position-3:gene_position]
                            mut_triplet = np.copy(nuc_triplet)
                            mut_triplet[2] = mut #mutated nucleotide triplet
                            mut_position = 2
                        elif triplet_position == 1: #first position in triplet is mutated
                            nuc_triplet = subset_fasta[gene_position-1:gene_position+2]
                            mut_triplet = np.copy(nuc_triplet)
                            mut_triplet[0] = mut
                            mut_position = 0
                        else: #second position in triplet is mutated
                            nuc_triplet = subset_fasta[gene_position-2:gene_position+1]
                            mut_triplet = np.copy(nuc_triplet)
                            mut_triplet[1] = mut
                            mut_position = 1
                        #print(nuc_triplet, mut_triplet)
                        nuc_triplet = np.where(nuc_triplet=='T', 'U', nuc_triplet) #replace T->U
                        mut_triplet = np.where(mut_triplet=='T', 'U', mut_triplet) #replace T->U
                        aa_orig = aa_codes[nuc_order.index(nuc_triplet[0]),nuc_order.index(nuc_triplet[1]),nuc_order.index(nuc_triplet[2])] #original amino acid
                        aa_mut = aa_codes[nuc_order.index(mut_triplet[0]),nuc_order.index(mut_triplet[1]),nuc_order.index(mut_triplet[2])] #mutated amino acid
                        aa_change = aa_orig != aa_mut #check if amino acid changed
                        #position -= 1 #reindex position
                        matching_mut = functools.reduce(np.intersect1d, (np.where(output_mat[:,0]==gene)[0], np.where(output_mat[:,1]==str(position))[0], np.where(output_mat[:,3]==mut_triplet[mut_position])[0])) #check if mutation already in output_mat
                        #format: region, position, original nuc, mut nuc, original triplet, mut triplet, mut name (nuc), original aa, mut aa, orig aa name, mut aa name, mut name (aa), aa change, variant frequencies
                        #treat sub regions of the spike as a continuous set for positions
                        if gene in ['NTD','RBD','SD1_2']:
                            aa_pos = int(np.ceil((position-21562)/3))
                        else:
                            aa_pos = int(np.ceil((position-positions[0])/3))
                        mut_row = np.array([[gene, position, nuc_triplet[mut_position], mut_triplet[mut_position], ''.join(nuc_triplet), ''.join(mut_triplet), nuc_triplet[mut_position]+str(position)+mut_triplet[mut_position], aa_abbrevs[aa_orig], aa_abbrevs[aa_mut], aa_orig, aa_mut, aa_abbrevs[aa_orig]+str(aa_pos)+aa_abbrevs[aa_mut], aa_change] + [np.nan]*len(variant_order)])

                        #mutation not found
                        if not matching_mut.size > 0:
                            #add mutation
                            if output_mat.size > 0:
                                output_mat = np.append(output_mat, mut_row, axis=0)
                            else:
                                output_mat = mut_row
                            #get index of new mutation row
                            matching_mut = functools.reduce(np.intersect1d, (np.where(output_mat[:,0]==gene)[0], np.where(output_mat[:,1]==str(position))[0], np.where(output_mat[:,3]==mut_triplet[mut_position])[0]))
                        
                        #update mutation row with frequency for variant
                        output_mat[matching_mut[0], 13+variant_order.index(variant)] = complete_mut_df.loc[index, (variant, mut+'_freq')]
        
    output_mat = pd.DataFrame(output_mat, columns=['region','position','original','mutation','original_nucleotide','mutated_nucleotide','mut_shortened','original_aa_code','mutated_aa_code','original_amino_acid','mutated_amino_acid','aa_shortened','aa_change']+variant_order)
    output_mat['position'] = np.array(output_mat['position'], dtype=np.float64)
    output_mat.sort_values(['region','position'], inplace=True)
    
    #count number of each variant grouping with mutation
    count_columns = pd.DataFrame(np.zeros([output_mat.shape[0],3]), index=output_mat.index, columns=['dominant_count','transient_count','omicron+subs_count'])
    count_dict = {'dominant_count':['alpha','delta','kraken','omicron','pirola'], 'transient_count':['beta','epsilon','eta','gamma','iota','kappa','lambda','mu'], 'omicron+subs_count':['hongkong','kraken','omicron','pirola']}
    for row in output_mat.index:
        for key, value in count_dict.items():
            mat = output_mat.loc[row, value].to_numpy().astype('float')
            count_columns.loc[row, key] = np.count_nonzero(np.where(mat >= threshold, mat, 0))
    output_mat = pd.concat([output_mat, count_columns], axis=1)
    output_mat.loc[output_mat['omicron+subs_count']>0].to_csv('simulation_output/rdrp_and_spike/omi_muts.csv')

    #print(output_mat)
    output_mat.to_csv('simulation_output/rdrp_and_spike/all_muts.csv')
    
    #count amino acid changes
    output_mat = output_mat.to_numpy()
    AA_analysis_df = pd.DataFrame(np.zeros([len(aa_abbrevs),len(aa_abbrevs)]), index=aa_abbrevs.values(), columns=aa_abbrevs.values())
    for row_index in range(output_mat.shape[0]):
        AA_analysis_df.loc[output_mat[row_index, 7], output_mat[row_index, 8]] += 1
    AA_analysis_df.to_csv('simulation_output/rdrp_and_spike/aa_analysis.csv')

    #count unique mutations for each variant
    mut_analysis_df = pd.DataFrame(np.zeros([19,len(variant_order)]), index=['nsp'+str(i+1) for i in range(11)]+['rdrp']+['nsp'+str(i) for i in range(13,17)]+['NTD','RBD','SD1_2'], columns=variant_order)
    for row_index in range(output_mat.shape[0]):
        print(output_mat[row_index,:])
        print(np.where(output_mat[row_index,13:27].astype(float)>.1))
        matching_cols = np.where(output_mat[row_index,13:27].astype(float)>.1)[0]
        #check if mutation is unique
        if matching_cols.shape[0] == 1:
            mut_analysis_df.loc[output_mat[row_index,0], variant_order[matching_cols[0]]] += 1

        elif matching_cols.shape[0] == 2:
            #special case for mutation originating in omicron and being passed on to kraken/pirola/hongkong      
            if np.all(matching_cols == np.array([12,13])) or np.all(matching_cols == np.array([9,12])) or np.all(matching_cols == np.array([6,12])):
                mut_analysis_df.loc[output_mat[row_index,0], 'omicron'] += 1
            #special case for mutation shared between kraken/hk or kraken/pirola, attributes to kraken because of origination date
            elif np.all(matching_cols == np.array([6,9])) or np.all(matching_cols == np.array([9,13])):
                mut_analysis_df.loc[output_mat[row_index,0], 'kraken'] += 1
        elif matching_cols.shape[0] == 4:
            #special case for mutation originating in omicron and being passed on to kraken/pirola/hongkong      
            if np.all(matching_cols == np.array([6,9,12,13])):
                mut_analysis_df.loc[output_mat[row_index,0], 'omicron'] += 1
    print(mut_analysis_df)
    mut_analysis_df.to_csv('simulation_output/rdrp_and_spike/unique_mut_counts.csv')

#get count of 4fold muts for each variant at thresholds to see how representative each step is of overall variation
def analyze_fourfold_thresholds(thresholds):
    #thresholds = list of strings

    output_df = pd.DataFrame()
    totals_df = []
    variants = []

    #loop through variants
    for var_folder in [folder for folder in os.listdir('sim_ref_data') if '_full_clade' in folder and 'jean' not in folder]:
        variants.append(re.search(r'\(\w+\)', var_folder).group(0)[1:-1])

        #read in collapsed mut list
        collapsed_muts = pd.read_csv('sim_ref_data/'+var_folder+'/collapsed_mutation_list.csv', index_col=0, header=0)
              
        #read in valid 4fold positions
        #valid positions is indexed at 0, ref_muts are indexed at 1
        valid_fourfold_positions = pd.read_csv('sim_ref_data/fourfold_gwtc/valid_fourfold_positions/total.csv', index_col=0, header=0).to_numpy().flatten()
        #drop muts not at 4fold sites
        collapsed_muts.drop(collapsed_muts.loc[~collapsed_muts['position'].isin(valid_fourfold_positions+1)].index, axis=0, inplace=True)

        var_df = pd.DataFrame()

        #loop through thresholds
        for threshold in thresholds:

            print(var_folder, threshold, collapsed_muts.shape)
            mat = pd.DataFrame(np.zeros([4,4]), index=['T>','G>','C>','A>'], columns=['T','G','C','A'])

            if 'alpha' in var_folder and threshold == '5e-05':
                testing_mat = pd.DataFrame()

            #loop through original nucleotides
            for orig_nuc in ['T','G','C','A']:

                #loop through mut_nuc
                for mut_nuc in ['T','G','C','A']:

                    #subset collapsed mutations to only include those going from orig_nuc > mut_nuc at freq >= threshold
                    subset_df = collapsed_muts.loc[(collapsed_muts['original']==orig_nuc) & (collapsed_muts[mut_nuc+'_freq'].astype(float) >= float(threshold))]
                    mat.loc[orig_nuc+'>', mut_nuc] = subset_df.shape[0]
                    if 'alpha' in var_folder and threshold == '5e-05':
                        testing_mat = pd.concat([testing_mat, subset_df], axis=0)
                        print(subset_df.shape[0])
            
            var_df = pd.concat([var_df, mat], axis=1)
            total = len(np.nonzero(np.where(collapsed_muts.loc[:,['T_freq','G_freq','C_freq','A_freq']].to_numpy(dtype=float) >= float(threshold), collapsed_muts.loc[:,['T_freq','G_freq','C_freq','A_freq']].to_numpy(dtype=float), 0))[0])
            if np.sum(mat.to_numpy()) == total:
                totals_df.append(total)
            else:
                print(variant_folder, threshold, ' totals do not line up, inspect')
            if 'alpha' in var_folder and threshold == '5e-05':
                print('pls')
                testing_mat.to_csv('simulation_output/analyze_fourfold_thresholds_alpha_testing.csv')

        output_df = pd.concat([output_df, var_df], axis=0)
    output_df.index = pd.MultiIndex.from_product([variants, ['T>','G>','C>','A>']])
    output_df.columns = pd.MultiIndex.from_product([thresholds, ['T','G','C','A']])
    output_df.to_csv('simulation_output/final_info/variant_threshold_fourfold_matrix.csv')
    totals_df = pd.DataFrame(np.array(totals_df).reshape(len(variants), len(thresholds)), index=variants, columns = thresholds)
    totals_df.to_csv('simulation_output/final_info/variant_threshold_fourfold_counts.csv')

    fig, axs = plt.subplots(figsize=(15,15))
    
    for variant in totals_df.index:
        var_hist = np.array([1,2,3,4,5]).repeat(totals_df.loc[variant,thresholds])
        print(np.unique(var_hist, return_counts=True))
        var_hist = pd.Series(var_hist, dtype=float)
        ax = sns.ecdfplot(data=var_hist, stat='proportion')
    ax.set_xticks(range(0,6,1), [0,.00005, .0005, .005, .05, .5])
    
    plt.savefig('simulation_output/final_info/variant_threshold_fourfold_counts_cdf.png')
    plt.close()

#plotting helper for gen_aggregate_plots()
def aggregate_plots_helper(mat, variance, mat_type, fmt):
    titles = ["Effects of 3' Nucleotides", "Effects of 5' Nucleotides", "3' Context Counts", "5' Context Counts"]
    figsizes = [(15,6),(15,5),(15,6),(15,5)]
    height_ratios = [[.8,.2], [.75,.25],[.8,.2], [.75,.25]]
    file_names = ['5_prime_collapsed_aggregate', '3_prime_collapsed_aggregate', '5_prime_collapsed_counts', '3_prime_collapsed_counts']

    fig = plt.figure(layout='constrained', dpi=200, figsize=figsizes[mat_type])
    grid = gridspec.GridSpec(2,1, figure=fig, wspace=.1, hspace=.1, height_ratios=height_ratios[mat_type])
    ax0 = fig.add_subplot(grid[0,0])
    ax1 = fig.add_subplot(grid[1,0])
    sns.heatmap(np.round(mat, 3), cmap='Greys', ax=ax0, annot=True, fmt=fmt, linewidths=.5, linecolor='gray', annot_kws={'fontsize':18})
    ax0.set_title(titles[mat_type], fontsize=20)
    ax0.set_yticklabels(mat.index, rotation='horizontal', fontsize=16)
    ax0.set_xticks([])
    sns.heatmap(np.round(variance.reshape(1,12), 3), cmap='Greys', ax=ax1, annot=True, fmt=fmt, linewidths=.5, linecolor='gray', annot_kws={'fontsize':18}, xticklabels=columns_figs, cbar=False)
    ax1.set_xticklabels(columns_figs, fontsize=16)
    ax1.set_yticklabels(['max.var'], rotation='horizontal', fontsize=16)
    plt.savefig('simulation_output/final_info/'+file_names[mat_type]+'.png')
    plt.close()

#gen plots for aggregate when collapsing 5' and 3'
def gen_aggregate_plots(mat):
    #averaging across rates is an approximation
    '''#collapsing 5' = U[X>Y]U,U[X>Y]G,U[X>Y]C,U[X>Y]A = N[X>Y]U,N[X>Y]G,N[X>Y]C,N[X>Y]A so 12x12 -> 4x12
    five_prime_df = pd.DataFrame(np.zeros([4,12]), index=['N[X>Y]U','N[X>Y]G','N[X>Y]C','N[X>Y]A'], columns=columns_figs)
    for index, row_indices in enumerate([[0,4,8],[1,5,9],[2,6,10],[3,7,11]]):
        five_prime_df.iloc[index] = np.mean(np.array(mat)[row_indices,:], axis=0)
    variance = np.max(five_prime_df.to_numpy(), axis=0) - np.min(five_prime_df.to_numpy(), axis=0)
    aggregate_plots_helper(five_prime_df, variance, 0)

    #3'
    three_prime_df = pd.DataFrame(np.zeros([3,12]), index=['U[X>Y]N','G[X>Y]N','C[X>Y]N'], columns=columns_figs)
    for row_index in range(0,12,4):
        three_prime_df.iloc[int(row_index/4)] = np.mean(np.array(mat)[row_index:row_index+4], axis=0)
    variance = np.max(three_prime_df.to_numpy(), axis=0) - np.min(three_prime_df.to_numpy(), axis=0)
    aggregate_plots_helper(three_prime_df, variance, 1)'''
    
    #re-calc method
    triplet_counts = pd.read_csv('sim_ref_data/fourfold_gwtc/triplets/total.csv', index_col=0, header=0)
    row_counts = pd.DataFrame(np.zeros([4,12]), index=['N[X>Y]U','N[X>Y]G','N[X>Y]C','N[X>Y]A'], columns=columns_figs)
    row_triplets = pd.DataFrame(np.zeros([4,12]), index=['N[X>Y]U','N[X>Y]G','N[X>Y]C','N[X>Y]A'], columns=columns_figs)
    for index, row_indices in enumerate([[0,4,8],[1,5,9],[2,6,10],[3,7,11]]):
        row_counts.iloc[index] = np.sum(np.array(mat[row_indices] * np.repeat(triplet_counts.iloc[row_indices].to_numpy(), [3]).reshape(-1,12)), axis=0)
        row_triplets.iloc[index] = np.sum(np.repeat(triplet_counts.iloc[row_indices].to_numpy(),[3]).reshape(-1,12), axis=0)
        #five_prime_df.iloc[index] = row_counts / row_triplets
    row_counts.to_csv('simulation_output/final_info/five_prime_cdp_counts.csv')
    row_triplets.to_csv('simulation_output/final_info/five_prime_triplet_counts.csv')
    five_prime_df = row_counts / row_triplets
    variance = np.max(five_prime_df.to_numpy(), axis=0) - np.min(five_prime_df.to_numpy(), axis=0)
    aggregate_plots_helper(five_prime_df, variance, 0, '.2f')
    variance = np.max(row_triplets.to_numpy(), axis=0) - np.min(row_triplets.to_numpy(), axis=0)
    aggregate_plots_helper(row_triplets, variance, 2, 'g')

    row_counts = pd.DataFrame(np.zeros([3,12]), index=['U[X>Y]N','G[X>Y]N','C[X>Y]N'], columns=columns_figs)
    row_triplets = pd.DataFrame(np.zeros([3,12]), index=['U[X>Y]N','G[X>Y]N','C[X>Y]N'], columns=columns_figs)
    for row_index in range(0,12,4):
        row_counts.iloc[int(row_index/4)] = np.sum(np.array(mat[row_index:row_index+4] * np.repeat(triplet_counts.iloc[row_index:row_index+4].to_numpy(), [3]).reshape(-1,12)), axis=0)
        row_triplets.iloc[int(row_index/4)] = np.sum(np.repeat(triplet_counts.iloc[row_index:row_index+4].to_numpy(),[3]).reshape(-1,12), axis=0)
        #three_prime_df.iloc[int(row_index/4)] = row_counts / row_triplets
    row_counts.to_csv('simulation_output/final_info/three_prime_cdp_counts.csv')
    row_triplets.to_csv('simulation_output/final_info/three_prime_triplet_counts.csv')
    three_prime_df = row_counts / row_triplets
    variance = np.max(three_prime_df.to_numpy(), axis=0) - np.min(three_prime_df.to_numpy(), axis=0)
    aggregate_plots_helper(three_prime_df, variance, 1, '.2f')
    variance = np.max(row_triplets.to_numpy(), axis=0) - np.min(row_triplets.to_numpy(), axis=0)
    aggregate_plots_helper(row_triplets, variance, 3, 'g')



def testing_alpha():
    analyze_df = pd.read_csv('simulation_output/analyze_fourfold_thresholds_alpha_testing.csv', index_col=0, header=0)
    convert_df = pd.read_csv('simulation_output/convert_reference_figure_to_list_alpha_testing.csv', index_col=0, header=0)

    df_1 = analyze_df.drop(analyze_df.loc[analyze_df['position'].isin(convert_df['position'])].index, axis=0)
    df_2 = convert_df.drop(convert_df.loc[convert_df['position'].isin(analyze_df['position'])].index, axis=0)

    print(df_1)
    print(df_2)


#create a dataframe with 12x12 full contexts, 1x12 naive, 1x12 variances
def collate_mat_rates_and_variances(global_avg_subset_mat, global_naive_subset_mat, variant_order):
    output_df = pd.DataFrame()
    #loop through variants
    for var_index, variant in enumerate(variant_order):
        full_mat = pd.DataFrame(global_avg_subset_mat[var_index], index=rows_figs[:-4], columns=columns_figs)
        naive_mat = pd.DataFrame(global_naive_subset_mat[var_index].reshape(1,12), index=['N[X>Y]N'], columns=columns_figs)
        max_variance = pd.DataFrame((torch.max(global_avg_subset_mat[var_index], dim=0)[0] - torch.min(global_avg_subset_mat[var_index], dim=0)[0]).reshape(1,12), index=['max.var'], columns=columns_figs)
        output_df = pd.concat([output_df, pd.concat([full_mat,naive_mat,max_variance], axis=0)], axis=1)
    output_df.columns = pd.MultiIndex.from_product([variant_order,(columns_figs)])
    output_df.to_csv('simulation_output/final_info/variant_comparison_matrices.csv')

#plot two mut matrices next to each other for comparison
def compare_two_variants(variants):
    fig, axs = plt.subplots(figsize=(15,6), ncols=2)
    #read in matrices
    mat_1 = pd.read_csv('sim_ref_data/'+[folder for folder in os.listdir('sim_ref_data') if 'clade' in folder and '('+variants[0]+')' in folder][0]+'/thresholded_mutations/5e-05_mat.csv', index_col=0, header=0)
    mat_2 = pd.read_csv('sim_ref_data/'+[folder for folder in os.listdir('sim_ref_data') if 'clade' in folder and '('+variants[1]+')' in folder][0]+'/thresholded_mutations/5e-05_mat.csv', index_col=0, header=0)
    #plot matrices
    sns.heatmap(mat_1, ax=axs[0], linewidth=.5, linecolor='gray', cbar=False, cmap='Greys')
    sns.heatmap(mat_2, ax=axs[1], linewidth=.5, linecolor='gray', cmap='Greys')
    axs[0].set_title(variants[0])
    axs[1].set_title(variants[1])
    axs[0].set_xticklabels(columns_figs)
    axs[1].set_xticklabels(columns_figs)
    axs[0].set_yticklabels(rows_figs[:-4], rotation='horizontal')
    axs[1].set_yticklabels(rows_figs[:-4], rotation='horizontal')
    plt.savefig('simulation_output/final_info/'+variants[0]+'_vs_'+variants[1]+'_comparison.png')
    plt.close()

#place mutations provided into 12x12 cdp matrix
def gen_mut_mat(positions, orig_nucs, mut_nucs, shape=(12,12)):
    #read in fasta
    fasta = np.array([*get_fasta()[:-1]])
    #output matrix
    output_df = pd.DataFrame(np.zeros(shape), columns=columns)
    if shape == (12,12):
        output_df.index = rows
    elif shape == (16,12):
        output_df.index = rows + ["A[X>Y]T","A[X>Y]G","A[X>Y]C","A[X>Y]A"]
    else:
        print('invalid shape sent to gen_mut_mat')

    for index, mut in enumerate(zip(positions,orig_nucs,mut_nucs)):
        try:
            triplet = fasta[mut[0]-2:mut[0]+1]
            output_df.loc[triplet[0]+'[X>Y]'+triplet[-1], mut[1]+'>'+mut[-1]] += 1
        except:
            print(f'position {mut[0]} is out of bounds')
            continue
    return output_df

#gen genome context counts
def gen_genome_context_counts():
    fasta = np.array([*get_fasta()[:-1]])
    context_counts = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=columns_shortened)
    context_list = pd.DataFrame(np.zeros([len(fasta),6]), columns=['T','G','C','A','matching','triplet'])
    for position in range(1,fasta.shape[0]-1):
        triplet = fasta[position-1:position+2]
        context_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
        context_list.loc[position, 'triplet'] = ''.join(triplet)
    context_counts.to_csv('simulation_output/context_counts/genome_wide_contexts.csv')
    context_list.to_csv('simulation_output/context_counts/genome_wide_list.csv')


#look through population mutation data and find regions/positions of lowest frequency
def search_population_mutations(subset_genes, window_sizes):
    dataframes = {}
    for dataset in ['population','vivo']:
        print(dataset)
        #create dataframe with each position, nucleotide, and mutation to store variant proportions
        mut_frequencies = pd.concat([pd.Series(np.repeat(np.arange(1,29904),4), name='position'), pd.Series(np.repeat(np.array([*get_fasta()[:-1]]), 4), name='old'), pd.Series(np.tile(['T','G','C','A'], 29903), name='mut')], axis=1)
        if dataset == 'population':
            var_folders = [folder for folder in os.listdir('sim_ref_data') if 'clade' in folder and 'jean' not in folder]
        else:
            var_folders = [folder for folder in os.listdir('sim_ref_data') if 'clade' in folder and 'jean' in folder]
        variants = [re.search(r'\(\w+\)', folder).group(0)[1:-1] for folder in var_folders]
        for var_folder in var_folders:
            variant = re.search(r'\(\w+\)', var_folder).group(0)[1:-1]
            var_muts = pd.read_csv('sim_ref_data/'+var_folder+'/reference_mutations/5e-05_reference_mutations.csv', header=0)
            dummy_merge = mut_frequencies.merge(var_muts, how='outer', left_on=['position','old','mut'], right_on=['position','old','mut'], suffixes=[None, variant])
            var_col = dummy_merge.loc[:,'proportion']
            var_col.name = variant
            mut_frequencies = pd.concat([mut_frequencies, var_col], axis=1)
        #mut_frequencies.to_csv('simulation_output/final_info/'+dataset+'_mut_freqs.csv')
        #shape = 119612x21

        #create series for genome that shows avg mut freq across variants
        mut_freq_avg = pd.Series(np.zeros([29903]), index=np.arange(1,29904))
        for position in range(1,29904):
            subset_df = mut_frequencies.loc[mut_frequencies['position']==position].copy()
            subset_df.drop(subset_df.loc[subset_df['old']==subset_df['mut']].index, axis=0, inplace=True) #drop mut row that cannot exist
            subset_df = subset_df.loc[:,variants] #pull mut proportions for variants
            mut_freq_avg.loc[position] = np.mean(subset_df.fillna(0).to_numpy())
        #mut_freq_avg.to_csv('simulation_output/final_info/'+dataset+'_mut_freq_avg.csv')
        
        thresholds = [0, .00005, .0005, .005, .05, .5, 1]
        for t_index, threshold in enumerate(thresholds[1:]):
            num_muts = mut_freq_avg.loc[(mut_freq_avg >= thresholds[t_index]) & (mut_freq_avg <= threshold)]
            #num_muts = mut_freq_avg.to_numpy()
            #num_muts = num_muts[(num_muts>=thresholds[t_index]) & (num_muts<=threshold)]
            print(f'{threshold} has {num_muts.shape[0]}')
        dataframes[dataset] = [mut_frequencies, mut_freq_avg]
    
    #now want to look for regions of lowest frequency
    region_df = pd.DataFrame([])
    for dataset in ['population','vivo']:
        print(dataset)
        for window_size in window_sizes:
            print(window_size)
            mut_freq_avg = dataframes[dataset][1].to_numpy()
            region_freq_avg = np.array([])
            for position in range(mut_freq_avg.shape[0]-window_size):
                region_freq_avg = np.append(region_freq_avg, np.mean(mut_freq_avg[position:position+window_size]))
            region_df = pd.concat([region_df, pd.Series(np.append(region_freq_avg, np.zeros([29903-region_freq_avg.shape[0]])), name=dataset+'_'+str(window_size))], axis=1)
    print(region_df)
    region_df.to_csv('simulation_output/final_info/low_freq_regions.csv')
    region_minimums = pd.DataFrame([])
    for col in region_df.columns:
        column = region_df.loc[:,col] #isolate dataset/window_size
        column = column.loc[column>0] #remove window_size number of final elements that can't have a full window
        print(col)
        #print(np.argpartition(column.loc[column>0].to_numpy().flatten(), 10))
        #partition moves elements < kth to the left of kth
        #then sort those to get the kth lowest values
        #region_minimums[col] = np.argpartition(column.loc[column>0], kth=20, axis=None)
        lowest_values = column.loc[np.argpartition(column, kth=20, axis=None)[:20]]
        region_minimums[col] = lowest_values.index[np.argsort(lowest_values)]
        region_minimums[col+'_vals'] = np.sort(lowest_values)
    region_minimums.to_csv('simulation_output/final_info/low_freq_regions_mins.csv')

    fig, axs = plt.subplots(figsize=(15,6))
    columns = [column for column in region_minimums.columns if '_vals' not in column]
    empty_arr = np.zeros([29903, len(columns)])
    empty_arr[:] = np.nan
    plot_df = pd.DataFrame(empty_arr, columns=columns)
    for height, column in enumerate(columns):
        min_index = region_minimums.loc[:,column].min()
        max_index = region_minimums.loc[:,column].max()
        window_size = int(column.split('_')[-1])
        if max_index <= min_index+window_size:
            #only 1 region
            plot_df.loc[np.arange(min_index,min_index+window_size),column] = height
        else:
            #multiple regions
            #going to try and find all non-overlapping windows
            indices = np.sort(region_minimums.loc[:,column])
            final_indices = []
            for position, index in enumerate(indices):
                if position != 0:
                    if index >= indices[position-1]+window_size:
                        final_indices.append(index)
            for index in final_indices:
                plot_df.loc[np.arange(min_index,min_index+window_size),column] = height
        #sns.lineplot(plot_df.loc[:,column], ax=axs)
    #print(plot_df)
    plot_df.to_csv('simulation_output/final_info/low_freq_regions_plot.csv')
    sns.lineplot(plot_df, ax=axs)
    plt.savefig('simulation_output/final_info/low_freq_regions_mins.png')
    plt.close()




    #do the same but for genes?
    region_df = pd.DataFrame([])
    for dataset in ['population','vivo']:
        region_freq_avg = np.array([])
        mut_freq_avg = dataframes[dataset][1].to_numpy()
        for gene, positions in subset_genes.items():
            if len(positions) > 2:
                sub_regions = np.array([])
                for sub_position in positions:
                    sub_regions = np.append(sub_regions, np.mean(mut_freq_avg[sub_position[0]:sub_position[1]]))
                region_freq_avg = np.append(region_freq_avg, np.mean(sub_regions))
            else:
                region_freq_avg = np.append(region_freq_avg, np.mean(mut_freq_avg[positions[0]:positions[1]]))
        region_df[dataset] = pd.Series(region_freq_avg, index=subset_genes.keys())
    region_df.to_csv('simulation_output/final_info/low_freq_genes.csv')
    
#append neighboring nucleotide columns to collapsed_mutation_list
def append_neighboring_nucs_to_collapsed_mutation_list(variants):
    fasta = np.array([*get_fasta()[:-1]])
    #print(len(fasta))
    #loop through variants
    for variant in variants:
        var_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder and 'clade' in folder][0]
        collapsed_mut_list = pd.read_csv('sim_ref_data/'+var_folder+'/collapsed_mutation_list.csv', index_col=0, header=0)
        upstream_nucs, downstream_nucs = [],[]
        for position in collapsed_mut_list['position']:
            #print(position, fasta[int(position)-2], fasta[int(position)])
            upstream_nucs.append(fasta[int(position)-2]) #adjust for indexing
            downstream_nucs.append(fasta[int(position)])
        collapsed_mut_list['upstream_nucs'] = pd.Series(upstream_nucs, index=collapsed_mut_list['position'])
        collapsed_mut_list['downstream_nucs'] = pd.Series(downstream_nucs, index=collapsed_mut_list['position'])
        #if 1 in collapsed_mut_list['position'].to_numpy():
        #    collapsed_mut_list.loc[1.0, 'upstream_nucs'] = np.nan
        #elif 29903 in collapsed_mut_list['position'].to_numpy():
        #    collapsed_mut_list.loc[29903.0, 'downstream_nucs'] = np.nan
        collapsed_mut_list.to_csv('sim_ref_data/'+var_folder+'/collapsed_mutation_list_extended.csv')

#create fasta from averaged sim muts
def prep_sim_fasta_for_blast(context_type, analysis_variant, analysis_threshold, num_muts):
    mut_df = pd.DataFrame()
    for mut_dict in os.listdir('simulation_output/global/'+context_type+'/'+analysis_variant+'/mut_dicts/'+analysis_threshold):
        muts = pd.read_csv('simulation_output/global/'+context_type+'/'+analysis_variant+'/mut_dicts/'+analysis_threshold+'/'+mut_dict, index_col=0, header=0)
        muts.drop(99999999, inplace=True) #drop metadata row
        muts['left'] = [nuc[0] for nuc in muts['original'].to_numpy()]
        muts['right'] = [nuc[-1] for nuc in muts['original'].to_numpy()]
        muts['middle'] = [nuc[1] for nuc in muts['original'].to_numpy()]
        muts['mut'] = [nuc[1] for nuc in muts['mutation'].to_numpy()]
        print(mut_df)
        if not mut_df.empty:
            if np.any(muts['position'].isin(mut_df['position'])):
                matching_positions = muts.loc[muts['position'].isin(mut_df['position'])]
                #mut_df_subset = mut_df.loc[mut_df['position'].isin(muts['position'])]
                #matching_positions = matching_positions.loc[(matching_positions['position']==mut_df_subset['position']) & (matching_positions['mut']==mut_df_subset['mut'])]
                print(matching_positions)
                mut_df.loc[mut_df['position'].isin(matching_positions['position']),'count'] += 1
                muts.drop(matching_positions.index, inplace=True)
            new_muts = muts.loc[:,['left','middle','right','mut','position']]
            new_muts['count'] = np.ones([new_muts.shape[0],1])
            mut_df = pd.concat([mut_df, new_muts])
        else:
            mut_df['left'] = muts['left']
            mut_df['middle'] = muts['middle']
            mut_df['right'] = muts['right']
            mut_df['mut'] = muts['mut']
            mut_df['position'] = muts['position']
            mut_df['count'] = np.ones([mut_df.shape[0],1])
    mut_df.to_csv('simulation_output/testing_000.csv')



#create an image of population aggregate vs vivo aggregate with and without c->t column
def gen_all_vs_total_output_figure():
    fig, axs = plt.subplots(figsize=(26,20), nrows=2, ncols=2, dpi=200, layout='tight') #output figure
    for fig_index, fig_type in enumerate(['full','cut']): #correlate full matrices and matrices with removing C>U
        for index, dataset in enumerate(['all','jean_total']): #loop through datasets
            var_folder = [folder for folder in os.listdir('sim_ref_data') if 'full_clade' in folder and '('+dataset+')' in folder][0]
            if dataset=='all':
                name = 'Population'
                mut_mat = pd.read_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/5e-05_mat.csv', index_col=0, header=0).to_numpy()
            else:
                name = 'In Vivo'
                mut_mat = pd.read_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/both_mut_rate_mat.csv', index_col=0, header=0).to_numpy()
            
            if mut_mat.shape[0] > 12: #check if matrix includes A[]N rows
                mut_mat = mut_mat[:12,:12]
            
            convert_val = lambda x: str(int(x)) if x in [0,1] else str(np.round(x,2))
            if fig_type == 'full':
                mut_mat = mut_mat / np.max(mut_mat) #normalize matrix so each rate in [0,1]
                annot = np.array([[convert_val(mut_mat[index,col]) for col in range(12)] for index in range(12)])

            else:
                mut_mat = np.delete(mut_mat, 6, 1)
                mut_mat = mut_mat / np.max(mut_mat)
                mut_mat = np.insert(mut_mat, 6, np.repeat(-1,12), axis=1)
                annot = np.array([[convert_val(mut_mat[index,col]) for col in range(12)] for index in range(12)])
            

            #plot full matrix
            #annot = np.round(mut_mat.to_numpy(), 2)
            print(annot)
            print(np.where(annot=='-1.0', True,False))
            sns.heatmap(mut_mat, cmap='Greys', ax=axs[fig_index,index], linecolor='gray', linewidth=.5, annot=annot, fmt='s', annot_kws={'fontsize':22}, vmax=1, vmin=0, mask=annot=='-1.0')
            axs[fig_index,index].set_xticklabels(columns_figs, rotation=45, fontsize=22)
            axs[fig_index,index].set_yticklabels(rows_figs[:12], rotation='horizontal', fontsize=22)
            axs[fig_index,index].set_title(name+' '+fig_type, fontsize=28)

    plt.savefig('simulation_output/final_info/final_figs/all_vs_total.png')


#pull fastas from simulation and format them with variant reference fastas
def format_sim_fastas(output_variant, threshold, mut_count, dataset_toggle='population'):
    if dataset_toggle == 'population':
        analysis_folder = 'analysis_'+str(mut_count)
    elif dataset_toggle == 'vivo':
        analysis_folder = 'analysis_'+str(mut_count)+'_weighted'
    for context_type in ['blind_contexts','naive_contexts','full_contexts']:
        with open('simulation_output/global/'+analysis_folder+'/'+context_type+'_fastas.fa', 'w') as file:
            for sim_fasta in os.listdir('simulation_output/global/'+analysis_folder+'/'+context_type+'/'+output_variant+'/fastas/'+str(threshold)+'/'):
                lines = open('simulation_output/global/'+analysis_folder+'/'+context_type+'/'+output_variant+'/fastas/'+str(threshold)+'/'+sim_fasta).readlines()
                for line in lines:
                    file.write(line)
                file.write('\n')

#generate mean along series given a window size
def calc_series_windows(df, window_size, cols, type, thresholds, pop_type='frequency', directions=['middle']):
    #pop_type=='frequency: use population frequencies directly; pop_type==binary: convert to presence/absense of mutation
    print(f'generating series for {window_size}')
    starting_time = time.perf_counter()

    #df.loc[:,cols] = (df.loc[:,cols]-df.loc[:,cols].max(axis=None))/df.loc[:,cols].std(axis=None)
    #df.loc[:,cols] = (df.loc[:,cols]/df.loc[:,cols].max(axis=None))
    #df.loc[:,cols] = (df.loc[:,cols]-df.loc[:,cols].max(axis=None))/(df.loc[:,cols].max(axis=None)-df.loc[:,cols].min(axis=None))

    if type == 'matching':
        df = df.loc[:,cols].sum(axis=1) #sum across cols, collapsing TGCA muts
        df = df.to_numpy(na_value=0.0) #convert df to numpy array, convert nans to absence of mutation
    elif type == 'pop':
        #need to drop rows that can't occur (T>T, G>G, C>C, A>A)
        #print('starting shape: ',df.shape)
        df.drop(df.loc[df['old']==df['mut']].index, axis=0, inplace=True)
        #print(df.shape)
        df = df.loc[:,cols]
        
        df[df<thresholds[0]] = 0 #remove mutations with frequency lower than min threshold
        df[df>thresholds[1]] = 0 #remove mutations with frequency higher than max threshold
        df.fillna(0, inplace=True) #absense of mutation should be 0
        df = np.reshape(df, shape=(-1,len(cols)*3)) #collapse every set of 3 rows; [T>G,T>C,T>A] becomes T>N
        if pop_type == 'frequency':
            df = df.mean(axis=1)
        elif pop_type == 'binary':
            df[df>0] = 1
            df = df.mean(axis=1)
        num_pop_muts = np.count_nonzero(df) #count number of positions that mutated
        #print('ending shape: ',df.shape)
        print('number of mutations: ',num_pop_muts)
    
    temp_df = pd.DataFrame(np.zeros([29903,3]), columns=['forward','backward','middle']).replace(0,np.nan)

    for direction_index, direction in enumerate(directions):
        #ignore window
        if window_size == 0:
            temp_df.loc[:,direction] = df

        #calc mean of window
        else:
            if direction == 'forward': #position is start of window
                for index in range(0,29903-window_size):
                    if type == 'matching':
                        temp_df.loc[index,'forward'] = np.mean(df[index:index+window_size])
                    elif type == 'pop':
                        if pop_type == 'frequency':
                            temp_df.loc[index,'forward'] = np.nanmean(df[index:index+window_size])
                        elif pop_type == 'binary':
                            temp_df.loc[index,'forward'] = np.sum(df[index:index+window_size])
            
            elif direction == 'backward': #position is end of window
                for index in range(29903,0+window_size,-1):
                    if type == 'matching':
                        temp_df.loc[index,'backward'] = np.mean(df[index-window_size:index])
                    elif type == 'pop':
                        if pop_type == 'frequency':
                            temp_df.loc[index,'backward'] = np.nanmean(df[index-window_size:index])#each position has 3 possible muts so every 4 indices account for 1 nucleotide position
                        elif pop_type == 'binary':
                            temp_df.loc[index,'backward'] = np.sum(df[index-window_size:index])
            
            elif direction == 'middle': #position is middle of window
                if window_size > 1:
                    half_window = int(window_size/2)
                    for index in range(half_window, 29903-half_window):
                        if type == 'matching':
                            temp_df.loc[index,'middle'] = np.sum(df[index-half_window:index+half_window])/window_size #mean gives different result because of counting 0's on a 2d array
                        elif type == 'pop':
                            if pop_type == 'frequency':
                                temp_df.loc[index,'middle'] = np.nanmean(df[index-half_window:index+half_window]) #each position has 3 possible muts so every 3 indices account for 1 nucleotide position
                            elif pop_type == 'binary':
                                temp_df.loc[index,'middle'] = np.sum(df[index-half_window:index+half_window])
        
    #normalize by number of mutations
    if pop_type == 'binary':
        temp_df = temp_df / num_pop_muts
    
    print(f'time elapsed: {time.perf_counter() - starting_time}')

    return temp_df.loc[:,directions]
    



#take simulation output and population mutation data to plot regions of high/low mutation frequency
def low_and_high_figure(output_types=['sim'], sim_folder_path='', window_size=10, thresholds=[], regions_dict={}):
    #read in simulation_output 
    sim_hist = pd.read_csv(sim_folder_path, index_col=0, header=0)

    #going to have a dataframe with all 29903 positions with each column being an averaged result over the window size
    plot_df = pd.DataFrame(np.zeros([29903,1]))
    plot_df.columns = ['matching']

    #loop through genome with sliding window
    plot_df.loc[:,'sim_avg_'+str(window_size)] = calc_series_windows(sim_hist, window_size, ['T','G','C','A'], 'matching', None, None, ['middle'])
    
    #create figure
    fig = plt.figure(layout='constrained', dpi=200, figsize=(22,6))
    grid = gridspec.GridSpec(2,1, figure=fig, wspace=.1, hspace=.1, height_ratios=[.8,.2])
    ax0 = fig.add_subplot(grid[0,0]) #simulation data
    ax1 = fig.add_subplot(grid[1,0]) #population data

    #pre-plot manipulation
    #x-axis labels
    x_labels = ['','ORF1ab','S','ORF3ab','E','M','ORF6','ORF7ab','ORF8','N','ORF10','']
    x_positions = [0,265,21562,25392,26244,26522,27201,27393,27893,28273,29557,29675]
    minor_tick_labels = ['nsp1','nsp2','nsp3','nsp4','nsp5','nsp6','nsp7','nsp8','nsp9','nsp10','nsp11\nnsp12','nsp13','nsp14','nsp15','nsp16']
    minor_tick_positions = [266,805,2719,8554,10054,10972,11842,12091,12685,13024,13441,16234,18037,19618,20656]
    #minor_tick_positions = np.arange(0,29903,1000)

    #sim data
    dist_stats = plot_df.loc[:,'sim_avg_'+str(window_size)].describe([0.05,.95]) #calc significantly high and low regions of sim line
    plot_df['sim_avg_final'] = plot_df.loc[:,'sim_avg_'+str(window_size)] #simulation values
    plot_df['sim_avg_low_final'] = np.repeat(dist_stats.loc['5%'], plot_df.shape[0]) #line denoting significantly high values
    plot_df['sim_avg_high_final'] = np.repeat(dist_stats.loc['95%'], plot_df.shape[0]) #line denoting significantly low values

    #read and manipulate population data
    pop_mut_freqs = pd.read_csv('simulation_output/final_info/population_mut_freqs.csv', index_col=0, header=0)
    #loop through window sizes
    for threshold in thresholds:
        #arguments: dataframe, window_size, variant, dataset_type, threshold, calculation_type, window_positioning
        plot_df.loc[:,'pop_'+str(window_size)+'_'+str(threshold[0])] = calc_series_windows(pop_mut_freqs, 100, ['all'], 'pop', threshold, 'binary', ['middle'])
    #save dataframes for tracking
    plot_df.loc[:,[col for col in plot_df.columns if 'pop' in col]].to_csv('simulation_output/final_info/final_sim_analysis/population_mut_windows.csv')
    plot_df.to_csv('simulation_output/final_info/final_sim_analysis/plot_df.csv')
    
    #pop data, calc high and low regions
    #threshold_plot_values = [.1,.2,.3,.4,.5]
    low, high = 0.05, 0.95
    for index, threshold in enumerate(thresholds):
        mean_dist = plot_df.loc[:,'pop_'+str(window_size)+'_'+str(threshold[0])]
        dist_stats = mean_dist.describe([low,high])
        low_indices = mean_dist.loc[mean_dist<=dist_stats.iloc[4]].index
        high_indices = mean_dist.loc[mean_dist>=dist_stats.iloc[6]].index
        print(threshold, dist_stats.iloc[4], dist_stats.iloc[6])

        #loop through low,high,normal regions
        for plot_index, plot_type in enumerate(['_low_final','_high_final','_final']):
            temp_series = np.zeros([mean_dist.shape[0]])
            temp_series[:] = np.nan
            temp_series = pd.Series(temp_series)
            if plot_index == 0: #low regions
                temp_series.loc[low_indices] = index*.1 #subset low_indices
            elif plot_index == 1: #high regions
                temp_series.loc[high_indices] = index*.1 #subset high_indices
            else: #normal regions
                temp_series.loc[~temp_series.index.isin(low_indices.join(high_indices))] = index*.1 #subset indices that are not low or high
            plot_df['pop_'+str(threshold[0])+plot_type] = temp_series
    

    #creating and formatting output plot
    final_plot_df = plot_df.drop(labels=[col for col in plot_df.columns if not 'final' in col], axis=1)
    colors = {}
    for col in final_plot_df.columns:
        if 'high' in col:
            colors[col] = 'r'
        elif 'low' in col:
            colors[col] = 'b'
        else:
            colors[col] = 'k'
    linestyles = {'sim_avg':'solid', 'pop':'solid', 'low':'dashed', 'high':'dashed'}
    #plot
    final_plot_df.loc[:,'sim_avg_final'].plot.line(ax=ax0, color=colors, ls=linestyles['sim_avg'], legend=False)
    final_plot_df.loc[:,'sim_avg_low_final'].plot.line(ax=ax0, color='k', ls=linestyles['low'], legend=False)
    final_plot_df.loc[:,'sim_avg_high_final'].plot.line(ax=ax0, color='k', ls=linestyles['low'], legend=False)
    pop_cols = [col for col in final_plot_df.columns if 'pop' in col and ('high' in col or 'low' in col)]
    final_plot_df.loc[:,pop_cols].plot.line(ax=ax1, color=colors, ls=linestyles['pop'], legend=False, linewidth=5)
 
    #set minor x ticks for nsp labelling
    ax0.set_xticks(minor_tick_positions, labels=minor_tick_labels, minor=True, rotation='vertical', fontsize=12)
    ax0.tick_params(axis='x', which='minor', top=True, labeltop=True, bottom=False, labelbottom=False)
    #remove ticks from population axis
    ax1.set_xticks([])
    #set genome position ticks
    ax0.set_xticks(np.arange(0,29903,2000), labels=np.arange(0,29903,2000))
    #cut x-axis to remove 5' and 3' UTR
    ax0.set_xlim(265,29675)
    ax1.set_xlim(265,29675)
    ax1.set_yticks([])
    ax1.set_ylim(-.1,.1)
    #save figure and dataframes
    plt.savefig('simulation_output/final_info/final_figs/low_and_high_testing/low_and_high_regions.png')
    plot_df = plot_df.loc[:,[col for col in plot_df.columns if 'avg' not in col]]
    final_plot_df.to_csv('simulation_output/final_info/final_sim_analysis/final_plot_df.csv')

    '''spike specific'''
    #'NTD':[21598,22474], 'RBD':[22516,23185],'SD1_2':[23188,25186]
    ax0.set_xticks([21598,22516,23188,23602],labels=['NTD','RBD','SD_1&2','FCS'], minor=True, rotation='vertical', fontsize=12)
    ax0.set_xlim(21562,25384)
    ax1.set_xlim(21562,25384)
    ax0.set_xticks(np.arange(21600,25384,500),np.arange(21600,25384,500))
    plt.savefig('simulation_output/final_info/final_figs/low_and_high_regions_spike.png')

    #Want to compare simulated regions of significance with the significance of SNP data
    output_df = []
    for sig_index, sig_type in enumerate(['high','low']):
        if sig_type == 'high':
            sig_region_indices = final_plot_df.loc[final_plot_df['sim_avg_final']>final_plot_df['sim_avg_high_final']].index.to_numpy()
        elif sig_type == 'low':
            sig_region_indices = final_plot_df.loc[final_plot_df['sim_avg_final']<final_plot_df['sim_avg_low_final']].index.to_numpy()
        
        #check for gaps > window_size base pairs to seperate regions
        region_positions = []
        prev_pos = 0
        for position in sig_region_indices:
            if prev_pos == 0: #first region
                region_positions.append([position])
                prev_pos = position
            else:
                #check if position is far enough away from previous
                if position > prev_pos + window_size: #this should be updated if working with multiple windows
                    #new window
                    region_positions[-1].append(prev_pos)
                    region_positions.append([position])
                #update previous positon
                prev_pos = position
                #if looking at the final index, set final window to end at this position
                if position == sig_region_indices[-1]:
                    region_positions[-1].append(position)

        #calculate info about each region
        #loop through sig low and sig high regions
        output_df_temp = pd.DataFrame(np.zeros([len(region_positions),7]), columns=['start','end','length','pop_high','pop_low','high%','low%'])
        for reg_index, reg_pos in enumerate(region_positions):
            sig_region_df = final_plot_df.loc[reg_pos[0]-(window_size/2)+1:reg_pos[1]+(window_size/2)+1]
            #print(sig_region_df)
            output_df_temp.loc[reg_index,'start'] = reg_pos[0] - (window_size/2) + 1 #reindexing to account for window size
            output_df_temp.loc[reg_index,'end'] = reg_pos[1] + (window_size/2) + 1
            output_df_temp.loc[reg_index,'length'] = sig_region_df.shape[0]
            output_df_temp.loc[reg_index,'pop_high'] = sig_region_df['pop_5e-05_high_final'].count()
            output_df_temp.loc[reg_index,'pop_low'] = sig_region_df['pop_5e-05_low_final'].count()
            output_df_temp.loc[reg_index,'high%'] = output_df_temp.loc[reg_index,'pop_high'] / output_df_temp.loc[reg_index,'length']
            output_df_temp.loc[reg_index,'low%'] = output_df_temp.loc[reg_index,'pop_low'] / output_df_temp.loc[reg_index,'length']
        output_df.append(output_df_temp)
    output_df = pd.concat(output_df).sort_values(by='start', ignore_index=True)
    output_df.to_csv('simulation_output/final_info/final_sim_analysis/sig_regions_df.csv')
    print(output_df)


#convert shared_muts mutation list into nucleotide-mutations.csv format
#want to look at spectrum of unique mutations
def convert_shared_mutations_list(output_folder, threshold, number_of_variants):
    mutations_df = pd.read_csv('simulation_output/final_info/shared_muts/'+str(threshold)+'/'+str(float(number_of_variants))+'_vars.csv', index_col=0, header=0)

    if not os.path.exists('sim_ref_data/'+output_folder):
        os.mkdir('sim_ref_data/'+output_folder)
    
    variants = [column for column in mutations_df.columns if column not in ['position','old','mut','total']]
    output_df = pd.DataFrame()
    for variant in variants:
        variant_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder][0]
        variant_muts = pd.read_csv('sim_ref_data/'+variant_folder+'/nucleotide-mutations.csv', header=0)

        variant_unique_mutations = mutations_df.loc[mutations_df[variant]==1]
        variant_unique_mutations = pd.Series(np.array([variant_unique_mutations.loc[index,'old']+str(variant_unique_mutations.loc[index,'position'])+variant_unique_mutations.loc[index,'mut'] for index in variant_unique_mutations.index]))
        #variant_unique_mutations = pd.Series(variant_unique_mutations.loc[:,'old']+str(variant_unique_mutations.loc[:,'position'])+variant_unique_mutations.loc[:,'mut'])
        #variant_unique_mutations = variant_unique_mutations.loc[:,['old','position','mut']].str.cat(sep='')
        print(variant_unique_mutations)
        subset_variant_muts = variant_muts.loc[(variant_muts['mutation'].isin(variant_unique_mutations))]
        output_df = pd.concat([output_df, subset_variant_muts])
    output_df.to_csv('sim_ref_data/'+output_folder+'/nucleotide-mutations.csv', index=False)

#generate context count matrix for given positions
def gen_context_count_matrix(fasta, positions):
    #create matrix
    context_counts = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=columns_shortened)
    
    #check if sub_region is continuous or broken up
    if len(positions)==2:
        #loop through fasta from starting position to ending position
        for index in range(positions[0],positions[1]):
            #iterate context count
            triplet = fasta[index-1:index+2]
            context_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
    else:
        for sub_positions in positions:
            #loop through fasta from starting position to ending position
            for index in range(sub_positions[0],sub_positions[1]):
                #iterate context count
                triplet = fasta[index-1:index+2]
                context_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
    return context_counts



#analyze genome and gene context counts
def analyze_genome_context_counts(sub_regions):
    #read in reference fasta
    fasta = np.array([*get_fasta()[:-1]])
    #add genome to gene dict
    sub_regions['genome'] = [1,29902]
    #generate context count matrix for each gene
    context_counts = {gene:gen_context_count_matrix(fasta, sub_regions[gene]) for gene in sub_regions.keys()}
    #correlate the gene context counts
    gene_correlation_df = pd.DataFrame(np.zeros([len(context_counts)*2,len(context_counts)]), index=pd.MultiIndex.from_product([['corr','p'],context_counts.keys()]), columns=context_counts.keys())
    for gene_1, context_mat_1 in context_counts.items():
        for gene_2, context_mat_2 in context_counts.items():
            gene_correlation_df.loc[(['corr','p'],gene_1),gene_2] = stats.pearsonr(context_mat_1.to_numpy().flatten(), context_mat_2.to_numpy().flatten())
    #print(gene_correlation_df)

    #correlate gene context counts with 'all' mut mat
    mut_mat = pd.read_csv('sim_ref_data/0(all)_full_clade/thresholded_mutations/5e-05_mat.csv', index_col=0, header=0)
    mut_correlation_df = pd.DataFrame(np.zeros([2,len(context_counts)]), index=['corr','p'], columns=context_counts.keys())
    for gene, context_mat in context_counts.items():
        #context mat shape: 16,4
        #mut_mat shape: 12,12
        mut_correlation_df.loc[['corr','p'],gene] = stats.pearsonr(np.repeat(context_mat.to_numpy()[:12,:], 3), mut_mat.to_numpy().flatten())
    #print(mut_correlation_df)

    #correlate gene context count usage
    gene_context_usage = {gene:context_counts[gene]/context_counts['genome'] for gene in context_counts.keys()}
    gene_usage_correlation_df = pd.DataFrame(np.zeros([len(context_counts)*2,len(context_counts)]), index=pd.MultiIndex.from_product([['corr','p'],context_counts.keys()]), columns=context_counts.keys())
    for gene_1, context_mat_1 in gene_context_usage.items():
            for gene_2, context_mat_2 in gene_context_usage.items():
                if gene_2 != 'genome':
                    gene_usage_correlation_df.loc[(['corr','p'],gene_1),gene_2] = stats.pearsonr(context_mat_1.to_numpy().flatten(), context_mat_2.to_numpy().flatten())
    
    #correlate gene context count usage with 'all' mut mat
    mut_usage_correlation_df = pd.DataFrame(np.zeros([2,len(context_counts)]), index=['corr','p'], columns=context_counts.keys())
    for gene, context_mat in gene_context_usage.items():
        mut_usage_correlation_df.loc[['corr','p'],gene] = stats.pearsonr(np.repeat(context_mat.to_numpy()[:12,:], 3), mut_mat.to_numpy().flatten())

    #save data
    if not os.path.exists('simulation_output/final_info/gene_context_analysis'):
        os.mkdir('simulation_output/final_info/gene_context_analysis')
        os.mkdir('simulation_output/final_info/gene_context_analysis/context_counts')
        os.mkdir('simulation_output/final_info/gene_context_analysis/context_usage')
    gene_correlation_df.to_csv('simulation_output/final_info/gene_context_analysis/gene_correlations.csv')
    mut_correlation_df.to_csv('simulation_output/final_info/gene_context_analysis/mut_mat_correlations.csv')
    gene_usage_correlation_df.to_csv('simulation_output/final_info/gene_context_analysis/gene_usage_correlations.csv')
    mut_usage_correlation_df.to_csv('simulation_output/final_info/gene_context_analysis/usage_mut_mat_correlations.csv')
    for gene in context_counts.keys():
        context_counts[gene].to_csv('simulation_output/final_info/gene_context_analysis/context_counts/'+gene+'_mat.csv')
        gene_context_usage[gene].to_csv('simulation_output/final_info/gene_context_analysis/context_usage/'+gene+'_mat.csv')
    
#create figure for rdrp mutations
def rdrp_table_fig():
    rdrp_df = pd.read_excel('simulation_output/final_info/final_tables/final_rdrp_and_spike.xlsx', sheet_name='rdrp', header=0, nrows=13, usecols=['alpha','beta','delta','epsilon','eta','gamma','hongkong','iota','kappa','kraken','lambda','mu','omicron','pirola'])
    print(rdrp_df)
    rdrp_df.replace(np.nan, 0, inplace=True)

    fig,axs = plt.subplots(figsize=(14,10))
    #by alphabet 'alpha', 'beta', 'delta', 'epsilon', 'eta', 'gamma', 'hongkong', 'iota', 'kappa', 'kraken', 'lambda', 'mu', 'omicron', 'pirola'
    #by start date: 'beta','epsilon','iota','kappa','alpha','gamma','mu','lambda','omicron','kraken','hongkong','pirola'
    variant_order = ['alpha', 'beta', 'delta', 'epsilon', 'eta', 'gamma', 'hongkong', 'iota', 'kappa', 'kraken', 'lambda', 'mu', 'omicron', 'pirola']
    sns.heatmap(rdrp_df.loc[:,variant_order], cmap='Greys', annot=False, ax=axs, linewidth=.5, linecolor='gray')
    axs.set_yticks([])
    abbrevs = {'alpha':'α','beta':'β','delta':'δ','epsilon':'ε','eta':'η','gamma':'γ','hongkong':'HK.3','iota':'ι','kappa':'κ','kraken':'kraken','lambda':'λ','mu':'μ','omicron':'ο','pirola':'pirola'}
    axs.set_xticklabels([abbrevs[var] for var in variant_order])
    axs.tick_params(axis='x', which='major', top=False, labeltop=True, bottom=False, labelbottom=False)
    plt.savefig('simulation_output/final_info/final_figs/rdrp_table_fig_3.png')
    plt.close()

#correlate vaccine/spike with each variant's mutation matrix
def vaccine_corr_updated(global_avg_subset_mat, variant_order):
    vax_dfs = {}
    #calc triplets of vaccines
    for vax_index, vaccine in enumerate(['pfizer','moderna']):
        files = ['pfizer_spike_vaccine.fasta','moderna_spike_vaccine.fasta']
        with open(files[vax_index]) as f:
            vax_ref = f.readlines()[1][:-1]
        triplet_counts = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=columns_shortened)
        context_counts = pd.DataFrame(np.zeros([16,4]), index=rows+['A[X>Y]T','A[X>Y]G','A[X>Y]C','A[X>Y]A'], columns=columns_shortened)
        #calc triplets and contexts
        for index in range(1,len(vax_ref)-1):
            triplet = vax_ref[index-1:index+2]
            context_counts.loc[triplet[0]+'[X>Y]'+triplet[-1],triplet[1]] += 1
            if index%3==0:
                triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1],triplet[1]] += 1
        vax_dfs[vaccine] = [context_counts,triplet_counts]
    
    #correlate context and triplet counts to variant matrices
    output_df = pd.DataFrame(np.zeros([4,len(variant_order)]), index=pd.MultiIndex.from_product([['pfizer','moderna'],['corr','p']]), columns=variant_order)
    for mat, variant in zip(global_avg_subset_mat, variant_order):
        for vax_index, vaccine in enumerate(['pfizer','moderna']):
            for vax_mat in vax_dfs[vaccine]:
                #print(mat, np.repeat(vax_mat.to_numpy()[:12,:], 3))
                corr = stats.pearsonr(mat.numpy().flatten(), np.repeat(vax_mat.to_numpy()[:12,:], 3).flatten())
                output_df.loc[vaccine,variant] = corr
    output_df.to_csv('simulation_output/final_info/vaccine_corr_4_9_25.csv')


#compare mutation rates for each gene
def analyze_gene_mutation_frequencies(variant_order, gene_dict):
    '''
    want to know if spike has a lot of low and high frequency mutations but low middle freq suggesting that selection is a strong factor
    ([4fold,all],var,[high,med,low]),gene
    '''
    output_df = pd.DataFrame(np.zeros([2*len(variant_order)*3,len(gene_dict)]), index=pd.MultiIndex.from_product([['4fold','all'],variant_order,['low','med','high']]), columns=gene_dict.keys())

    #read in valid 4fold site positions
    #valid_fsp = pd.read_csv('sim_ref_data/fourfold_gwtc/valid_fourfold_positions/total.csv', index_col=0, header=0, dtype=int).to_numpy().reshape(-1)

    for var_index, variant in enumerate(variant_order):
        #read in reference mutations for variant
        var_folder = [folder for folder in os.listdir('sim_ref_data') if '('+variant+')' in folder and '_full_clade' in folder][0]
        ref_muts = pd.read_csv('sim_ref_data/'+var_folder+'/reference_mutations/5e-05_reference_mutations.csv', index_col=0, header=0)

        #loop through each gene
        for gene, positions in gene_dict.items():
            if len(positions) == 2:
                gene_ref_muts = ref_muts.loc[ref_muts['position'].isin(np.arange(positions[0],positions[1]))] #subset to mutations in gene

                for mut_type in ['all','4fold']:
                    if mut_type == '4fold':
                        valid_fsp = pd.read_csv('sim_ref_data/fourfold_gwtc/valid_fourfold_positions/'+gene+'.csv', index_col=0, header=0, dtype=int).to_numpy().reshape(-1)
                        temp_ref_muts = gene_ref_muts.loc[gene_ref_muts['position'].isin(valid_fsp)]
                    else:
                        temp_ref_muts = gene_ref_muts
                    print(gene, variant, temp_ref_muts.shape)
                    #low
                    output_df.loc[(mut_type,variant,'low'), gene] = temp_ref_muts.loc[temp_ref_muts['proportion']<5e-03].shape[0]
                    #medium
                    output_df.loc[(mut_type,variant,'med'), gene] = temp_ref_muts.loc[(temp_ref_muts['proportion']>=5e-03) & (temp_ref_muts['proportion']<.5)].shape[0]
                    #high
                    output_df.loc[(mut_type,variant,'high'), gene] = temp_ref_muts.loc[temp_ref_muts['proportion']>=.5].shape[0]
            elif len(positions)>2:
                for sub_positions in positions:
                    gene_ref_muts = ref_muts.loc[ref_muts['position'].isin(np.arange(sub_positions[0],sub_positions[1]))] #subset to mutations in gene

                    for mut_type in ['all','4fold']:
                        if mut_type == '4fold':
                            valid_fsp = pd.read_csv('sim_ref_data/fourfold_gwtc/valid_fourfold_positions/'+gene+'.csv', index_col=0, header=0, dtype=int).to_numpy().reshape(-1)
                            temp_ref_muts = gene_ref_muts.loc[gene_ref_muts['position'].isin(valid_fsp)]
                        else:
                            temp_ref_muts = gene_ref_muts
                        print(temp_ref_muts.shape)
                        #low
                        output_df.loc[(mut_type,variant,'low'), gene] += temp_ref_muts.loc[temp_ref_muts['proportion']<5e-03].shape[0]
                        #medium
                        output_df.loc[(mut_type,variant,'med'), gene] += temp_ref_muts.loc[(temp_ref_muts['proportion']>=5e-03) & (temp_ref_muts['proportion']<.5)].shape[0]
                        #high
                        output_df.loc[(mut_type,variant,'high'), gene] += temp_ref_muts.loc[temp_ref_muts['proportion']>=.5].shape[0]
    
    output_df.to_csv('simulation_output/final_info/gene_mut_frequency_analysis.csv')

    output_df_2 = pd.DataFrame(np.zeros([2*4*3,len(gene_dict)]), index=pd.MultiIndex.from_product([['4fold','all'],['all','dominant','transient','individual'],['low','med','high']]), columns=gene_dict.keys())

    variant_dict = {'all':'all', 'dominant':'dominant', 'transient':'transient', 'individual':['alpha','beta','delta','epsilon','eta','gamma','hongkong','iota','kappa','kraken','lambda','mu','omicron','pirola']}
    for mut_type in ['4fold','all']:
        for variant,variants in variant_dict.items():
            for gene in output_df.columns:
                for mut_freq in ['low','med','high']:
                    if variant == 'individual':
                        output_df_2.loc[(mut_type,variant,mut_freq),gene] = output_df.loc[(mut_type,variants,mut_freq),gene].mean(axis=0)
                    else:
                        output_df_2.loc[(mut_type,variant,mut_freq),gene] = output_df.loc[(mut_type,variants,mut_freq),gene]
    output_df_2.to_csv('simulation_output/final_info/gene_mut_frequency_analysis_ext.csv')
                    

#look at triplet and context counts of introns
def analyze_introns():
    #output dfs
    triplet_counts = pd.DataFrame(np.zeros([16,4]), index=rows_figs, columns=columns_shortened_figs)
    context_counts = pd.DataFrame(np.zeros([16,4]), index=rows_figs, columns=columns_shortened_figs)
    #read in fasta
    fasta = np.array([*get_fasta()[:-1]])
    fasta[fasta=='T'] = 'U'

    #loop through introns
    introns = [[0,265], [29674,29903]]
    for positions in introns:
        #contexts
        for index in range(positions[0]+1,positions[1]-1):
            triplet = fasta[index-1:index+2]
            context_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
        #triplets
        for index in range(positions[0]+2,positions[1],3):
            triplet = fasta[index-2:index+1]
            triplet_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
    if not os.path.exists('simulation_output/final_info/intron_analysis'):
        os.mkdir('simulation_output/final_info/intron_analysis')
    triplet_counts.to_csv('simulation_output/final_info/intron_analysis/triplet_counts.csv')
    context_counts.to_csv('simulation_output/final_info/intron_analysis/context_counts.csv')

    for type, mat in zip(['triplet_counts','context_counts'],[triplet_counts,context_counts]):
        fig, axs = plt.subplots(figsize=(3,4), layout='tight', dpi=200)
        sns.heatmap(mat, ax=axs, annot=False, cmap='Greys', linewidth=.5, linecolor='gray', xticklabels=columns_shortened_figs, yticklabels=rows_figs)
        plt.savefig('simulation_output/final_info/intron_analysis/'+type+'.png')


#calculate context count matrices for each region in region_dict
def calc_contexts_of_regions(regions_dict):
    fasta = np.array([*get_fasta()[:-1]])
    fasta[fasta=='T'] = 'U'
    context_counts_dict = {}

    #loop through each region
    for region, positions in regions_dict.items():
        context_counts = pd.DataFrame(np.zeros([16,4]), index=rows_figs, columns=columns_shortened_figs)
        for index in range(positions[0],positions[1]):
            triplet = fasta[index-1:index+2]
            context_counts.loc[triplet[0]+'[X>Y]'+triplet[-1], triplet[1]] += 1
        context_counts_dict[region] = context_counts

    #read in mutation matrix
    mut_mat = pd.read_csv('sim_ref_data/0(all)_full_clade/thresholded_mutations/5e-05_mat.csv', index_col=0, header=0).to_numpy()
    
    #output matrix for correlations
    output_df = pd.DataFrame(np.zeros([len(context_counts_dict)*2,len(context_counts_dict)+1]), index=pd.MultiIndex.from_product([['corr','p'],context_counts_dict.keys()]), columns=[key for key in context_counts_dict.keys()]+['mut_mat'])
    
    
    #correlate region contexts...
    for region, context_df in context_counts_dict.items():
        #with each other
        for region_2, context_df_2 in context_counts_dict.items():
            corr = stats.pearsonr(context_df.to_numpy().flatten(), context_df_2.to_numpy().flatten())
            output_df.loc[(['corr','p'],region),region_2] = corr
        
        #with mut_mat
        corr = stats.pearsonr(np.repeat(context_df.to_numpy()[:12,:12], 3).flatten(), mut_mat.flatten())
        output_df.loc[(['corr','p'],region),'mut_mat'] = corr

    #save data
    if not os.path.exists('simulation_output/final_info/final_sim_analysis/interesting_regions'):
        os.mkdir('simulation_output/final_info/final_sim_analysis/interesting_regions')
    output_df.to_csv('simulation_output/final_info/final_sim_analysis/interesting_regions/correlations.csv')
    for region, context_df in context_counts_dict.items():
        context_df.to_csv('simulation_output/final_info/final_sim_analysis/interesting_regions/'+str(region)+'_contexts_df.csv')

#generate fasta for gene that includes high frequency polymorphisms
def gen_mutated_fasta(genes, collapse=False, gene_dict={}, variants=False):
    #create output folder
    if not os.path.exists('simulation_output/final_info/mutated_fastas'):
        os.mkdir('simulation_output/final_info/mutated_fastas')

    #read in polymorphism data
    mut_df = pd.read_excel('simulation_output/final_info/final_tables/final_rdrp_and_spike_2.xlsx', sheet_name='complete_set', header=0, index_col=0, nrows=229, usecols='A,B,D,F,N:AA')
    #forward-fill nan values in index
    mut_df.index = mut_df.index.to_series().ffill()
    #print(mut_df)
    #print(mut_df.index)
    #print(mut_df.columns)
    #subset to mutations only provided by variants
    if not variants: #include all variants
        mut_df.drop(['alpha','beta','delta','epsilon','eta','gamma','hongkong','iota','kappa','kraken','lambda','mu','omicron','pirola'], inplace=True)
    else: #subset of variants
        mut_df = mut_df.loc[:,['position','mutation','mutated_nucleotide']+variants]
        #print(mut_df)
        #mut_df.loc[:,variants] = np.array(mut_df.loc[:,variants], dtype=np.float64)
        for index in mut_df.index: #remove any mutation not attributed to selected variants
            if not np.any(mut_df.loc[index,variants]>0.5):
                mut_df.drop(index, axis=0, inplace=True)
    print(mut_df)


    ref = np.array([*get_fasta()[:-1]])
    ref[ref=='T'] = 'U'
    fasta = np.copy(ref)

    for gene in genes:
        gene_df = mut_df.loc[gene]
        gene_df.index = np.arange(gene_df.shape[0])
        print(gene_df)
        if gene_df.size > 3:
            for row in gene_df.index:
                print(fasta[gene_df.loc[row,'position']-2:gene_df.loc[row,'position']+3])
                fasta[gene_df.loc[row,'position']] = gene_df.loc[row,'mutation']
                print(fasta[gene_df.loc[row,'position']-2:gene_df.loc[row,'position']+3])
                print(gene_df.loc[row,'mutated_nucleotide'])
        else:
            fasta[gene_df.loc['position']] = gene_df.loc['mutation']
        if collapse == False:
            with open('simulation_output/final_info/mutated_fastas/'+gene+'.fa', 'w') as f:
                f.write('>'+gene+' mutated fasta from variants: ')
                if not variants:
                    f.write('all')
                else:
                    [f.write(variant+', ') for variant in variants]
                f.write('\n')
                f.write(''.join(fasta[gene_dict[gene][0]:gene_dict[gene][1]]))
                f.write('\n>'+gene+' reference fasta\n')
                f.write(''.join(ref[gene_dict[gene][0]:gene_dict[gene][1]]))
    if collapse != False:
        if collapse in gene_dict.keys():
            gene_start,gene_end = gene_dict[collapse]
        else:
            gene_start = gene_dict[genes[0]][0]
            gene_end = gene_dict[genes[-1]][1]
        with open('simulation_output/final_info/mutated_fastas/'+collapse+'.fa', 'w') as f:
            f.write('>'+gene+' mutated fasta from variants: ')
            if not variants:
                    f.write('all')
            else:
                [f.write(variant+', ') for variant in variants]
            f.write('\n')
            f.write(''.join(fasta[gene_start:gene_end]))
            f.write('\n>'+gene+' reference fasta\n')
            f.write(''.join(ref[gene_start:gene_end]))
    
#calculate ...
def gene_mut_frequency_analysis():
    number_of_sites = pd.read_excel('simulation_output/final_info/gene_mut_frequency_analysis.xlsx', sheet_name='Sheet1')


#analyze WHO covid-19 cases and deaths dataset
def analyze_who_data():
    who_df = pd.read_csv('WHO-COVID-19-global-data.csv', header=0)
    date_col = pd.to_datetime(who_df['Date_reported'])
    date_df = pd.DataFrame([date_col.dt.year, date_col.dt.month, date_col.dt.day], index=['year','month','day'], columns=who_df.index).T
    who_df = pd.concat([who_df, date_df], axis=1)
    who_df.set_index(keys=['Country','year','month','day'], inplace=True) #index df based on country and time
    print(who_df)

    countries = np.unique(who_df.index.get_level_values(0))
    years = np.unique(who_df.index.get_level_values(1))
    output_df = pd.DataFrame(np.zeros([2*len(years),len(countries)+1]), index=pd.MultiIndex.from_product([['cases','deaths'],years]), columns=np.append(countries,['global']))
    #calculate cumulative cases and deaths per year
    for year in years:
        for type in ['cases','deaths']:
            for country in countries:
                if year == years[0]:
                    output_df.loc[(type,year),country] = who_df.loc[(country,year),'Cumulative_'+type].max(axis=0)
                else:
                    output_df.loc[(type,year),country] = who_df.loc[(country,year),'Cumulative_'+type].max(axis=0) - output_df.loc[(type),country].sum(axis=0)
            #global
            output_df.loc[(type,year),'global'] = output_df.loc[(type,year),:].sum(axis=0)
    output_df.to_csv('simulation_output/final_info/WHO_cases_and_deaths_summary.csv')

#convert details.txt files to csv storing all of the gisaid sequences used
def gen_gisaid_id_csv():
    output_df = pd.Series([])
    for variant_folder in [folder for folder in os.listdir('sim_ref_data') if '_full_clade' in folder]:
        if os.path.exists('sim_ref_data/'+variant_folder+'/details.txt'):
            with open('sim_ref_data/'+variant_folder+'/details.txt','r') as f:
                lines = f.readlines()
            output_df = pd.concat([output_df, pd.Series(lines)])
        else:
            print(variant_folder)
    output_df.to_csv('simulation_output/final_info/gisaid_id_set.csv')

#csv was taking too long to load the set so going to copy paste from text file
def gen_gisaid_ids():
    if not os.path.exists('simulation_output/final_info/gisaid_ids'):
        os.mkdir('simulation_output/final_info/gisaid_ids')
    for variant_folder in [folder for folder in os.listdir('sim_ref_data') if '_full_clade' in folder]:
        if os.path.exists('sim_ref_data/'+variant_folder+'/details.txt'):
            with open('sim_ref_data/'+variant_folder+'/details.txt','r') as f:
                    lines = f.readlines()
                    lines = [line.strip() for line in lines]    
            with open('simulation_output/final_info/gisaid_ids/'+re.search(r'\(\w+\)', variant_folder).group(0)[1:-1]+'.txt', 'w') as output:
                    output.write(','.join(lines))

#plot lethal, nonlethal, and combined CDM matrices together
def plot_cdm_types():
    fig, axs = plt.subplots(figsize=(24,8), dpi=250, layout='tight', ncols=3)
    for type_index, type in enumerate(['both', 'non-lethals', 'lethals']):
        mat = pd.read_csv('sim_ref_data/j7(jean_total)_full_clade/thresholded_mutations/'+type+'_mut_rate_mat.csv', index_col=0, header=0).to_numpy()
        cbar_formatter = ticker.ScalarFormatter()
        cbar_formatter.set_scientific(True)
        cbar_formatter.set_powerlimits((-1,1))
        sns.heatmap(mat, ax=axs[type_index], cmap='Greys', linecolor='grey', linewidth=.5, annot=False, cbar=True, cbar_kws={'format':cbar_formatter})
        axs[type_index].set_yticklabels(rows_figs, fontsize=16, rotation='horizontal')
        axs[type_index].set_xticklabels(columns_figs, fontsize=16, rotation=45)
        axs[type_index].set_title(type, fontsize=20)
    plt.savefig('simulation_output/final_info/final_figs/supp_figs/cdm_mats.png')
    plt.close()

#generate triplet or context count matrix based on positions
def calc_count_mat(positions):
    #output df
    counts = pd.DataFrame(np.zeros([16,4]), index=rows_figs, columns=columns_shortened_figs)
    #read in fasta
    fasta = np.array([*get_fasta()[:-1]])
    fasta[fasta=='T'] = 'U' #convert nucleotide

    #loop through positions
    for position in positions:
        triplet = fasta[position-1:position+2]
        counts.loc[triplet[0]+'[X>Y]'+triplet[2], triplet[1]] += 1
    return counts

#save generic count mat to path
def save_count_fig(mat, path):
    path_split = path.split('/')
    fig, axs = plt.subplots(figsize=(3,6), layout='tight', dpi=200)
    sns.heatmap(mat, ax=axs, annot=False, cmap='Greys', linewidth=.5, linecolor='gray')
    axs.set_yticklabels(rows_figs, fontsize=10, rotation='horizontal')
    axs.set_xticklabels(columns_shortened_figs, fontsize=10)
    axs.set_title(path_split[-1][:-4], fontsize=14)
    #[os.mkdir('simulation_output/final_info/final_figs/'+'/'.join(path_split[:i])) for i in range(len(path_split)-1) if not os.path.exists('simulation_output/final_info/final_figs/'+'/'.join(path_split[:i]))]
    temp_str = ''
    for i in range(len(path_split)-1):
        if not os.path.exists('simulation_output/final_info/final_figs/'+temp_str+path_split[i]):
            os.mkdir('simulation_output/final_info/final_figs/'+temp_str+path_split[i])
        temp_str +=path_split[i]+'/'
    plt.savefig('simulation_output/final_info/final_figs/'+path)
    plt.close()

#create table with each gene's total sites, 4fold count, overlapping 4fold count, analyzable 4fold count
def gene_site_analysis(regions_dict, gene_info):
    print(gene_info.keys(), regions_dict.keys())
    '''[[[26252, 'CAT'], [26258, 'TTT'], [26261, 'CGG'], [26270, 'CAG'], [26273, 'GTA'], [26276, 'CGT'], [26285, 'TTA'], [26294, 'TAC'],
    [26297, 'TTC'], [26300, 'TTT'], [26306, 'TTG'], [26309, 'CTT'], [26315, 'TGG'], [26318, 'TAT'], [26327, 'TAG'], [26330, 'TTA'],
    [26333, 'CAC'], [26336, 'TAG'], [26339, 'CCA'], [26345, 'TTA'], [26348, 'CTG'], [26351,'CGC'], [26354, 'TTC'], [26357, 'GAT'],
    [26366, 'CGT'], [26384, 'TTA'], [26390, 'TGA'], [26396, 'TTG'], [26399, 'TAA'], [26405, 'CTT'], [26408, 'CTT'], [26417, 'TTT'],
    [26423, 'CTC'], [26426, 'GTG'], [26429, 'TTA'], [26438, 'TGA'], [26444, 'CTT'], [26447, 'CTA'], [26453, 'TTC'], [26456, 'CTG'],
    [26462, 'TTC'], [26465, 'TGG'], [26468, 'TCT']], [26244, 26472]]'''
    output_df = []
    temp_list = []

    ''' orf1ab > ORF1a > nsp1:16 which will have overlapping since ORF1b is not included
        S > NTD,RBD,sd1_2, no overlapping
        ORF3a is normal?
        E is normal?
        M is normal?
        ORF6 is normal?
        ORF7a is normal?
        orf7b is subset a little
        ORF8 is normal?
        N >orf9b,orf9c overlapping
        orf10 is normal?
    '''
    for gene, positions in gene_info.items():
        #print(f'gene: {gene}, full: {positions[1]}, subset: {regions_dict[gene]}')
        #no overlaps
        '''if gene in ['S']:
            gene_series = pd.Series(np.zeros([4]), index=['total_sites','4fold_sites','overlapping_4folds','analyzable_4folds'], name='full_gene')
            gene_series.loc['total_sites'] = positions[1][1] - positions[1][0]
            gene_series.loc['4fold_sites'] = len(positions[0])
            gene_series.loc['analyzable_4folds'] = len(positions[0])
            series_list = [gene_series]
            for sub_region in ['NTD','RBD','SD1_2']:
                gene_series = pd.Series(np.zeros([4]), index=['total_sites','4fold_sites','overlapping_4folds','analyzable_4folds'], name=sub_region)
                valid_fourfold_positions = pd.read_csv('sim_ref_data/fourfold_gwtc/valid_fourfold_positions/'+sub_region+'.csv', index_col=0, header=0)
                gene_series.loc['total_sites'] = regions_dict[sub_region][1] - regions_dict[sub_region][0]
                gene_series.loc['4fold_sites'] = valid_fourfold_positions.shape[0]
                gene_series.loc['analyzable_4folds'] = valid_fourfold_positions.shape[0]
                series_list.append(gene_series)
            gene_df = pd.concat(series_list, axis=1)'''
        #overlaps
        if gene in ['ORF1ab', 'N', 'ORF7b']:
            if gene == 'ORF1ab':
                sub_regions = ['ORF1a']+['nsp'+str(i) for i in range(1,17)]
                sub_regions[12] = 'rdrp'
                val_check = np.arange(regions_dict['ORF1a'][1], positions[1][1]+1)
            elif gene == 'N':
                sub_regions = ['ORF9b','ORF9c']
                print(regions_dict['ORF9b'], regions_dict['ORF9c'])
                val_check = np.concatenate([np.arange(regions_dict['ORF9b'][0],regions_dict['ORF9b'][1]), np.arange(regions_dict['ORF9c'][0],regions_dict['ORF9c'][1])])
            elif gene == 'ORF7b':
                sub_regions = ['ORF7b']
                val_check = np.arange(regions_dict['ORF7a'][0],regions_dict['ORF7b'][0]+1)
            print(sub_regions)

            gene_series = pd.Series(np.zeros([4]), index=['total_sites','4fold_sites','overlapping_4folds','analyzable_4folds'], name='full_gene')
            gene_series.loc['total_sites'] = positions[1][1] - positions[1][0]
            gene_series.loc['4fold_sites'] = len(positions[0])
            gene_series.loc['analyzable_4folds'] = len(positions[0])
            series_list = [gene_series]
            for sub_region in sub_regions:
                gene_series = pd.Series(np.zeros([4]), index=['total_sites','4fold_sites','overlapping_4folds','analyzable_4folds'], name=sub_region)
                valid_fourfold_positions = pd.read_csv('sim_ref_data/fourfold_gwtc/valid_fourfold_positions/'+sub_region+'.csv', index_col=0, header=0)
                gene_series.loc['total_sites'] = regions_dict[sub_region][1] - regions_dict[sub_region][0]
                gene_series.loc['4fold_sites'] = valid_fourfold_positions.shape[0]
                gene_series.loc['overlapping_4folds'] = valid_fourfold_positions.to_numpy()[valid_fourfold_positions.isin(val_check)].shape[0]
                gene_series.loc['analyzable_4folds'] = gene_series.loc['4fold_sites']-gene_series.loc['overlapping_4folds']
                series_list.append(gene_series)
            gene_df = pd.concat(series_list, axis=1)
            print(gene_df.columns)
            gene_df.loc['overlapping_4folds','full_gene'] = gene_df.iloc[2,1:].sum(axis=None)
            gene_df.loc['analyzable_4folds','full_gene'] = gene_df.loc['4fold_sites','full_gene'] - gene_df.loc['overlapping_4folds','full_gene']
            gene_df.columns = pd.MultiIndex.from_product([[gene],['full_gene']+sub_regions])
            print(gene_df)
            output_df.append(gene_df)
        
        #entire overlap
        elif gene in ['ORF3b']:
            gene_series = pd.Series(np.zeros([4]), index=['total_sites','4fold_sites','overlapping_4folds','analyzable_4folds'], name=gene)
            gene_series.loc['total_sites'] = positions[1][1]-positions[1][0]
            gene_series.loc['4fold_sites'] = len(positions[0])
            gene_series.loc['overlapping_4folds'] = len(positions[0])
            temp_list.append(gene_series)

        #no overlaps
        else:
            gene_series = pd.Series(np.zeros([4]), index=['total_sites','4fold_sites','overlapping_4folds','analyzable_4folds'], name=gene)
            gene_series.loc['total_sites'] = positions[1][1]-positions[1][0]
            gene_series.loc['4fold_sites'] = len(positions[0])
            gene_series.loc['analyzable_4folds'] = len(positions[0])
            temp_list.append(gene_series)
    temp_df = pd.concat(temp_list, axis=1)
    temp_df.columns = pd.MultiIndex.from_product([temp_df.columns.to_numpy(),['full_gene']])
    output_df.append(temp_df)
    output_df = pd.concat(output_df, axis=1)
    print(output_df)
    output_df.to_csv('simulation_output/final_info/final_tables/supp_tables/gene_site_analysis.csv')

#create a figure to show the distribution of sequences for each variant over the analyzed period of time
def gen_variant_seq_dist_fig():
    #read in variant distribution info
    variant_dists = []
    variants = []
    for file in os.listdir('sim_ref_data/variant_time_dists'):
        variants.append(file.split('_')[0])
        variant_dists.append(pd.read_csv('sim_ref_data/variant_time_dists/'+file, header=0))
    #organize variant dist info into df
    dist_df = variant_dists[0]
    for dist_index, dist in enumerate(variant_dists[1:]):
        dist_df = pd.merge(dist_df, dist, on='yearWeek', how='left', suffixes=['', variants[dist_index+1]])
    dist_df.set_index('yearWeek', inplace=True)
    dist_df.columns = pd.MultiIndex.from_product([variants,['numberSamples','proportion']])
    #add in empty rows to display data better
    dist_df.loc['2020-01'] = np.nan
    dist_df.loc['2024-26'] = np.nan
    dist_df.sort_index(inplace=True)
    print(dist_df)
    
    #create figure
    fig, axs = plt.subplots(figsize=(8,12), dpi=200, nrows=2)
    dist_1 = dist_df.loc[:,(variants,'numberSamples')]
    dist_2 = dist_df.loc[:,(variants,'proportion')]
    dist_1.columns = variants
    dist_1[dist_1<100] = np.nan
    dist_2.columns = variants
    #dist_2[dist_2<.002] = np.nan
    
    sns.lineplot(dist_2, ax=axs[0], hue_order=variants, style_order=variants)
    dates = ['2020-26','2021-01','2021-26','2022-01','2022-26','2023-01','2023-26','2024-01','2024-26']
    indices = []
    for date in dates:
        indices.append(np.argwhere(dist_df.index.to_numpy()==date)[0,0])
    axs[0].set_xticks(indices, dates, rotation='vertical')
    axs[0].set_xlim(indices[0],indices[-1])
    
    sns.lineplot(dist_2.loc['2020-32':'2021-44',['beta','epsilon','eta','iota','gamma','kappa','lambda','mu']], ax=axs[1], hue_order=variants, style_order=variants)
    axs[1].set_xticks([0,16,32,48,64], ['2020-32','2020-48','2021-12','2021-28','2021-44'], rotation='vertical')
    plt.savefig('simulation_output/final_info/final_figs/supp_figs/variant_time_distribution.png')
    plt.close()

#create a figure showing sim performance for mut counts
def gen_sim_mut_count_fig():
    #read in simulation info
    sim_df = pd.read_excel('simulation_output/final_info/final_sim_analysis/all_df_final.xlsx', sheet_name='df_for_fig', header=[0,1], index_col=[0,1,2])

    #normalize match counts by the number of mutations placed
    norm_sim_df = sim_df.copy()
    for index in norm_sim_df.index:
        norm_sim_df.loc[index,:] = norm_sim_df.loc[index,:] / index[-1]
    
    '''#create figure
    fig, axs = plt.subplots(figsize=(8,8), dpi=200, nrows=2, layout='tight')
    #positional matches
    pos_df = norm_sim_df.loc[('population','positional_matches'),5e-5]
    pos_df.index = np.arange(100,2100,100)
    pos_df.columns = ['Naive', 'TSTV', 'Context-dependent']
    sns.lineplot(pos_df, ax=axs[0])
    axs[0].set_xticks(np.arange(100,2100,100), np.arange(100,2100,100), rotation='vertical')
    axs[0].set_title('Positional Match Frequency')
    #contextual matches
    con_df = norm_sim_df.loc[('population','contextual_matches'),5e-5]
    con_df.index = np.arange(100,2100,100)
    con_df.columns = ['Naive', 'TSTV', 'Context-dependent']
    sns.lineplot(con_df, ax=axs[1])
    axs[1].set_xticks(np.arange(100,2100,100), np.arange(100,2100,100), rotation='vertical')
    axs[1].set_title('Contextual Match Frequency')
    #axs[1].set_yticks(np.arange(.1,.4,.05), np.round(np.arange(.1,.4,.05), 2))

    plt.savefig('simulation_output/final_info/final_figs/supp_figs/sim_match_count_fig_5e-5.png')
    plt.close()'''
    
    mut_counts = np.arange(100,2100,100)
    #create figure
    fig, axs = plt.subplots(figsize=(8,8), dpi=200, nrows=2, layout='tight')
    #positional performance per added mutation placed
    pos_df = norm_sim_df.loc[('population','positional_matches'),5e-1]
    pos_df.index = np.arange(100,2100,100)
    pos_df.columns = ['Naive', 'TSTV', 'Context-dependent']
    for mc_index, mut_count in enumerate(mut_counts):
        pos_df.loc[mut_count] = pos_df.loc[mut_count] / (mut_counts[mc_index]-mut_counts[mc_index-1])
    pos_df.drop([100], axis=0, inplace=True)
    sns.lineplot(pos_df, ax=axs[0])
    axs[0].set_xticks(np.arange(200,2100,100), np.arange(200,2100,100), rotation='vertical')
    axs[0].set_title('Positional Match Frequency')
    #contextual matches
    con_df = norm_sim_df.loc[('population','contextual_matches'),5e-1]
    con_df.index = np.arange(100,2100,100)
    con_df.columns = ['Naive', 'TSTV', 'Context-dependent']
    for mc_index, mut_count in enumerate(mut_counts):
        con_df.loc[mut_count] = con_df.loc[mut_count] / (mut_counts[mc_index]-mut_counts[mc_index-1])
    con_df.drop([100], axis=0, inplace=True)
    sns.lineplot(con_df, ax=axs[1])
    axs[1].set_xticks(np.arange(200,2100,100), np.arange(200,2100,100), rotation='vertical')
    axs[1].set_title('Contextual Match Frequency')
    plt.savefig('simulation_output/final_info/final_figs/supp_figs/sim_mut_count_fig_5e-1.png')
    plt.close()

#create a figure showing the number of simulation matches weighted by the increase in number of SNPs placed
def gen_sim_mut_thresholding_fig():
    #read in simulation info
    sim_df = pd.read_excel('simulation_output/final_info/final_sim_analysis/all_df_final.xlsx', sheet_name='df_for_fig', header=[0,1], index_col=[0,1])

    mut_counts = np.arange(100,2100,100)
    mut_increases = (mut_counts - (mut_counts-100)) / mut_counts #how many muts added compared to previous mut count
    updated_context_types = ['Naive', 'TSTV', 'Context-Dependent']
    #create figure
    fig, axs = plt.subplots(figsize=(8,8), dpi=200, nrows=3, layout='tight')
    for context_index, context_type in enumerate(['blind_contexts','naive_contexts','full_contexts']):
        pos_match_freq_df = sim_df.loc['positional_matches',(5e-05,context_type)] * mut_increases / mut_counts
        con_match_freq_df = sim_df.loc['contextual_matches',(5e-05,context_type)] * mut_increases / mut_counts
        plot_df = pd.concat([pos_match_freq_df, con_match_freq_df], axis=1)
        #plot_df.reset_index(inplace=True)
        plot_df.columns = ['Positional Frequency', 'Contextual Frequency']
        print(plot_df)
        sns.lineplot(plot_df, ax=axs[context_index])
        axs[context_index].set_title(updated_context_types[context_index])
        axs[context_index].set_ylabel('Matches Per\nAdded Polymorphism', wrap=True)
        axs[context_index].set_xticks(mut_counts,mut_counts,rotation=45)
    plt.savefig('simulation_output/final_info/final_figs/supp_figs/sim_mut_thresholding_5e-5.png')
    plt.close()

#compare in vivo mutation rate contexts to find rates above and below 95% confidence interval
def compare_vivo_rate_dist():
    #read in mut mat
    total_df = pd.read_csv('sim_ref_data/j7(jean_total)_full_clade/thresholded_mutations/both_mut_rate_mat.csv', index_col=0, header=0)
    total_arr = total_df.to_numpy().reshape([-1])
    #calc confidence interval
    dist_stats = pd.Series(total_arr).describe([.025,.975])
    print(dist_stats)
    
    total_arr = np.where((total_arr>=dist_stats.loc['2.5%'])&(total_arr<=dist_stats.loc['97.5%']), np.nan, total_arr)
    total_arr = pd.DataFrame(total_arr.reshape([16,12]), index=total_df.index, columns=total_df.columns)
    print(total_arr)
    total_arr.to_csv('simulation_output/final_info/vivo_rate_dist.csv')

#aggregate and analyze SNP data for variants
def analyze_snps_in_population(region_dict):
    variants = ['alpha','beta','delta','epsilon','eta','hongkong','gamma','iota','kappa','kraken','lambda','mu','omicron','pirola']
    variant_folders = [folder for folder in os.listdir('sim_ref_data') if 'full_clade' in folder]

    mut_df = []
    #read in variant SNP data
    for variant in variants:
        variant_folder = [folder for folder in variant_folders if '('+variant+')' in folder][0]
        mut_df.append(pd.read_csv('sim_ref_data/'+variant_folder+'/collapsed_mutation_list.csv', index_col=0, header=0))

    var_freqs = []
    #convert individial SNP frequencies at each site to agnostic frequencies
    for variant_df in mut_df:
        freqs = variant_df.loc[:,['T_freq','G_freq','C_freq','A_freq']].to_numpy()
        freqs[freqs==0] = 1
        var_coverage = variant_df.loc[:,['T','G','C','A']].to_numpy() / freqs
        var_freqs.append(variant_df.loc[:,['T','G','C','A']].sum(axis=1) / np.sum(var_coverage, axis=1))
    
    #create dataframe including all variants
    freq_df = pd.concat([mut_df[0].loc[:,'position']] + var_freqs, axis=1)
    freq_df.columns = ['position']+variants
    freq_df.to_csv('simulation_output/final_info/final_tables/supp_tables/variant_snp_data.csv')
    print(freq_df)


    output_df = pd.DataFrame(np.zeros([len(region_dict),12]), index=region_dict.keys(), columns=pd.MultiIndex.from_product([[5e-5,.9],['z','p','gene_snps','gene_size','genome_snps','genome_size']]))
    #calculate if snps are clustered in specific genes
    for threshold in [5e-5, .9]:
        #remove snps that are below the threshold
        freq_df.where(freq_df.iloc[:,1:]>threshold, np.nan, inplace=True)
        genome_mat = freq_df.loc[:,variants].to_numpy()
        genome_count = np.count_nonzero(genome_mat.sum(axis=1, where=genome_mat>0))
        #loop through genes
        for gene, positions in region_dict.items():
            if len(positions) == 2:
                subset_df = freq_df.loc[(freq_df['position']>=positions[0])&(freq_df['position']<=positions[1])]
                size = positions[1]-positions[0]
                subset_mat = subset_df.loc[:,variants].to_numpy()
                gene_count = np.count_nonzero(subset_mat.sum(axis=1, where=subset_mat>0))
            else:
                subset_df = []
                size = 0
                for sub_positions in positions:
                    subset_df.append(freq_df.loc[(freq_df['position']>=sub_positions[0])&(freq_df['position']<=sub_positions[1])])
                    size += sub_positions[1]-sub_positions[0]
                subset_df = pd.concat(subset_df)
                subset_mat = subset_df.loc[:,variants].to_numpy()
                gene_count = np.count_nonzero(subset_mat.sum(axis=1, where=subset_mat>0))
            exp_freq = (genome_count - gene_count) / (freq_df.shape[0] - size)
            #proportions test the high freq snps in gene vs rest of genome
            result = stats.binomtest(gene_count, size, exp_freq)
            output_df.loc[gene,threshold] = [result.statistic, result.pvalue, gene_count, size, genome_count-gene_count, freq_df.shape[0]-size]
    print(output_df)
    output_df.to_csv('simulation_output/final_info/final_tables/supp_tables/gene_prop_tests.csv')


