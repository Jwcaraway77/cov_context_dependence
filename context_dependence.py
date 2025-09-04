import numpy as np
import pandas as pd
from utility_funcs import *


#subset fourfold_positions by values within position_bounds
def subset_fourfold_positions(fourfold_positions, position_bounds):
    return np.intersect1d(fourfold_positions, np.arange(*position_bounds))

#generate triplet_count matrix based on subset 4fold positions
def calc_valid_triplet_count(fasta, valid_fourfold_positions):
    #output df
    triplet_counts = pd.DataFrame(np.zeros([12,4]), index=rows, columns=columns_shortened)

    #loop through valid_fourfold_positions
    for position in valid_fourfold_positions:
        triplet = fasta.iloc[position-2:position+2].to_numpy() #[triplet]N
        #print(f'{triplet} at {position} : {triplet[1]}+"[X>Y]"+{triplet[-1]},{ triplet[2]} += 1')
        triplet_counts.loc[triplet[1]+'[X>Y]'+triplet[-1], triplet[2]] += 1
    return triplet_counts

#re-calc 4fold positions for regions without info already (indexing nsps based on nsp rather than ORF1a to avoid moonlighting kinda)
def calc_fourfold_positions(positions, fasta):
    four_fold_codes = ['CT', 'GT', 'TC', 'CC', 'AC', 'GC', 'CG', 'GG']
    valid_positions = []
    for position in range(positions[0]+2,positions[1],3):
        #print(''.join(fasta.loc[position-2:position-1]))
        if ''.join(fasta.loc[position-2:position-1]) in four_fold_codes:
            valid_positions.append(position)
    return valid_positions

#generate 4fold triplet counts and valid 4fold positions for each gene
def calc_fourfold_positions_for_subset_genes(subset_genes, fourfold_positions):
    #read in fasta
    fasta = pd.Series([*get_fasta()[:-1]], name='ref')
    #sum of gene triplet counts = genome wide triplet counts
    total_triplet_counts = pd.DataFrame(np.zeros([12,4]), index=rows, columns=columns_shortened)
    genome_valid_fourfold_positions = np.array([])
    #output directories
    if not os.path.exists('sim_ref_data/4fold/gwtc'):
        os.mkdir('sim_ref_data/4fold/gwtc')
    if not os.path.exists('sim_ref_data/4fold/gwtc/valid_fourfold_positions'):
        os.mkdir('sim_ref_data/4fold/gwtc/valid_fourfold_positions')
    if not os.path.exists('sim_ref_data/4fold/gwtc/triplets'):
        os.mkdir('sim_ref_data/4fold/gwtc/triplets')


    #loop through subset_genes
    for gene, positions in subset_genes.items():
        if not gene in fourfold_positions.keys():
            fourfold_positions[gene] = calc_fourfold_positions(positions, fasta)
        
        valid_fourfold_positions = np.array([], dtype=int)
        if len(positions) > 2:
            for sub_positions in positions:
                valid_fourfold_positions = np.append(valid_fourfold_positions, subset_fourfold_positions(fourfold_positions[gene], sub_positions))
        else:
            valid_fourfold_positions = subset_fourfold_positions(fourfold_positions[gene], positions)
        
        pd.Series(valid_fourfold_positions).to_csv('sim_ref_data/4fold/gwtc/valid_fourfold_positions/'+gene+'.csv')
        triplet_counts = calc_valid_triplet_count(fasta, valid_fourfold_positions)
        triplet_counts.to_csv('sim_ref_data/4fold/gwtc/triplets/'+gene+'.csv')
        if gene in ['S','E','M','N','ORF1a','ORF3a','ORF6','ORF7a','ORF7b','ORF8','ORF10']:
            total_triplet_counts += triplet_counts
            genome_valid_fourfold_positions = np.append(genome_valid_fourfold_positions, valid_fourfold_positions)
        #plot triplet counts
        fig, axs = plt.subplots(figsize=(3,4), layout='tight', dpi=200)
        sns.heatmap(triplet_counts, ax=axs, annot=False, cmap='Greys', linewidth=.5, linecolor='gray', xticklabels=columns_shortened_figs, yticklabels=rows_figs[:12])
        if not os.path.exists('sim_ref_data/4fold/gwtc/triplets/figs'):
            os.mkdir('sim_ref_data/4fold/gwtc/triplets/figs')
        plt.savefig('sim_ref_data/4fold/gwtc/triplets/figs/'+gene+'.png')
        #axs.tick_params(axis='x', which='major', top=True, labeltop=True, bottom=False, labelbottom=False)
    
    
    total_triplet_counts.to_csv('sim_ref_data/4fold/gwtc/triplets/total.csv')
    pd.Series(genome_valid_fourfold_positions).sort_values().to_csv('sim_ref_data/4fold/gwtc/valid_fourfold_positions/total.csv')
    #plot total triplet counts
    fig, axs = plt.subplots(figsize=(3,4), layout='tight', dpi=200)
    sns.heatmap(total_triplet_counts, ax=axs, annot=False, cmap='Greys', linewidth=.5, linecolor='gray', xticklabels=columns_shortened_figs, yticklabels=rows_figs[:12])
    plt.savefig('sim_ref_data/4fold/gwtc/triplets/figs/total.png')

#combine muts for variants too sparse to analyze at 5e-5 etc
def gen_pooled_variant(variants=[], variant_name='', num_req_vars=1):
    #generate output folder
    if not os.path.exists('sim_ref_data/'+variant_name+'_full_clade'):
        os.mkdir('sim_ref_data/'+variant_name+'_full_clade')
    #read in nucleotide-mutations file of first sub-variant
    nucleotide_mutations = pd.read_csv('sim_ref_data/'+[var_folder for var_folder in os.listdir('sim_ref_data') if '('+variants[0]+')' in var_folder and 'full_clade' in var_folder][0]+'/nucleotide-mutations.csv', header=0)
    nucleotide_mutations.index = nucleotide_mutations['mutation']
    nucleotide_mutations.columns = [column+'_'+variants[0] for column in nucleotide_mutations.columns]
    print(nucleotide_mutations.shape)
    
    #loop through rest of sub-variants pulling mutations
    var_csvs = []
    for variant in variants[1:]:
        var_csv = pd.read_csv('sim_ref_data/'+[var_folder for var_folder in os.listdir('sim_ref_data') if '('+variant+')' in var_folder and 'full_clade' in var_folder][0]+'/nucleotide-mutations.csv', header=0)
        var_csv.index = var_csv['mutation']
        var_csv.columns = [column+'_'+variant for column in var_csv.columns]
        print(var_csv.shape)
        var_csvs.append(var_csv)
        
    print(var_csvs)
    #join all variant mutations
    nucleotide_mutations = nucleotide_mutations.join(var_csvs, how='outer')
    #subset all counts for each variant and mutation
    counts_df = nucleotide_mutations.loc[:,['count_'+variant for variant in variants]]
    #subset all proportions for each variant and mutation
    proportions_df = nucleotide_mutations.loc[:,['proportion_'+variant for variant in variants]]
    proportions_df.fillna(0.0, inplace=True)
    counts_df.fillna(0, inplace=True)
    proportions_df.to_csv('sim_ref_data/'+variant_name+'_full_clade/proportions_df.csv') #checking dataframe
    
    
    #calc total number of sequences read at each site across variants
    total_coverage = np.sum(counts_df.to_numpy()/proportions_df.to_numpy(), axis=1, where=proportions_df.to_numpy()>0)

    #calc total mut counts at each site across variants
    total_counts = counts_df.sum(1)
    total_counts.name = 'count'
    #calc proportion of muts/seqs at each site across variants
    proportions = total_counts / total_coverage
    proportions.name = 'proportion'
    #create dummy col for jaccard
    jaccard_col = np.ones([proportions.size])
    jaccard_col[:] = np.nan
    
    
    #when pooling variants with low sequence counts, requiring multiple variants to contribute polymorphisms removes some noise
    nucleotide_mutations = pd.concat([nucleotide_mutations, proportions, total_counts, pd.Series(jaccard_col, index=total_counts.index, name='jaccard')], axis=1)
    nucleotide_mutations.dropna(axis=0, thresh=num_req_vars, subset=[col for col in nucleotide_mutations.columns if 'count_' in col], inplace=True)

    nucleotide_mutations.loc[:,['proportion','count','jaccard']].to_csv('sim_ref_data/'+variant_name+'_full_clade/nucleotide-mutations.csv')
    
    #read in old sequence counts
    with open('sim_ref_data/info', 'r') as f:
        lines = f.readlines()
    seq_counts = {re.search(r' \w+,', lines[line_index]).group(0)[1:-1].lower():[int(re.search(r', \d+', lines[line_index]).group(0)[2:]), line_index] for line_index in range(8, len(lines))}
    #sum number of sequences used in aggregate
    aggregate_seq_count = np.sum([seq_counts[variant][0] for variant in variants])
    #pull info for variant if it already exists
    variant_name_subset = [re.search(r'\d+\(', variant_name).group(0)[:-1], re.search(r'\(\w+\)', variant_name).group(0)]
    variant_line = [seq_counts[key] for key in seq_counts.keys() if variant_name_subset[1] in '('+key+')']

    with open('sim_ref_data/info', 'w') as f:
        #update existing variant if possible
        if len(variant_line) > 0:
            for line_index, line in enumerate(lines):
                if line_index != variant_line[0][1]: 
                    f.write(line)
                else:
                    f.write(variant_name_subset[0]+'* = '+variant_name_subset[1][1:-1]+', '+str(aggregate_seq_count)+' total sequences\n')
        #otherwise add new variant
        else:
            for line_index, line in enumerate(lines):
                f.write(line)
            f.write(variant_name_subset[0]+'* = '+variant_name_subset[1][1:-1]+', '+str(aggregate_seq_count)+' total sequences\n')

#building a new one
def convert_reference_mutation_list_to_figure(variant, threshold):
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
        #ref_mut_list.to_csv('simulation_output/convert_reference_figure_to_list_alpha_testing.csv')

        mut_counts = pd.DataFrame(np.ones([12,12]), index=rows, columns=columns)

    else:
        #read in total genome triplet counts
        total_triplet_counts = pd.read_csv('sim_ref_data/4fold/contexts/jean_total.csv', index_col=0, header=0)
        if total_triplet_counts.shape[0]>12:
            total_triplet_counts.drop(['A[X-->Y]T','A[X-->Y]G','A[X-->Y]C','A[X-->Y]A'], axis=0, inplace=True)
        '''for index in total_triplet_counts.index:
            for col in total_triplet_counts.columns:
                total_triplet_counts.loc[index,col] = 1'''
        mut_counts = pd.DataFrame(np.zeros([12,12]), index=rows, columns=columns)

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
            var_matrix.loc[triplet[0]+'[X>Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += 1
        else:
            #check if valid mutation for jean dataset (we can't use A[X>Y]N)
            if triplet[0]+'[X>Y]'+triplet[2] in var_matrix.index.to_numpy():
                #calc based on frequencies
                if jean_toggle == 'weighted':
                    var_matrix.loc[triplet[0]+'[X>Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += row.loc['proportion']
                    #testing_dict[triplet[0]+'[X>Y]'+triplet[2]][row.loc['old']+'>'+row.loc['mut']].append(float(row.loc['proportion']))
                    mut_counts.loc[triplet[0]+'[X>Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += 1
                #calc based on unique mutations
                else:
                    var_matrix.loc[triplet[0]+'[X>Y]'+triplet[2], row.loc['old']+'>'+row.loc['mut']] += 1
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
        var_matrix /= mut_counts
    for column in var_matrix.columns.values:
        var_matrix.loc[:,column] = var_matrix.loc[:,column].to_numpy() / total_triplet_counts.loc[:,column[0]].to_numpy()
    return var_matrix, mut_count

#convert_reference_mutation_list for each threshold,variant combo and normalize results by mut_count
def parse_reference_mutations(thresholds, variant_order):
    #loop through thresholds
    for threshold in thresholds:
        var_data = []
        max_rate = 0
        #convert_reference_mutation_list
        for variant in variant_order:
            var_data.append([*convert_reference_mutation_list_to_figure(variant, threshold)])

        #save var_matrix and heatmap of variant polymorphism rates
        for variant_index, variant in enumerate(variant_order):
            var_matrix = var_data[variant_index][0]
            #save and plot var_matrix
            var_folder = [var_folder for var_folder in os.listdir('sim_ref_data') if '('+variant+')' in var_folder and 'full_clade' in var_folder][0]
            if not os.path.exists('sim_ref_data/'+var_folder+'/thresholded_mutations'):
                os.mkdir('sim_ref_data/'+var_folder+'/thresholded_mutations')
            var_matrix.to_csv('sim_ref_data/'+var_folder+'/thresholded_mutations/'+str(threshold)+'_mat.csv')

            max_rate = 1
            min_rate=0
            plot_mut_mat(var_matrix.to_numpy(), 'sim_ref_data/'+var_folder+'/thresholded_mutations/'+str(threshold)+'_full_contexts_global.png', [], 'global', 'full_contexts', vmin=min_rate, vmax=max_rate, annot=True)

#grab mutation counts for each threshold
def gen_thresholded_mut_counts_for_analysis(thresholds):
    output_df = pd.DataFrame()
    if not os.path.exists('simulation_output/final_info/variant_mut_count_mats_all'):
        os.mkdir('simulation_output/final_info/variant_mut_count_mats_all')
    #loop through thresholds
    for threshold in thresholds:
        threshold_series = pd.Series(name=threshold)
        if not os.path.exists('simulation_output/final_info/variant_mut_count_mats_all/'+threshold):
            os.mkdir('simulation_output/final_info/variant_mut_count_mats_all/'+threshold)
        for var_folder in [var_folder for var_folder in os.listdir('sim_ref_data') if 'clade' in var_folder]:
            ref_muts = pd.read_csv('sim_ref_data/'+var_folder+'/reference_mutations/'+threshold+'_reference_mutations.csv',index_col=0,header=0).to_numpy()
            var_name = re.search(r'\(\w+\)', var_folder).group(0)[1:-1]
            threshold_series[var_name] = ref_muts.shape[0]
            #gen mut matrix
            gen_mut_mat(ref_muts[:,0],ref_muts[:,-2],ref_muts[:,-1], shape=(16,12)).to_csv('simulation_output/final_info/variant_mut_count_mats_all/'+threshold+'/'+var_name+'_mut_counts.csv')
        output_df = pd.concat([output_df, threshold_series], axis=1)
    #output_df.index = [re.search(r'\(\w+\)', index).group(0)[1:-1] for index in output_df.index]
    print(output_df)
    output_df.to_csv('sim_ref_data/thresholded_mut_counts.csv')

'''generate variant reference data'''
def gen_context_dependent_info(dataset='population', variant_order=[], gene_dict={}, fourfold_positions=[], thresholds=[]):
    calc_fourfold_positions_for_subset_genes(gene_dict, fourfold_positions) #calc fourfold site info
    '''generate aggregate folders'''
    gen_pooled_variant(variants=['alpha','beta','delta','epsilon','eta','iota','hongkong','gamma','kappa','kraken','lambda','mu','omicron','pirola'], variant_name='0(all)') #all variants
    gen_pooled_variant(variants=['beta','epsilon','eta','iota','gamma','kappa','lambda','mu'], variant_name='1(transient)', num_req_vars=3) #transient variants without hk
    gen_pooled_variant(variants=['alpha','delta','kraken','omicron','pirola'], variant_name='2(persistent)') #persistent variants
    gen_pooled_variant(variants=['alpha','beta','delta','omicron','gamma'], variant_name='3(mut_vars)') #direct comparison to jean_total
    pooled_variants = ['all','transient','persistent','mut_vars']
    '''reformat and subset mutation data for variants'''
    for variant in variant_order:
        collapse_mutation_list(variant)
        for threshold in thresholds:
            threshold_reference_mutations(variant, threshold)
    parse_reference_mutations(thresholds=thresholds, variant_order=variant_order+pooled_variants)
    gen_thresholded_mut_counts_for_analysis(thresholds)

