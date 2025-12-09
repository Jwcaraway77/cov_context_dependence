import os
import numpy as np
import pandas as pd
import regex as re

#read in reference fasta
with open('EPI_ISL_402124.fasta', 'r') as f:
    lines = f.readlines()
seq = lines[1]

#function to calculate the 4-fold degenerate sites in the genome
def calc_4fold():
    #genes from GISAID
    ORF1ab = seq[265:21555]
    S = seq[21562:25384]
    ORF3a = seq[25392:26220]
    ORF3b = seq[25764:26220]
    E = seq[26244:26472]
    M = seq[26522:27191]
    ORF6 = seq[27201:27387]
    ORF7a = seq[27393: 27759]
    ORF7b = seq[27755:27887]
    ORF8 = seq[27893:28259]
    N = seq[28273:29533]
    ORF10 = seq[29557:29674] #* There is no evidence that this protein is expressed or plays any role in pathogenesis and transmission.

    #these are sub genes
    ORF1a = seq[265:13483] #part of ORF1ab
    ORF9b = seq[28283:28577] #part of N
    ORF9c = seq[28733:28955] #part of N
    
    #dict storing genes
    genes = {'ORF1ab': [265, ORF1ab], 'S': [21562, S], 'ORF3a': [25392, ORF3a], 'ORF3b': [25764, ORF3b], 'E': [26244, E], 'M': [26522, M], 'ORF6': [27201, ORF6], 'ORF7a': [27393, ORF7a], 'ORF7b': [27755, ORF7b], 'ORF8': [27893, ORF8], 'N': [28273, N], 'ORF10': [29557, ORF10], 'ORF1a': [265, ORF1a], 'ORF9b': [28283, ORF9b], 'ORF9c': [28733, ORF9c]}

    
    four_fold_codes = ['CT', 'GT', 'TC', 'CC', 'AC', 'GC', 'CG', 'GG']
    four_fold_matches = {}
    #loop through genes
    for gene_name, gene_data in genes.items():
        #print('working on ', gene_name, len(gene_data[1]))

        for i in range(0, len(gene_data[1]), 3): #loop through seq[gene] 3 bases at a time
            if len(gene_data[1][i:]) >= 3: #check if there are 3 bases left
                triplet = gene_data[1][i:i+3] #pull triplet
                if triplet[:2] in four_fold_codes: #check if triplet is a 4fold
                    if gene_name in four_fold_matches.keys():
                        four_fold_matches[gene_name][0].append([i + gene_data[0] + 2, seq[i + gene_data[0] + 1 : i + gene_data[0] + 4]]) #store position of third nucleotide and triplet centered on 3rd position
                    else:
                        four_fold_matches[gene_name] = [[[i + gene_data[0] + 2, seq[i+gene_data[0] + 1 : i+gene_data[0] + 4]]]]
    return four_fold_matches

#calculate 4fold degenerate sites by looping through the genome in reverse
def calc_4fold_reverse():
    ORF1ab = seq[265:21555]
    S = seq[21562:25384]
    ORF3a = seq[25392:26220]
    ORF3b = seq[25764:26220]
    E = seq[26244:26472]
    M = seq[26522:27191]
    ORF6 = seq[27201:27387]
    ORF7a = seq[27393: 27759]
    ORF7b = seq[27755:27887]
    ORF8 = seq[27893:28259]
    N = seq[28273:29533]
    ORF10 = seq[29557:29674]
    ORF1a = seq[265:13468]
    ORF9b = seq[28283:28577]
    ORF9c = seq[28733:28955]
    

    genes = {'ORF1ab': [265, ORF1ab], 'S': [21562, S], 'ORF3a': [25392, ORF3a], 'ORF3b': [25764, ORF3b], 'E': [26244, E], 'M': [26522, M], 'ORF6': [27201, ORF6], 'ORF7a': [27393, ORF7a], 'ORF7b': [27755, ORF7b], 'ORF8': [27893, ORF8], 'N': [28273, N], 'ORF10': [29557, ORF10], 'ORF1a': [265, ORF1a], 'ORF9b': [28283, ORF9b], 'ORF9c': [28733, ORF9c]}

    four_fold_codes = ['CT', 'GT', 'TC', 'CC', 'AC', 'GC', 'CG', 'GG']
    four_fold_matches = {}
    for gene_name, gene_data in genes.items():
        #print('working on ', gene_name, len(gene_data[1]))
        for i in range(len(gene_data[1]), 0, -3):
            if len(gene_data[1][:i-3]) >= 3 or len(gene_data[1][:i-3]) == 0:
                triplet = gene_data[1][i-3:i][::-1]
                #if gene_name == 'E':
                #    print(triplet)
                if triplet[:2] in four_fold_codes:
                    if gene_name in four_fold_matches.keys():
                        four_fold_matches[gene_name][0].append([i + gene_data[0] - 3, seq[i + gene_data[0] - 4: i + gene_data[0] - 1][::-1]]) #store position of third position and triplet centered on 3rd position
                    else:
                        four_fold_matches[gene_name] = [[[i + gene_data[0] - 3, seq[i + gene_data[0] - 4 : i + gene_data[0] - 1][::-1]]]]
    return four_fold_matches

#create dict with each gene, starting and ending positions, and 4fold sites
def get_gene_info():
    four_fold_matches = calc_4fold()
    for i, key in enumerate(four_fold_matches.keys()):
        gene_locations = [[265,21555],[21562,25384],[25392,26220],[25764,26220],[26244,26472],[26522,27191],[27201,27387],[27393,27759],[27755,27887],[27893,28259],[28273,29533],[29557,29674],[265,13468],[28283,28577],[28733,28955]]
        four_fold_matches[key].append(gene_locations[i])

    return four_fold_matches

#converting flanking nucleotides into index of context on context-dependent matrix
def rowswitch(flanking_nucs):
    if flanking_nucs == 'TT':
        rowiter = 0
    elif flanking_nucs == 'TG':
        rowiter = 1
    elif flanking_nucs == 'TC':
        rowiter = 2
    elif flanking_nucs == 'TA':
        rowiter = 3
    elif flanking_nucs == 'GT':
        rowiter = 4
    elif flanking_nucs == 'GG':
        rowiter = 5
    elif flanking_nucs == 'GC':
        rowiter = 6
    elif flanking_nucs == 'GA':
        rowiter = 7
    elif flanking_nucs == 'CT':
        rowiter = 8
    elif flanking_nucs == 'CG':
        rowiter = 9
    elif flanking_nucs == 'CC':
        rowiter = 10
    elif flanking_nucs == 'CA':
        rowiter = 11
    elif flanking_nucs == 'AT':
        rowiter = 12
    elif flanking_nucs == 'AG':
        rowiter = 13
    elif flanking_nucs == 'AC':
        rowiter = 14
    elif flanking_nucs == 'AA':
        rowiter = 15
    else:
        print(flanking_nucs + ' is breaking the switch case')
        return -1
    return rowiter

#convert nucleotide to position on context-dependent matrix
def colswitch(middle_nuc):
    if middle_nuc == 'T':
        coliter = 0
    elif middle_nuc == 'G':
        coliter = 1
    elif middle_nuc == 'C':
        coliter = 2
    elif middle_nuc == 'A':
        coliter = 3
    else:
        print(middle_nuc + ' is breaking the switch case')
        return -1
    return coliter

#calculate genome wide triplet counts for 4fold degenerate sites
def gen_4fold_gwtc():
    #triplets are index of 3rd position and the nucleotides are centered on the third position so GTTT is [2, 'TTT']
    four_fold_matches = calc_4fold()
    for gene in four_fold_matches.keys():
        gwtc_mat = np.zeros((16,4))
        for four_fold in four_fold_matches[gene][0]:
            flanking_nucs = four_fold[1][0] + four_fold[1][-1]
            middle_nuc = four_fold[1][1]
            rowiter = rowswitch(flanking_nucs)
            coliter = colswitch(middle_nuc)
            gwtc_mat[rowiter, coliter] += 1
        #print(gene, gwtc_mat)
            
        gwtc_mat = pd.DataFrame(gwtc_mat, columns = ['T','G','C','A'], index = ['T[X]T','T[X]G','T[X]C','T[X]A','G[X]T','G[X]G','G[X]C','G[X]A','C[X]T','C[X]G','C[X]C','C[X]A','A[X]T','A[X]G','A[X]C','A[X]A'])
        gwtc_mat.to_csv('./4fold_gwtc/'+gene+'_gwtc_mat.csv')
        #print(gene)
        #print(gwtc_mat)
        #print(np.sum(gwtc_mat.values))

    #triplets are index of 3rd position and the nucleotides are centered on the third position so GTTT is [2, 'TTT']
    four_fold_matches = calc_4fold_reverse()
    #print(four_fold_matches['E'])
    for gene in four_fold_matches.keys():
        #print(len(four_fold_matches[gene][0]))
        gwtc_mat = np.zeros((16,4))
        for four_fold in four_fold_matches[gene][0]:
            flanking_nucs = four_fold[1][0] + four_fold[1][-1]
            middle_nuc = four_fold[1][1]
            rowiter = rowswitch(flanking_nucs)
            coliter = colswitch(middle_nuc)
            gwtc_mat[rowiter, coliter] += 1
        #print(gene, gwtc_mat)
            
        gwtc_mat = pd.DataFrame(gwtc_mat, columns = ['T','G','C','A'], index = ['T[X]T','T[X]G','T[X]C','T[X]A','G[X]T','G[X]G','G[X]C','G[X]A','C[X]T','C[X]G','C[X]C','C[X]A','A[X]T','A[X]G','A[X]C','A[X]A'])
        gwtc_mat.to_csv('./4fold_gwtc_rev/'+gene+'_gwtc_mat_rev.csv')
        #print(gene)
        #print(gwtc_mat)
        #print(np.sum(gwtc_mat.values))

#calculate genome wide triplet count for non-4fold degenerate sites
def gen_nonsyn_gwtc():
    gene_info = get_gene_info()

    #dataframe for triplet weights
    triplets = ['AAA', 'AAC', 'AAT', 'AAG', 'ACA', 'ACC', 'ACT', 'ACG', 'ATA', 'ATC', 'ATT', 'ATG', 'AGA', 'AGC', 'AGT', 'AGG', 'CAA', 'CAC', 'CAT', 'CAG', 'CCA', 'CCC', 'CCT', 'CCG', 'CTA', 'CTC', 'CTT', 'CTG', 'CGA', 'CGC', 'CGT', 'CGG', 'TAA', 'TAC', 'TAT', 'TAG', 'TCA', 'TCC', 'TCT', 'TCG', 'TTA', 'TTC', 'TTT', 'TTG', 'TGA', 'TGC', 'TGT', 'TGG', 'GAA', 'GAC', 'GAT', 'GAG', 'GCA', 'GCC', 'GCT', 'GCG', 'GTA', 'GTC', 'GTT', 'GTG', 'GGA', 'GGC', 'GGT', 'GGG']
    triplet_weights = pd.DataFrame(np.zeros([len(triplets), 3]), index=triplets)
    print(triplet_weights.loc['AAA', 1])
    for triplet in triplets:
        for i in range(3):
            if i == 0:
                #mutation in first position
                '''if triplet in ['TTT', 'TTC', 'CTT', 'CTC', 'ATT', 'ATC', 'ATA', 'ATG', 'GTT', 'GTC', 'GTA', 'GTG',
                               'TCT', 'TCC', 'TCA', 'TCG', 'CCT', 'CCC', 'CCA', 'CCG', 'ACT', 'ACC', 'ACA', 'ACG', 
                               'GCT', 'GCC', 'GCA', 'GCG', 'TAT', 'TAC', 'TAA', 'TAG', 'CAT', 'CAC', 'CAA', 'CAG', 
                               'AAT', 'AAC', 'AAA', 'AAG', 'GAT', 'GAC', 'GAA', 'GAG', 'TGT', 'TGC', 'TGA', 'TGG',
                               'CGT', 'CGC', 'AGT', 'AGC', 'GGT', 'GGC', 'GGA', 'GGG']:
                    triplet_weights.loc[triplet,0] = 1 #other three nucs result in a different amino acid'''
                if triplet in ['TTA', 'TTG', 'CTA', 'CTG', 'CGA', 'CGG', 'AGA', 'AGG']:
                    triplet_weights.loc[triplet,0] = 2/3 #2/3 other nucs result in a different amino acid
                else:
                    triplet_weights.loc[triplet,0] = 1 #other three nucs result in a different amino acid
            elif i == 1:
                #mutation in the second position
                if triplet in ['TAA', 'TGA']:
                    triplet_weights.loc[triplet, 1] = 2/3
                else: # triplet in ['TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG']: #else
                    triplet_weights.loc[triplet, 1] = 1 #other 3 are different amino acids
            else:
                #mutation in the third position
                if triplet in ['ATG', 'TGG', 'TGA']:
                    triplet_weights.loc[triplet, 2] = 1
                elif triplet in ['TGT', 'TGC', 'TGA', 'TGG', 'TTT', 'TTC', 'TTA', 'TTG', 'TAT',
                                 'TAC', 'CAT', 'CAC', 'CAA', 'CAG', 'AAT', 'AAC', 'AAA', 'AAG',
                                 'GAT', 'GAC', 'GAA', 'GAG', 'AGT', 'AGC', 'AGA', 'AGG', 'TAA', 'TAG']:
                    triplet_weights.loc[triplet, 2] = 2/3
                elif triplet in ['ATT', 'ATC', 'ATA']:
                    triplet_weights.loc[triplet, 2] = 1/3
                #else:
                #    triplet_weights.loc[triplet, 2] = 0 #4fold shouldn't need to have this because its initialized as zeros
    print(triplet_weights)
    triplet_weights.to_csv('./nonsyn_triplets_weights')
    #triplet_weights.loc['CTT', 2] = 5.0

    #calc forward nonsyn triplet counts
    #seq = 'CATGACGAGGTAAT'
    for gene, values in gene_info.items():
        #if gene == 'E':
            nonsyn_triplet_count = pd.DataFrame(np.zeros([16,4]), columns = ['T','G','C','A'], index = ['T[X]T','T[X]G','T[X]C','T[X]A','G[X]T','G[X]G','G[X]C','G[X]A','C[X]T','C[X]G','C[X]C','C[X]A','A[X]T','A[X]G','A[X]C','A[X]A'])
            gene_location = values[1]
            fasta_subset = seq[gene_location[0]-1:gene_location[1]+1]
            #print(gene, len(re.findall(r'CGT', fasta_subset, overlapped=True)))
            #print('fasta length ', len(fasta_subset))
            #print(gene, fasta_subset)
            for triplet_index in range(1, len(fasta_subset)-3, 3):
                for nucleotide_index in range(3):
                    #print(triplet_index)
                    triplet = fasta_subset[triplet_index:triplet_index+3]
                    context_string = fasta_subset[triplet_index-1:triplet_index+4]
                    #print('context string', context_string, ' triplet ', triplet, ' triplet_index ', triplet_index)
                    context = context_string[nucleotide_index] + '[X]' + context_string[nucleotide_index+2]
                    mutation = triplet[nucleotide_index]
                    #if context == 'T[X]T':
                    #    print(triplet, context, mutation, triplet_weights.loc[triplet,:])
                    #print(triplet_index, triplet, context, mutation)
                    if triplet_index == 1:
                        #ATG will always be 1
                        #print('adding ', triplet_weights.loc[triplet, 1])
                        nonsyn_triplet_count.loc[context, mutation] += triplet_weights.loc[triplet, 1]
                    else:
                        if nucleotide_index%3 == 0: #mutation at first position 
                            #print('adding ', triplet_weights.loc[triplet, 0])
                            nonsyn_triplet_count.loc[context, mutation] += triplet_weights.loc[triplet, 0]
                        elif nucleotide_index%3 == 1:
                            #print('adding ', triplet_weights.loc[triplet, 1])
                            nonsyn_triplet_count.loc[context, mutation] += triplet_weights.loc[triplet, 1]
                        else:
                            #print('adding ', triplet_weights.loc[triplet, 2])
                            nonsyn_triplet_count.loc[context, mutation] += triplet_weights.loc[triplet, 2]
            #print(nonsyn_triplet_count)
            nonsyn_triplet_count.to_csv('./nonsyn_gwtc/'+gene+'_gwtc_mat.csv')


            #reverse nonsyn triplets
            nonsyn_triplet_count = pd.DataFrame(np.zeros([16,4]), columns = ['T','G','C','A'], index = ['T[X]T','T[X]G','T[X]C','T[X]A','G[X]T','G[X]G','G[X]C','G[X]A','C[X]T','C[X]G','C[X]C','C[X]A','A[X]T','A[X]G','A[X]C','A[X]A'])
            gene_location = values[1]
            fasta_subset = seq[gene_location[0]-1:gene_location[1]+1][::-1]
            print(gene, len(re.findall(r'CGT', fasta_subset, overlapped=True)))
            print('fasta length ', len(fasta_subset))
            #print(gene, fasta_subset)
            for triplet_index in range(1, len(fasta_subset)-3, 3):
                for nucleotide_index in range(3):
                    print(triplet_index)
                    triplet = fasta_subset[triplet_index:triplet_index+3]
                    context_string = fasta_subset[triplet_index-1:triplet_index+4]
                    print('context string', context_string, ' triplet ', triplet, ' triplet_index ', triplet_index)
                    context = context_string[nucleotide_index] + '[X]' + context_string[nucleotide_index+2]
                    mutation = triplet[nucleotide_index]
                    #if context == 'T[X]T':
                    #    print(triplet, context, mutation, triplet_weights.loc[triplet,:])
                    print(triplet_index, triplet, context, mutation)
                    #if triplet_index == 1:
                    #    #ATG will always be 1
                    #    print('adding ', triplet_weights.loc[triplet, 1])
                    #    nonsyn_triplet_count.loc[context, mutation] += triplet_weights.loc[triplet, 1]
                    #else:
                    if nucleotide_index%3 == 0: #mutation at first position 
                        print('adding ', triplet_weights.loc[triplet, 0])
                        nonsyn_triplet_count.loc[context, mutation] += triplet_weights.loc[triplet, 0]
                    elif nucleotide_index%3 == 1:
                        print('adding ', triplet_weights.loc[triplet, 1])
                        nonsyn_triplet_count.loc[context, mutation] += triplet_weights.loc[triplet, 1]
                    else:
                        print('adding ', triplet_weights.loc[triplet, 2])
                        nonsyn_triplet_count.loc[context, mutation] += triplet_weights.loc[triplet, 2]
            print(nonsyn_triplet_count)
            nonsyn_triplet_count.to_csv('./nonsyn_gwtc_rev/'+gene+'_gwtc_mat_rev.csv')
      
def main():
    #gen_nonsyn_gwtc()
    #gen_4fold_gwtc()
    gene_info = get_gene_info()
    print(seq[gene_info['S'][1][0]-1:gene_info['S'][1][1]+1])

    #forward checking
    with open('checking_gwtc', 'w') as f:
        for i in range(0, len(seq[gene_info['S'][1][0]:gene_info['S'][1][1]]), 3):
            f.write(seq[gene_info['S'][1][0]:gene_info['S'][1][1]][i:i+3]+'\n')
    
    #reverse checking
    #with open('checking_gwtc', 'w') as f:
    #    for i in range(0, len(seq[gene_info['S'][1][0]:gene_info['S'][1][1]]), 3):
    #        f.write(seq[gene_info['S'][1][0]:gene_info['S'][1][1]][::-1][i:i+3]+'\n')
    #print(len('../../../CDMAP_Output/Correlation_Repository/nonsyn/Downstream/RevComp_Lcore_Replichore/A/avgsubset\\Caprahircus-Nigeria_ORF1ab_muts_nonsyn_uniques_RevCompliment_Context_MutLeft_Replichore.csv'))
    #m = gene_info['M']
    #m = [item[1] for item in m[0]]
    #print(m.count('CGC'))
    #print(seq[21424:21428])
    #print(seq[26269:26272])
    #print(get_gene_info()[1]['E'])
    #print(seq[26269:26273])
    #print(seq[26274])
    #print(seq[get_gene_info()[0]['ORF1ab'][0][0]-2:get_gene_info()[0]['ORF1ab'][0][0]+1])
    #print(get_gene_info()[1]['ORF1ab'][0])
    #calc_4fold()
    '''with open('./regions_combined/USA/csv_total/ORF1ab/ORF1ab_muts_4fold_uniques.csv', 'r') as f:
        lines = f.readlines()
    with open('./USA_ORF1ab_muts_4fold_uniques.csv', 'w') as f:
        for line in lines:
            line= line.split(',')
            line_num = int(line[0])
            line_num -= 265
            f.write(str(line_num) +','+','.join(line[1:]))'''
    pass

if __name__ == '__main__':
    main()