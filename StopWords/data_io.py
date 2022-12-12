import os, sys
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import array_caster

def texts_nwd_csr(list_texts):
    '''
    Make a csr n_wd matrix from a list of texts.
    each text is a list of tokens.
    provide dict_w_iw == mapping of words to indices i_w=0,...,V-1
    '''

    ## unqiue words and alphabetically sorted list of words
    set_vocab = set([ token for doc in list_texts for token in doc ])
    list_w = sorted(list(set_vocab))
    V = len(list_w)
    dict_w_iw = dict(zip( list_w,np.arange(V)))


    D = len(list_texts)
    V = len(dict_w_iw)

    ## csr-format of the data
    rows = []
    cols = []
    data = []

    for i_doc, doc in enumerate(list_texts):
        data += [1]*len(doc)
        cols += [i_doc]*len(doc)
        rows += [ dict_w_iw[h] for h in doc]
    
    data= array_caster.to_array(data)
    cols=array_caster.to_array(cols)
    rows=array_caster.to_array(rows)

    n_wd_csr = csr_matrix( ( data, (rows, cols) ), shape=(V, D), dtype=np.int64, copy=False)
    
    return n_wd_csr, dict_w_iw

# def fill_sparse(i):#sample):
#     try:
#         #print(i)
#         sample=data[i,:]
#         gene_idx_1=[]
#         gene_idx_2=[]
#         vals=[]
#         nz=np.nonzero(sample)
#         nz=nz[0]
#         for k in range(len(nz)-1):
#             t=len(nz)-1 - k
#             temp=[[nz[k]]*t]
#             temp2=nz[k+1:len(nz)]
#             gene_idx_2.append(temp2)
#             temp=list(itertools.chain(*temp))
#             gene_idx_1.append(temp)
#             vals.append(0.5*rho[temp, temp2]*(sample[temp]+sample[temp2]))  
#         gene_idx_1=list(itertools.chain(*gene_idx_1))
#         gene_idx_2=list(itertools.chain(*gene_idx_2))
#         vals=list(itertools.chain(*vals))
#         u=csr_matrix((vals, (gene_idx_1, gene_idx_2)), shape=(data.shape[1], data.shape[1]), dtype=np.float32)
        
#     except Exception as e:
#         print(e)


