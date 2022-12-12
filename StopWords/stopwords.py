#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 22:26:06 2021

@author: orod0002
"""
import os,sys
import logging
import numpy as np
import pandas as pd
from filter_words import run_stopword_statistics
from data_io import texts_nwd_csr
from filter_words import *
## save the stopword statistics for all corpora
def apply_stopwords(file_path, output_path, output_name):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #Data load
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #src_dir = os.path.join(os.pardir, 'src')
    #sys.path.append(src_dir)
    #path_read = os.path.join(os.pardir,os.pardir,'data','data-rna_2019-05-30')
    #filename = os.path.join(path_read,fname_read)
    filename=file_path#'./data/mouse_pancreas_UMI.csv'#Genes x Cells
    #filename='../data/mouse_pancreas_UMI.csv'#Genes x Cells
    #df_wd = pd.read_csv(filename,index_col=0,na_values=0).dropna(how='all',axis=0).to_sparse() ## contains the data
    df_wd = pd.read_csv(filename,index_col=0,na_values=0).dropna(how='all',axis=0)#.astype(pd.SparseDtype(int, fill_value=1))
    df_wd.fillna(0, inplace=True)
    df_wd = df_wd.astype(pd.SparseDtype(int, fill_value=1))

    N_s = 1000
    ## make csr matrix
    n_wd_csr = csr_matrix(df_wd.values).astype('int')
    V,D = n_wd_csr.shape

    logging.info('Computing entropy measure')
    result_H = nwd_H_shuffle(n_wd_csr,N_s=N_s)

    ## get tfidf
    arr_tfidf_w = nwd_tfidf_csr(n_wd_csr)


    ## make dataframe
    df=pd.DataFrame(index = df_wd.index )

    df['F'] = result_H['F-emp']
    df['I'] = result_H['H-null-mu'] - result_H['H-emp']
    df['tfidf'] = arr_tfidf_w
   
    ## get entropy and random entropy too
    logging.info('Collecting data...')
    df['H'] = result_H['H-emp']
    df['H-tilde'] =  result_H['H-null-mu']
    df['H-tilde_std'] =  result_H['H-null-std']
    df['N'] = np.array(n_wd_csr.sum(axis=1))[:,0] ## number of counts
    fname_save = output_name+'_Ns%s.csv'%(N_s) #'mouse_pancreas_stopword-statistics_Ns%s.csv'%(N_s)
    logging.info('Saving to CVS file: '+fname_save)
    path_save = output_path#'./output/mouse_pancreas_stopwords'
    filename = os.path.join(path_save,fname_save)
    df.to_csv(filename)
    print("Done!")
