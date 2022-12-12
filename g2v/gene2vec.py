#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Oscar Rodriguez
"""
import gensim, logging
import os
import random
import datetime
import generateMatrix as gM
import argparse

def apply_gene2vec(gene_pairs_file, embedding_dimension=300, sourceDir='./', export_dir='./output'):
    ending_pattern = 'txt' 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("start!")
    # training file format:
    #   TOX4 ZNF146
    #   TP53BP2 USP12
    #   TP53BP2 YRDC

    #num_db = 0
    #files = os.listdir(sourceDir)
    #size = len(files)
    gene_pairs = list()
    #random.shuffle(files)

    #load all the data
#     for fname in files:
#         if not fname.endswith(ending_pattern):
#             continue
#         num_db = num_db + 1
    now = datetime.datetime.now()
    print(now)
    logging.info("Gene pairs file: "+ gene_pairs_file)
    f = open(os.path.join(sourceDir, gene_pairs_file), 'r', encoding='windows-1252')
    for line in f:
        gene_pair = line.strip().split()
        gene_pairs.append(gene_pair)
    f.close()

    current_time = datetime.datetime.now()
    print(current_time)
    logging.info("shuffle start " + str(len(gene_pairs)))
    random.shuffle(gene_pairs)
    current_time = datetime.datetime.now()
    print(current_time)
    logging.info("shuffle done " + str(len(gene_pairs)))

    ####training parameters########
    dimension = embedding_dimension  # dimension of the embedding
    num_workers = 32  # number of worker threads
    sg = 1  # sg =1, skip-gram, sg =0, CBOW
    max_iter = 10  # number of iterations
    window_size = 1  # The maximum distance between the gene and predicted gene within a gene list
    txtOutput = True
    
    #if current_iter == 1:
    model = gensim.models.Word2Vec(gene_pairs, size=dimension, window=window_size, min_count=1, workers=num_workers, iter=1, sg=sg)
    model.save(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_"+str(1))
    if txtOutput:
        gM.outputTxt(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_"+str(1), dimension)
    logging.info("gene2vec dimension "+ str(dimension) +" iteration "+ str(1)+ " done")
    del model
    for current_iter in range(2,max_iter+1):     
        #else:
        current_time = datetime.datetime.now()
        print(current_time)
        logging.info("shuffle start " + str(len(gene_pairs)))
        random.shuffle(gene_pairs)
        current_time = datetime.datetime.now()
        print(current_time)
        logging.info("shuffle done " + str(len(gene_pairs)))

        logging.info("gene2vec dimension " + str(dimension) + " iteration " + str(current_iter) + " start")
        model = gensim.models.Word2Vec.load(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter-1))
        model.train(gene_pairs,total_examples=model.corpus_count,epochs=model.iter)
        model.save(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter))
        if txtOutput:
            gM.outputTxt(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter), dimension)
        logging.info("gene2vec dimension " + str(dimension) + " iteration " + str(current_iter) + " done")
        del model

    gM.outputPkl(export_dir+"gene2vec_dim_"+str(dimension)+"_iter_10", sourceDir, dims=dimension)
    logging.info("Saving vocabulary and embeddings: Done")