import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import os
import random
import torch
import argparse
from scipy.io import loadmat
import sys  
sys.path.insert(0, './ETM/')
import data

def get_knn_indices(adata: ad.AnnData,
    use_rep: str = "delta",
    n_neighbors: int = 25,
    random_state: int = 0,
    calc_knn: bool = True
) -> np.ndarray:

    if calc_knn:
        assert use_rep == 'X' or use_rep in adata.obsm, f'{use_rep} not in adata.obsm and is not "X"'
        neighbors = sc.Neighbors(adata)
        neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True, use_rep=use_rep, random_state=random_state, write_knn_indices=True)
        adata.obsp['distances'] = neighbors.distances
        adata.obsp['connectivities'] = neighbors.connectivities
        adata.obsm['knn_indices'] = neighbors.knn_indices
        adata.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'knn_indices_key': 'knn_indices',
            'params': {
                'n_neighbors': n_neighbors,
                'use_rep': use_rep,
                'metric': 'euclidean',
                'method': 'umap'
            }
        }
    else:
        assert 'neighbors' in adata.uns, 'No precomputed knn exists.'
        assert adata.uns['neighbors']['params']['n_neighbors'] >= n_neighbors, f"pre-computed n_neighbors is {adata.uns['neighbors']['params']['n_neighbors']}, which is smaller than {n_neighbors}"

    return adata.obsm['knn_indices']

def load_cluster_labels(fname):
    clust=pd.read_csv(fname)
    labels ={}
    for i in range(len(clust.index)):
        labels[clust.loc[i][0]]=clust.loc[i][1]
    f=np.unique(list(labels.values()))
    #f
    lab=pd.DataFrame.from_dict(labels, orient='index', columns=['cluster'])
    idx_list=[lab['cluster']==item for item in f]
    for i in range(len(idx_list)):
        lab.loc[idx_list[i], :] = lab.loc[idx_list[i], :].replace([f[i]],i)
    lab.head()
    return list(lab['cluster'])

def evaluate(adata, cell_type_col, clustering_method=None, leiden_resolutions=None, save_as='clustering_output', save_scores=False):

    from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples
    from math import inf
    
    if clustering_method is None:
        print('No clustering method specified. Applying all methods: k-means, Aglomerative, and Leiden')
        clustering_method=['kmeans', 'AG', 'leiden']
        if leiden_resolutions is None:
            print('No resolution given for Leiden. Using np.arange(0.1,1.,0.01) as default') 
            leiden_resolutions=np.arange(0.1,1.,0.01)

    if 'theta' in adata.obsm.keys():
        data=adata.obsm['theta']
        rep='theta'
    else: 
        data=adata.X
        rep='X'

    scores={}
    if 'kmeans' in clustering_method:
        #TODO: Separate method to get scores
        clust_km=KMeans(n_clusters=len(np.unique(adata.obs[cell_type_col]))).fit(data)
        nmi_kmeans=normalized_mutual_info_score(adata.obs[cell_type_col], clust_km.labels_)
        ari_kmeans=adjusted_rand_score(adata.obs[cell_type_col], clust_km.labels_)
        print(f'kMeans         :\tARI: {ari_kmeans:7.4f}\tNMI: {nmi_kmeans:7.4f}')
        scores['kmeans']={'ari':ari_kmeans, 'nmi':nmi_kmeans}
        adata.obs['kmeans']=pd.Categorical(clust_km.labels_)
        
    if 'AG' in clustering_method:
        clust_ag=AgglomerativeClustering(n_clusters=len(np.unique(adata.obs[cell_type_col]))).fit(data)
        nmi_ag=normalized_mutual_info_score(adata.obs[cell_type_col], clust_ag.labels_)
        ari_ag=adjusted_rand_score(adata.obs[cell_type_col], clust_ag.labels_)
        adata.obs['ag']=pd.Categorical(clust_ag.labels_)
        print(f'Agglomerative  :\tARI: {ari_ag:7.4f}\tNMI: {nmi_ag:7.4f}')
        scores['AG']={'ari':ari_ag, 'nmi':nmi_ag}
        
    if 'leiden' in clustering_method:
        get_knn_indices(adata, use_rep=rep)
        best_res, best_ari, best_nmi = None, -inf, -inf
        resolutions=leiden_resolutions
        for res in resolutions:
            col = f'{res}'
            sc.tl.leiden(adata, resolution=res, key_added=col)
            ari = adjusted_rand_score(adata.obs['assigned_cluster'], adata.obs[col])
            nmi = normalized_mutual_info_score(adata.obs['assigned_cluster'], adata.obs[col])
            n_unique = adata.obs[col].nunique()
            if ari > best_ari:
                best_res = res
                best_ari = ari
            if nmi > best_nmi:
                best_nmi = nmi
        print(f'Leiden         :\tARI: {best_ari:7.4f}\tNMI: {best_nmi:7.4f}\tResolution: {best_res:7.4f}')
        scores['Leiden']={'ari':best_ari, 'nmi':best_nmi, 'res':best_res}
        
              
    if save_scores:
        np.save(save_as, scores)
        print(f'Saving scores as {save_as}')
    return scores

def load_etm_model(sourceDir):
    try:
        files = os.listdir(sourceDir)
        random.shuffle(files)
    
        model=None
        #load all the data   
        for fname in files:
            try:
                with open(os.path.join(sourceDir, fname), 'rb') as f:
                    model = torch.load(f)
            except:
                continue
        if model==None:
            print('Unable to load model: No matching file found in the given path.')
            print('Verify file name and path')
        else:
            print('Model file loaded.')
        fpath=sourceDir+'/tf_idf_doc_terms_matrix_time_window_1'
        tf_idf = loadmat(fpath)['doc_terms_matrix']
        tf_idf= tf_idf.todense()
        theta, _ = model.get_theta(torch.tensor(tf_idf).float())
        theta=theta.detach().numpy()
        print('Embeddings (theta): loaded')
        return model, theta
    except FileNotFoundError as e:
        print(e)
        return None, None
    
def get_model_vocab(sourceDir):
    vocab, _, _, _, _ = data.get_data(doc_terms_file_name=sourceDir+"/tf_idf_doc_terms_matrix_time_window_1",
                                                               terms_filename=sourceDir+"/tf_idf_terms_time_window_1")
    vocab=[str(w.split()[0]) for w in vocab]
    return vocab

          
def show_topics(model, num_topics, sourceDir):
    parser = argparse.ArgumentParser(description='The Embedded Topic Model')
    args = parser.parse_args("")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## get data
    # 1. vocabulary
    vocab, training_set, valid, test_1, test_2 = data.get_data(doc_terms_file_name=sourceDir+"/tf_idf_doc_terms_matrix_time_window_1",
                                                               terms_filename=sourceDir+"/tf_idf_terms_time_window_1")
    vocab_size = len(vocab)
    args.vocab_size = vocab_size
    print('Vocabulary size: '+str(vocab_size))
    # 1. training data
    args.num_docs_train = training_set.shape[0]

    # 2. dev set
    args.num_docs_valid = valid.shape[0]

    # 3. test data
    args.num_docs_test = test_1.shape[0] + test_2.shape[0]

    args.num_docs_test_1 = test_1.shape[0]
    args.num_docs_test_2 = test_2.shape[0]
    #All the following values are set by default in the original script
    args.eval_batch_size = 100
    args.batch_size = 1000
    args.bow_norm=1
    args.num_topics=num_topics
    args.num_words=10

    with torch.no_grad():
            ## get document completion perplexities
            test_ppl = model.evaluate(args, 'val', training_set, vocab,  test_1, test_2)#, args.tc, args.td)
            ## get most used topics
            indices = torch.tensor(range(args.num_docs_train))
            indices = torch.split(indices, args.batch_size)
            thetaAvg = torch.zeros(1, args.num_topics).to(device)
            theta_weighted_average = torch.zeros(1, args.num_topics).to(device)
            cnt = 0
            for idx, indice in enumerate(indices):
                data_batch = data.get_batch(training_set, indice, device)
                sums = data_batch.sum(1).unsqueeze(1)
                cnt += sums.sum(0).squeeze().cpu().numpy()
                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                theta, _ = model.get_theta(normalized_data_batch)
                thetaAvg += theta.sum(0).unsqueeze(0) / args.num_docs_train
                weighed_theta = sums * theta
                theta_weighted_average += weighed_theta.sum(0).unsqueeze(0)
                if idx % 100 == 0 and idx > 0:
                    print('batch: {}/{}'.format(idx, len(indices)))
            theta_weighted_average = theta_weighted_average.squeeze().cpu().numpy() / cnt
            print('\nThe 10 most used topics are {}'.format(theta_weighted_average.argsort()[::-1][:10]))

            ## show topics
            beta = model.get_beta()
            #topic_indices = list(np.random.choice(args.num_topics, 10)) # 10 random topics
            print('\n')
            for k in range(args.num_topics):#topic_indices:
                gamma = beta[k]
                top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
                topic_words = [str(vocab[a].split()[0]) for a in top_words]
                print('Topic {}: {}'.format(k, topic_words))