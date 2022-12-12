import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from scipy.io import savemat
#convert binary gene2vec to matrix txt

def load_embeddings(file_name):
    model = KeyedVectors.load(file_name)
    wordVector = model.wv
    vocabulary, wv = zip(*[[word, wordVector[word]] for word, vocab_obj in wordVector.vocab.items()])
    return np.asarray(wv), vocabulary

def outputTxt (embeddings_file, dimension):
    embeddings_file = embeddings_file  # gene2vec file address
    wv, vocabulary = load_embeddings(embeddings_file)
    index = 0
    matrix_txt_file = embeddings_file+".txt"  # gene2vec matrix txt file address
    with open(matrix_txt_file, 'w') as out:
        out.write(str(len(vocabulary)) + "\t "+str(dimension)+ " ")
        out.write("\n")
        for ele in wv[:]:
            out.write(str(vocabulary[index]) + " ")
            index = index + 1
            for elee in ele:
                out.write(str(elee) + " ")
            out.write("\n")
    out.close()

def outputPkl (embeddings_file, export_dir, dims=100):
    embeddings_file = embeddings_file  # gene2vec file address
    wv, vocabulary = load_embeddings(embeddings_file)    
    embeddings_matrix = np.zeros(shape=(len(vocabulary),dims))

    for i in range(len(wv)):
        vector = wv[i]
        embeddings_matrix[i,:] = vector
    with open(export_dir+'gene_embeddings', 'wb') as file_path:
        np.save(file_path, embeddings_matrix)
    with open(export_dir+'tf_idf_terms_time_window_1', 'wb') as file_path:
        #np.save(file_path, vocabulary)
        savemat(file_path, {"terms" : vocabulary}, do_compression=True)

# scores={}
# for i in range(10):
#     emb_w2v_file = "../mouseP/embeddings/gene2vec_dim_300_iter_"+str(i+1)+".txt"
#     scores[i]=targetFunc(emb_w2v_file)
    
#     "../mouseP/mouse_gene_list.csv"