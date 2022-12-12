# getm
Embedded gene and topic modelling for scRNA-seq analysis

# Project Title


Topic models are gaining attention within the bioinformatics community as a powerful tool that delivers more intuitive data
representations that facilitate interpretability. However, it has only recently been adopted as a tool for single-cell transcriptome analysis.
The interpretation can be difficult as topics are an abstract representation of the whole data. Recent advances in Natural Language
Processing include the introduction of neural word embeddings and embedded topic models. Both represent words and topics as
vectors in a Euclidean space. The introduction of neural word embeddings, and Word2Vec in particular, represented a breakthrough in
NLP whose usefulness is actively explored beyond this field. We propose combining these two methods for scRNA-seq analysis. The
essential characteristic of the neural word embeddings that we employ is that the method learns the vector representations based on
word co-occurrence, that is, words that appear within the same context in a phrase. We leverage this characteristic to exploit gene
correlation to codify co-regulation information into the embeddings. Additionally, we employ an information-theoretic approach to detect
uninformative genes. The proposed framework produces high-quality, easy-to-interpret topic representations and provides a direct link
between cell types and functional sets of genes.

This framework employs the following methods and adapts the respective python code:

-ETM[1]

-Gene2Vec[2]

-StopWords[3]

Refer to the included Jupyter notebooks for an example of the pipeline and model analysis.


## References
[1] Adji B Dieng, Francisco JR Ruiz, and David M Blei. Topic
modeling in embedding spaces. Transactions of the Association for
Computational Linguistics, 8:439–453, 2020.

[2] Jingcheng Du, Peilin Jia, Yulin Dai, Cui Tao, Zhongming Zhao, and
Degui Zhi. Gene2vec: distributed representation of genes based on
co-expression. BMC genomics, 20(1):7–15, 2019.

[3] Martin Gerlach, Hanyu Shi, and Luis A Nunes Amaral. A
universal information theoretic approach to the identification of
stopwords. Nature Machine Intelligence, 1(12):606–612, 2019.
## Authors

- [@oscar-rt](https://www.github.com/oscar-rt)



