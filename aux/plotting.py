import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
import scanpy as sc
import sys
#sys.path.insert(0,'./aux/')
from aux import *
import umap
import seaborn as sns
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
import warnings;
warnings.filterwarnings('ignore');

def plot_HvsN(df,thresh, n_samples, hkg_list=None, name=None):
        # number of pt for column in latex-document
    fig_width_pt = 246  # single-column:510, double-column: 246; Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.1/72.27 # Convert pt to inches
    width_vs_height = (np.sqrt(5)-1.0)/1.75#(np.sqrt(5)-1.0)/2.0 # Ratio of height/width [(np.sqrt(5)-1.0)/2.0]
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = width_vs_height*fig_width  # height in inches
    fig_size = [fig_width,fig_height]

    df = df.sort_values(by='N', ascending=0)

    I_thres = thresh
    flt_df=df[df['I']>I_thres]
    flt_df_comp=df[df['I']<I_thres]

    word_count_array = flt_df['N'].values
    info_mean_array = flt_df['H-tilde'].values
    info_real_array = flt_df['H'].values
    yerr_array = flt_df['H-tilde_std'].values * 2

    word_count_array_comp = flt_df_comp['N'].values
    info_mean_array_comp = flt_df_comp['H-tilde'].values
    info_real_array_comp = flt_df_comp['H'].values
    yerr_array_comp = flt_df_comp['H-tilde_std'].values * 2

    D = n_samples
    upper_limit = np.log2(D)
    cmap = 'tab10'
    ccolors = plt.get_cmap(cmap)(np.arange(10, dtype=int))
    set_blue = ccolors[0]
    set_orange = ccolors[1]
    set_green = ccolors[2]
    set_gray = ccolors[7]

    plt.close('all')
    fig_size=(15,15)
    # here you can set the parameters of the plot (fontsizes,...) in pt
    params = {'backend': 'ps',
              'axes.titlesize':30,
              'axes.labelsize': 25,
    #            'text.fontsize': 15,
              'legend.fontsize': 20,
    #            'figtext.fontsize': 15,
              'xtick.labelsize': 17,
              'ytick.labelsize': 17,

              'text.usetex': True,
              'ps.usedistiller' : 'xpdf',
              'figure.figsize': fig_size,
              #'text.latex.unicode':True,
              'text.latex.preamble': [r'\usepackage{bm}'],

              'xtick.direction':'out',
              'ytick.direction':'out',

              'axes.spines.right' : False,
              'axes.spines.top' : False
             }
    plt.rcParams.update(params)
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=fig_size)#(15,15))
    x_annot1 = -0.2
    y_annot1 = 1.06
    x_annot2 = -0.1
    y_annot2 = 1.06
    x_annot_text = 0.05
    y_annot_text = 0.95
    ymin,ymax=-1,upper_limit+0.1
    if hkg_list is not None:
        chosen_word_list = hkg_list
        chosen_word_dict = {}
        for i in chosen_word_list:
            chosen_word_dict[i] = {
                'word_count': float(df.loc[i]['N']),
                'info_real': float(df.loc[i]['H']),
            }
        x_factor_list = [.9, .9, .2,  .5,]
        y_factor_list = [-1.0, -1.0, 0.7, -1.1,]
        for tmp_id, tmp_word in enumerate(chosen_word_list):
            ax1.scatter(chosen_word_dict[tmp_word]['word_count'], chosen_word_dict[tmp_word]['info_real'], marker='x', facecolors='r', zorder=3)#, s=84)

    ax1.plot(word_count_array, info_real_array, ms=2, lw=0, marker='o', color=set_green, alpha=0.5, rasterized=1,zorder=1)
    ax1.plot(word_count_array_comp, info_real_array_comp, ms=2, lw=0, marker='o', color=set_blue, alpha=0.5, rasterized=1,zorder=1)
    ax1.errorbar(word_count_array, info_mean_array, yerr=yerr_array, color=set_orange, lw=1, alpha=0.9, rasterized=1,zorder=2)
    ax1.plot(word_count_array, info_mean_array-I_thres, lw=1, color='k', linestyle='--', alpha=0.9, rasterized=1,zorder=2)

    yann=info_mean_array-I_thres
    gene_percentage=np.count_nonzero(df['I']>I_thres)/len(df['I'])*100
    note='$I(w)$ threshold $I$*='+str(I_thres)+', $\%$genes: '+str(f'{gene_percentage:.2f}')#+'$'

    ax1.set_xscale("log")
    ax1.set_xlabel('Frequency $n(w)$')
    ax1.set_ylabel('$H(w|C)$')

    plt.title(note)
    if name is not None:
        plt.savefig('./Plots/'+name+'.png', dpi=250, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    
def plot_IvsN(df, thres=0.1, name=None):
    ###########
    ## Setup ##
    ###########
    # number of pt for column in latex-document
    fig_width_pt = 510/2  # single-column:510, double-column: 246; Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.1/72.27 # Convert pt to inches
    width_vs_height = (np.sqrt(5)-1.0)/1.75#(np.sqrt(5)-1.0)/2.0 # Ratio of height/width [(np.sqrt(5)-1.0)/2.0]
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = width_vs_height*fig_width  # height in inches
    fig_size = [fig_width,fig_height]

    # here you can set the parameters of the plot (fontsizes,...) in pt
    params = {'backend': 'ps',
              'axes.titlesize':30,
              'axes.labelsize': 25,
    #            'text.fontsize': 15,
              'legend.fontsize': 20,
    #            'figtext.fontsize': 15,
              'xtick.labelsize': 17,
              'ytick.labelsize': 17,

              'text.usetex': True,
              'ps.usedistiller' : 'xpdf',
              'figure.figsize': fig_size,
              #'text.latex.unicode':True,
              'text.latex.preamble': [r'\usepackage{bm}'],

              'xtick.direction':'out',
              'ytick.direction':'out',

              'axes.spines.right' : False,
              'axes.spines.top' : False
             }
    plt.rcParams.update(params)

    set_b = 0.22 # set bottom
    set_l = 0.1 # set left
    set_r = 0.925 # set right
    set_hs = 0.2 # set horizontal space
    set_vs = 0.25 # set vertical space

    set_ms = 0.0 # set marker size
    set_lw = 2.5 # set line width
    set_alpha = 0.8
    x_data_array = df['N'].values
    y_data_array = df['I'].values

    log10_x_data_array = np.log10(x_data_array)
    heatmap, xedges, yedges = np.histogram2d(log10_x_data_array, y_data_array, bins=50)
    heatmap_log = np.log10(heatmap)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    set_x_min = -.5
    set_x_max = 6.5
    set_y_min = -1
    set_y_max = df['I'].max()+0.5#8

    plt.close('all')
    fig = plt.figure(figsize=(15,15))#fig_size)

    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2)

    t= np.arange(1000)/100.
    x = np.sin(np.pi**t)
    y = np.cos(np.pi**t)
    z = np.cos(0.1*np.pi**t)

    ## subplot 1
    bin_1 = 30
    ax1.hist(log10_x_data_array, bin_1, histtype='step')
    ax1.set_yscale("log")
    #ax1.set_ylim(-100, 50000)
    ax1.set_xlim(set_x_min, set_x_max)
    #ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax1.set_xticklabels([])
    #ax1.set_yticks([10**0,10**2,10**4])
    ax1.set_ylabel('Count')

    ## subplot 2
    cax = ax2.imshow(heatmap_log.T, extent=extent, origin='lower', cmap='gnuplot2_r', aspect="auto")
    line,=ax2.plot([set_x_min, set_x_max], [thres,thres],  lw=1, ls='--',  color='black')
    ax2.set_ylim(set_y_min, set_y_max)
    ax2.set_ylabel('Information $I(w)$')
    ax2.set_xlabel('Frequency $n(w)$')
    ax2.set_xlim(set_x_min, set_x_max)
    ax2.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax2.set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$',  ])
    line.set_label('I(w) threshold='+str(thres))
    ax2.legend(loc='lower right')   
    cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3, 4, 5, 6, 7])
    cbar.mappable.set_clim(-1.0, 8.0)
    cbar.ax.set_yticklabels(['$10^0$', '$10^1$','$10^2$','$10^3$','$10^4$', '$10^5$', '$10^6$', '$10^7$'])  # horizontal colorbar

    ## subplot 3
    bin_3 = 30
    ax3.hist(y_data_array, bin_3,orientation='horizontal', histtype='step' )
    ax3.set_xscale("log")
    #ax3.set_xticks([10**0,10**2,10**4])
    ax3.set_ylim(set_y_min, set_y_max)
    ax3.set_yticklabels([])
    ax3.set_xlabel('Count')

    ###########
    # end
    ###########
    x_annot = -0.3
    y_annot = 1.05
    ax1.annotate(r'\textbf{A}',xy=(x_annot,y_annot),xycoords = 'axes fraction')
    plt.subplots_adjust(wspace=.2, hspace=0.3)
    if name is not None:
        plt.savefig('./Plots/'+name+'.png', dpi=250, bbox_inches='tight')
    plt.show()
    plt.close()
    
def load_embedding(filename):
    geneList = list()
    vectorList = list()
    f = open(filename)
    for line in f:
        values = line.split()
        gene = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        geneList.append(gene)
        vectorList.append(vector)
    f.close()
    genes=pd.DataFrame(geneList[1:], columns=['gene'], index=range(len(geneList)-1))
    vectors=np.stack(vectorList[1:])
    return vectors, genes
    

def plot_scanpy_embeddings(fpath):
    emb, genes = load_embedding(fpath)   
    adata=ad.AnnData(obs=genes)
    embedding_key='g2v'
    adata.obsm[embedding_key]=emb
    #get_knn_indices(adata, use_rep='g2v')
    neighbors = sc.Neighbors(adata)
    neighbors.compute_neighbors(n_neighbors=15, knn=True, use_rep='g2v', random_state=0, write_knn_indices=True)
    adata.obsp['distances'] = neighbors.distances
    adata.obsp['connectivities'] = neighbors.connectivities
    adata.obsm['knn_indices'] = neighbors.knn_indices
    adata.uns['neighbors'] = {
        'connectivities_key': 'connectivities',
        'distances_key': 'distances',
        'knn_indices_key': 'knn_indices',
        'params': {
            'n_neighbors': 15,
            'use_rep': 'g2v',
            'metric': 'euclidean',
            'method': 'umap'
        }
    }
    sc.tl.umap(adata) #Computes UMAP embedding
    sc.pl.umap(adata)# Plots UMAP

    
def get_gene_embeddings_umap(fpath):
    emb, genes = load_embedding(fpath)
    embedding = umap.UMAP(n_neighbors=15,
                      min_dist=0.3,
                      random_state=42).fit_transform(emb)
    embedding = pd.DataFrame(embedding)
    embedding.columns=['UMAP1','UMAP2']
    return embedding

def get_gene_embeddings_umap2(fpath):
    emb, genes = load_embedding(fpath)
    embedding = umap.UMAP(n_neighbors=15,
                      min_dist=0.3,
                      random_state=42,
                      metric='euclidean').fit_transform(emb)
    embedding = pd.DataFrame(embedding)
    embedding.columns=['UMAP1','UMAP2']
    return embedding
    
def apply_umap(df, metric='euclidean'):
    embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3, metric=metric,
                      random_state=1).fit_transform(df)
    embedding = pd.DataFrame(embedding)

    embedding.columns=['UMAP1','UMAP2']
    sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,
               fit_reg=False,legend=False,scatter_kws={"s": 1,"color":"green", "alpha": 0.45})
    plt.show()
    df['UMAP1']=list(embedding['UMAP1'])
    df['UMAP2']=list(embedding['UMAP2'])
    #return embedding

def plot_umap(umap_emb):
    sns.lmplot(x='UMAP1', y='UMAP2',data=umap_emb,
               fit_reg=False,legend=False,scatter_kws={"s": 1,"color":"steelblue"})
    plt.grid(False)
    plt.show()
    
def plot_gene_clusters(embedding, name=None):
    ap = cluster.AffinityPropagation(damping = 0.9, random_state=42).fit(np.array(embedding.iloc[:,-2:])) 
    y_pred=ap.labels_
    embedding["Cluster"]=y_pred
    f=sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,hue="Cluster",
               fit_reg=False,legend=False,scatter_kws={'s':5})
    for i in list(set(y_pred)):
        plt.annotate(i, 
                     embedding.loc[embedding['Cluster']==i,['UMAP1','UMAP2']].mean(),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=15, weight='bold')
    plt.title(f'Total inferred clusters: {len(np.unique(y_pred))}')

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    if name is not None:
        plt.savefig('./Plots/gene_cluster_'+name+'.png', dpi=250, bbox_inches='tight')
    plt.show()
    return y_pred


def add_topics_to_embeddings(gene_embeddings, model):
    beta=model.get_beta()
    beta=beta.detach().numpy().T
    for i in range(beta.shape[1]):
        gene_embeddings['topic'+str(i)]=beta[:,i]

def get_topic_embeddings_umap(model):
    topic_embeddings=model.alphas.weight.detach().numpy() # <- TOPIC EMBEDDINGS
    topic_umap = umap.UMAP(n_neighbors=15,
                     min_dist=0.3,
                     random_state=42,
                     metric='correlation').fit_transform(topic_embeddings)
    topic_umap = pd.DataFrame(topic_umap)
    topic_umap.columns=['UMAP1','UMAP2']
    return topic_umap


def get_topic_embeddings_umap_euc(model):
    topic_embeddings=model.alphas.weight.detach().numpy() # <- TOPIC EMBEDDINGS
    topic_umap = umap.UMAP(n_neighbors=15,
                     min_dist=0.3,
                     random_state=42,
                     metric='euclidean').fit_transform(topic_embeddings)
    topic_umap = pd.DataFrame(topic_umap)
    topic_umap.columns=['UMAP1','UMAP2']
    return topic_umap
        
    
def plot_topic_gene_proportion(gene_embeddings, model, topic_umap, topic=None, name=None):
    if 'topic0' not in gene_embeddings.columns:
        print('Adding topics')
        add_topics_to_embeddings(gene_embeddings, model)
        
    norm = plt.Normalize(gene_embeddings['topic'+str(topic)].min(), gene_embeddings['topic'+str(topic)].max())
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    ax=sns.lmplot(x='UMAP1', y='UMAP2',data=gene_embeddings,hue='topic'+str(topic),palette="Blues",
                 fit_reg=False,legend=False,height=5, aspect=1.2,scatter_kws={"s": 3})

    ax.figure.colorbar(sm)
    plt.scatter(x=topic_umap.iloc[topic,0], y=topic_umap.iloc[topic,1], marker='D', color='r', label='Topic '+str(topic))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Gene embeddings. Topic {topic} enrichment')
    if name is not None:
        plt.savefig('./Plots/gene_topic_intensity'+name+'.png', dpi=250, bbox_inches='tight')
    plt.show()
    
def top_genes_topic_proportion(topic, top_gene_embeddings):

    ax=sns.lmplot(x='UMAP1', y='UMAP2',data=top_gene_embeddings,hue=str(topic),palette="Blues",
                 fit_reg=False,legend=False,height=5, aspect=1.2,scatter_kws={"s": 3})

    norm = plt.Normalize(top_gene_embeddings[str(topic)].min(), top_gene_embeddings[str(topic)].max())
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm)
    plt.title(f'Top genes embeddings. Topic {topic} intensity')
    plt.show() 
    
def topic_intensity_heatmap(theta, pred_labels, name=None):
    topic_matrix=pd.DataFrame(theta)
    topic_matrix['Cluster']=pred_labels
    cluster_topic = topic_matrix.groupby(['Cluster']).mean()
    f=sns.heatmap(cluster_topic,cbar_kws={"shrink": 0.7}, cmap='mako', xticklabels=1, yticklabels=1, square=False, linewidths=0.05)
    plt.xlabel("Topic", fontsize = 20)
    plt.ylabel("Cluster", fontsize = 20)
    if name is not None:
        plt.savefig('./Plots/cell_topic_heatmap_'+name+'.png', dpi=250, bbox_inches='tight')
    plt.show()
    
    
def topic_intensity_scatter(topic, cell_umap, theta, name=None):
    cell_embedding=cell_umap
    col='topic'+str(topic)
    cell_embedding[col]=theta[:,topic]
    ax=sns.lmplot(x='UMAP1', y='UMAP2',data=cell_embedding,hue=col,palette="Purples",
                 fit_reg=False,legend=False,height=5, aspect=1.2,scatter_kws={"s": 3})
    norm = plt.Normalize(cell_embedding['topic'+str(topic)].min(), cell_embedding['topic'+str(topic)].max())
    sm = plt.cm.ScalarMappable(cmap="Purples", norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Cell embeddings. Topic {topic}')
    if name is not None:
        plt.savefig('./Plots/cell_topic_intensity_'+name+'.png', dpi=250, bbox_inches='tight')
    plt.show()

    
def plot_cell_agglomerative_clustering(cluster_labels, theta, cell_umap, fpath=None):
    
    default_base = {'n_neighbors': 15,
                    'n_clusters': len(np.unique(cluster_labels['assigned_cluster']))}
    params = default_base.copy()
    connectivity = kneighbors_graph(theta, 
                                    n_neighbors=params['n_neighbors'], 
                                    include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], 
                                           linkage='ward',connectivity=connectivity)
    ward.fit(theta)
    y_pred = ward.labels_
    cluster_size={}
    for i in range(params['n_clusters']):
        cluster_size[i]=y_pred.tolist().count(i)
    clusters = [i for i in cluster_size.keys() if cluster_size[i]>=20]    
    cell_umap["pred cluster"]=y_pred
    f=sns.lmplot(x='UMAP1', y='UMAP2',data=cell_umap,hue="pred cluster", 
               fit_reg=False,legend=False,scatter_kws={'s':1})
    for i in clusters:
        plt.annotate(i, 
                     cell_umap.loc[cell_umap['pred cluster']==i,['UMAP1','UMAP2']].mean(),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=15, weight='bold')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    if fpath is not None:
        plt.savefig(fpath+'.png', dpi=250, bbox_inches='tight')
    plt.show()
    return y_pred

    
def plot_cells_umap(theta, metric='euclidean', min_dist=0.3, n_neighbors=5, name=None):
    # Cell plot using theta
    cell_embedding = umap.UMAP(n_neighbors=n_neighbors,
                         min_dist=min_dist, metric=metric,
                         random_state=1).fit_transform(theta)
    cell_embedding = pd.DataFrame(cell_embedding)
    cell_embedding.columns=['UMAP1','UMAP2']
    f=sns.lmplot(x='UMAP1', y='UMAP2',data=cell_embedding,
               fit_reg=False,legend=False,scatter_kws={"s": 1,"color":"purple"})
    plt.grid(False)
    if name is not None:
        plt.savefig('./Plots/cells_umap_'+name+'.png', dpi=250, bbox_inches='tight')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return cell_embedding

def plot_cell_types_umap(cells_umap, cluster_labels, fpath=None):
    cells_umap['Cell type']=cluster_labels['assigned_cluster']
    scatter=sns.lmplot(x='UMAP1', y='UMAP2',data=cells_umap, palette='Paired',
               fit_reg=False, legend=False, scatter_kws={"s": 3}, hue='Cell type', facet_kws={'legend_out': True})
    plt.legend(fontsize = 12, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=4)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    if fpath is not None:
        plt.savefig(fpath+'.png', dpi=250, bbox_inches='tight')
    plt.show()

def gene_topic_intensity_heatmap(beta, pred_labels, name=None):
    topic_matrix=pd.DataFrame(beta)
    topic_matrix['Cluster']=pred_labels
    cluster_topic = topic_matrix.groupby(['Cluster']).mean()
    cmap=sns.cubehelix_palette(rot=-.55, light=0.98, as_cmap=True)
    f=sns.heatmap(cluster_topic, cmap=cmap, xticklabels=1, yticklabels=1, linewidths=0.5, square=False, cbar_kws={"shrink": 0.7})
    plt.xlabel("Topic", fontsize = 20)
    plt.ylabel("Cluster", fontsize = 20)
    if name is not None:
        plt.savefig('./Plots/gene_cluster_topic_heatmap_'+name+'.png', dpi=250, bbox_inches='tight')
    plt.show()

def gene_topic_proportion_heatmap(beta, n_genes=50):
    # Gene clustering
    n_components=50
    new_w = np.zeros((n_genes*n_components,n_components))
    new_w = pd.DataFrame(new_w)
    genes = []
    for i in range(n_components):
        a=beta.sort_values(by=[beta.columns[i]],ascending=False)
        genes.extend(a.index[:n_genes])
        new_w.iloc[n_genes*i:n_genes*(i+1),:]=a.iloc[:n_genes,:].values
    new_w.index=genes
    new_w = new_w.drop_duplicates()
    cols=map(str, new_w.columns)
    new_w.columns=cols

    cmap = sns.cubehelix_palette(rot=-.4, light=0.98, as_cmap=True)
    f=sns.clustermap(new_w,method='ward', metric='euclidean',cmap=cmap, xticklabels=1, yticklabels=1)
    plt.show()
    return new_w
    
def gene_cluster_topic_intensity_heatmap(data):

    d=data.iloc[:, :-3]
    d['Cluster']=list(data['Cluster'])
    data = d.sort_values("Cluster",ascending=True)
    gene_cluster = d['Cluster']
    gene_cluster_topic = d.groupby(['Cluster']).mean()

    f=sns.clustermap(gene_cluster_topic.iloc[:,1:],method='ward', 
                     metric='euclidean',cmap="YlOrBr",row_cluster=False, xticklabels=1, yticklabels=1)
    plt.show()
    
def plot_topic_averages(cell_topics, celltype):    
    matches_type=cell_topics['assigned_cluster']== celltype
    fig = plt.figure(figsize=(20,15))
    mx=cell_topics[matches_type].mean().idxmax(1)
    color=['salmon' if col_name==mx else 'steelblue' for col_name in cell_topics.iloc[:,2:].columns]
    axs=cell_topics[matches_type].mean().plot.bar(color=color)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['bottom'].set_color('#DDDDDD')
    # Second, remove the ticks as well.
    axs.tick_params(bottom=False, left=False)
    # Third, add a horizontal grid (but keep the vertical grid hidden).
    # Color the lines a light gray as well.
    axs.set_axisbelow(True)
    axs.yaxis.grid(True, color='#EEEEEE')
    axs.xaxis.grid(False)
    axs.set_ylabel('TOPIC PROPORTION AVERAGE', fontsize=17)
    axs.set_xlabel('TOPIC', fontsize=17)
    axs.legend([mx], fontsize=17);
    leg = axs.get_legend()
    leg.legendHandles[0].set_color('salmon')
    axs.set_xticklabels([str(t) for t in range(50)], {'fontsize': 12})
    axs.set_title(celltype.upper(), fontsize=17)
    plt.show()
    
def get_gene_and_topic_emb_umap(fpath, model):
    topic_embeddings=model.alphas.weight.detach().numpy()
    topics=pd.DataFrame(topic_embeddings, index=['topic'+str(i) for i in range(topic_embeddings.shape[0])])
    emb, genes = load_embedding(fpath)
    emb=np.concatenate((emb,topics), axis=0)
    embedding = umap.UMAP(n_neighbors=15,
                      min_dist=0.3,
                      random_state=42,
                      metric='correlation').fit_transform(emb)
    embedding = pd.DataFrame(embedding)
    embedding.columns=['UMAP1','UMAP2']
    topics=embedding.tail(50)
    genes=embedding.drop(embedding.tail(50).index)#, inplace = True)
    return topics, genes

def plot_cluster_genes(fpath, model, name=None):
    topic_embeddings=model.alphas.weight.detach().numpy()
    topics=pd.DataFrame(topic_embeddings, index=['topic'+str(i) for i in range(topic_embeddings.shape[0])])
    emb, genes = load_embedding(fpath)
    
    ap = cluster.AffinityPropagation(damping = 0.9, preference=-50, random_state=1).fit(np.array(emb))
    emb=np.concatenate((emb,topics), axis=0)
    embedding = umap.UMAP(n_neighbors=15,
                      min_dist=0.3,
                      random_state=42,
                      metric='correlation').fit_transform(emb)
    embedding = pd.DataFrame(embedding)
    embedding.columns=['UMAP1','UMAP2']
    #print(embedding.shape)
    topics=embedding.tail(50)
    genes=embedding.drop(embedding.tail(50).index)
    #print(genes.shape)
    y_pred=ap.labels_
    genes["Cluster"]=y_pred
    f=sns.lmplot(x='UMAP1', y='UMAP2',data=genes,hue="Cluster",
               fit_reg=False,legend=False,scatter_kws={'s':5})
    for i in list(set(y_pred)):
        plt.annotate(i, 
                     genes.loc[genes['Cluster']==i,['UMAP1','UMAP2']].mean(),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=15, weight='bold')
    plt.title(f'Total inferred clusters: {len(np.unique(y_pred))}')
    if name is not None:
        plt.savefig('./Plots/gene_cluster_'+name+'.png', dpi=250, bbox_inches='tight')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    return y_pred, genes, topics
    