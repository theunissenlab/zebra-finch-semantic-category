# Helper functions, this will eventually live in a library for us
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import umap

from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist


callColor = {
        'Be':(0/255.0, 230/255.0, 255/255.0),
        'Tu':(255/255.0, 200/255.0, 65/255.0),
        'Th':(255/255.0, 150/255.0, 40/255.0),
        'Alarm':(255/255.0, 200/255.0, 65/255.0),
        'Di':(255/255.0, 105/255.0, 15/255.0),
        'Ag':(255/255.0, 0/255.0, 0/255.0),
        'Fight':(255/255.0, 105/255.0, 15/255.0),
        'Wh':(255/255.0, 180/255.0, 255/255.0),
        'Ne':(255/255.0, 100/255.0, 255/255.0),
        'Te':(140/255.0, 100/255.0, 185/255.0),
        'Soft':(255/255.0, 180/255.0, 255/255.0),
        'DC':(100/255.0, 50/255.0, 200/255.0),
        'LT':(0/255.0, 95/255.0, 255/255.0),
        'Loud':(100/255.0, 50/255.0, 200/255.0),
        'song':(0, 0, 0),
        'So':(0,0,0), 
        'In': (0.49,0.60,0.55), 
        'Mo':(0.69,0.39,0.39),
        'Ri':(0,255/255.0,0)}
fine_to_coarse = {
    'Ag': 'Fight',
    'Di': 'Fight',
    'Ne': 'Soft',
    'Wh': 'Soft',
    'Te': 'Soft',
    'LT': 'Loud',
    'DC': 'Loud',
    'Th': 'Alarm',
    'Tu': 'Alarm',
    'Be': 'Be',
    'In': 'song',
    'Mo': 'song',
    'So': 'song'
}

def cluster_analysis(representation, labels, and_plot=False, dr_method='umap', method='ward',criterion='distance',save=None,score='ari'):
    numcomps = min(20, representation.shape[1])
    nfiles = 5
    MesPCS = dict()##representation[:,:]
    if dr_method == 'umap':
        MesPCS = umap.UMAP( n_neighbors=30,
                            min_dist=0.0,
                            n_components=2,
            ).fit_transform(representation[:,:])
    elif dr_method == 'pca':
        # first run PCA with a large amount of components
        pca = PCA(n_components=numcomps)
        pca.fit_transform(representation[:,:])
        #Find the number of components that represent 95% of variance
        nc = np.where(np.cumsum(pca.explained_variance_ratio_)>.95)[0][0]
        pca = PCA(n_components=nc)
        MesPCS=pca.fit_transform(representation[:,:])
    else:
        MesPCS = representation[:,:]

    PCs4Linkage=MesPCS
    clustered=linkage(PCs4Linkage,method =method)# this works now
    cophen_dist = cophenet(clustered)
    call_types = np.unique(labels)
    ctlist = call_types.tolist()

    max_ari = 0
    max_cl = 0

    delta = (clustered.max()-clustered.min())/500 #.1
    start_thresh = clustered.min()#100 #.25
    cur_grouping = None
    unique_groupings = []
    best_grouping = None
    for cl in range(1,500):
        F = fcluster(clustered, start_thresh + cl*delta, criterion=criterion)
        if np.all(cur_grouping == F):
            continue
        unique_groupings.append(F)
        cur_grouping = F
        
        # check ari
        compares = [[],[]]
        for gid in range(max(F)):
            for j in np.where(F==gid)[0]:
                    vt = labels[j]
                    compares[0].append(ctlist.index(vt))
                    compares[1].append(gid)
        if score == 'ari':
            ari = adjusted_rand_score(compares[0],compares[1])
        if score == 'ami':
            ari = adjusted_mutual_info_score(compares[0],compares[1])
        #print("%s %s"%(ari,start_thresh + cl*delta))
        # previously just go until 10 groups
        #if max(F) < 10:
        #    break
        if ari == 1.0:# or ari == 0.0:
            break
        if ari > max_ari:
            best_grouping = cur_grouping
            max_cl = cl
            max_ari = ari
            
    F = fcluster(clustered,  start_thresh + max_cl*delta, criterion=criterion)
    #print("-------------")
    #print("ARI: %s"%max_ari)
    if and_plot:
        plt.figure(figsize= (40,20))
        D = dendrogram(Z=clustered, leaf_rotation=90.,
             leaf_font_size=20., # determine number by plotting cluster colors on leaves first
             #labels= listfinesem_calls[0], #finesem_calls[1:7633],
             labels=np.array(labels), #voc_types,#plottinglabels,
             color_threshold= start_thresh+max_cl*delta,
             above_threshold_color='grey')
        #out_PSTH_KDEs['PCA_Group'][i,:] = F
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        for t in xlbls:
            t.set_color(callColor.get(t.get_text(),(0,0,0)))
        if save is not None:
            plt.savefig(save)
        plt.show()
    return F,max_ari, MesPCS, unique_groupings,best_grouping,start_thresh+max_cl*delta

def plot_groupings(signal, groups, save = None):
    ncs=int(max(groups)/5+.5)
    fig, ax = plt.subplots(min(5,max(groups)), ncs, sharex='col')
    for r in range(min(5,max(groups))):
        for c in range(ncs):
            gid = r*ncs+c%ncs+1
            ax_tmp = ax[r,c] if ncs > 1 else ax[r]
            for j in np.where(groups==gid)[0]:
                ax_tmp.plot(signal[j,:]) 
            ax_tmp.set_ylim(0,round(signal.max()/10 + .5)*10)

    if save is not None:
        plt.savefig(save)
    plt.show()