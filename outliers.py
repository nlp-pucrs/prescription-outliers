import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn import svm

def target(x): 
    dose = x['frequency'] * x['dose']
    if dose > x['overdose']:
        return 1
    else:
        return 0 

def getPrescriptions(prescription, med_name):
    prescription = prescription[prescription['medication'] == med_name]
    X = prescription[['dose','frequency']].values.astype(float)
    Y = prescription['target'].values.astype(int)
    return X, Y
    
def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def plotPrescriptions(X,Y):
    plt.plot(X[np.where(Y==0)[0],0],X[np.where(Y==0)[0],1],"bx",c='k')
    plt.plot(X[np.where(Y==1)[0],0],X[np.where(Y==1)[0],1],"bx",c='r')

class ddc_outlier():
    y_pred = []
    pr = {}
    frequency = pd.DataFrame([])
    alpha = 0.5
    metric = 'similarity'
    sim_matrix = np.zeros((1,1))

    def __init__(self, alpha=0.5, metric='similarity'):
        self.alpha = alpha
        self.metric = metric
    
    def fit(self, X):
        medication = pd.DataFrame(X, columns=['dose', 'freq'])
        medication['reg'] = 1
        MostFreq = medication[['reg','dose', 'freq']].groupby(['dose','freq']).agg(['count'])
        grouped = pd.DataFrame(MostFreq['reg']['count'])
        self.frequency = pd.DataFrame(grouped['count'].values, columns=['count'])
        dose_conv = []
        freq_dia = []
        for name in grouped.index:
            dose, freq = name
            dose_conv.append(dose)
            freq_dia.append(freq)

        self.frequency['dose'] = dose_conv
        self.frequency['freq'] = freq_dia
        X = self.frequency[['dose','freq']].values.astype(float)
        try:
            if self.metric == 'similarity':
                self.sim_matrix = cosine_similarity(X,X)
            else:
                self.sim_matrix = pairwise_distances(X,X,self.metric)
            medication_graph = nx.from_numpy_matrix(self.sim_matrix)
            self.pr = nx.pagerank(medication_graph, alpha=0.9, max_iter=1000, personalization=dict(self.frequency['count']))
        except:
            self.pr = dict(enumerate(np.zeros((len(X),1)).flatten()))
    
    def get_params(self):
        return self.pr, self.sim_matrix
    
    def predict(self, X):
        medication = pd.DataFrame(X, columns=['dose', 'freq'])
        medication['pr'] = 0

        for idx_frequency in self.frequency.index:
            med_frequency = self.frequency.iloc[idx_frequency]
            medication_index = medication[
                                        (medication['dose'] == med_frequency['dose']) &
                                        (medication['freq'] == med_frequency['freq'])
                                        ].index
            if len(medication_index) > 0:
                medication.loc[medication_index,'pr'] = self.pr[idx_frequency]
        
        pr_threshold = np.mean(np.array(list(self.pr.values())))

        y_pred = medication['pr'].values
        y_pred[y_pred < (pr_threshold*self.alpha)] = -1 # flag overdose
        y_pred[y_pred >= (pr_threshold*self.alpha)] = 1 # convert to false
        return y_pred

def getClassifiers(epsilon):
    classifiers = {
        "Cov": EllipticEnvelope(contamination=epsilon),
        "IsoF": IsolationForest(contamination=epsilon),
        "LOF": LocalOutlierFactor(n_neighbors=500, contamination=epsilon),
        "SVM": svm.OneClassSVM(nu=epsilon, gamma=4),
        "Gau": GaussianMixtureOutlier(alpha=epsilon),
        "DDC": ddc_outlier(alpha=epsilon),
        "DDC-C": ddc_outlier(alpha=epsilon,metric='cosine'),
        "DDC-J": ddc_outlier(alpha=epsilon,metric='jaccard'),
        "DDC-H": ddc_outlier(alpha=epsilon,metric='hamming'),
        "DDC-M": ddc_outlier(alpha=epsilon,metric='mahalanobis'),
    }
    return classifiers
    
def getRanges():
    classifiers = {
        "Cov": np.arange(0.01,0.5,0.01),
        "IsoF": np.arange(0.01,0.5,0.01),
        "LOF": np.arange(0.01,0.5,0.01),
        "DDC-J": np.arange(0.01,1.0,0.01),
        "SVM": np.arange(0.01,1.0,0.01),
        "Gau": np.arange(0.01,1.0,0.01),
        "DDC": np.arange(0.01,1.0,0.01),
        "DDC-C": np.arange(0.01,1.0,0.01),
        "DDC-J": np.arange(0.01,1.0,0.01),
        "DDC-H": np.arange(0.01,1.0,0.01),
        "DDC-M": np.arange(0.01,1.0,0.01),
    }
    return classifiers

class GaussianMixtureOutlier():
    pb = []
    alpha = 0.6

    def __init__(self, alpha):
        self.alpha = alpha
        
    def fit(self, X):
        gmm = GaussianMixture(n_components=2).fit(X)
        mu1, mu2 = gmm.means_
        sigma1, sigma2 = gmm.covariances_
        self.pb = multivariate_normal(mean=mu1, cov=sigma1)
        
    def get_params(self):
        return self.pb
        
    def predict(self, X):
        p = self.pb.pdf(X)
        positives = np.asarray(np.where(p <= self.alpha))
        
        y=np.array([1]*len(X))
        y[positives[0]]=-1
        
        return y
    
def evaluateMethods(X,Y, p_svm, p_cov, p_ift, p_lof, p_wpr, p_gmx, debug=True):
    classifiers = getClassifiers()
        
    results = pd.DataFrame()
    results['Time'] = 0
    results['Accuracy'] = 0
    results['Recall'] = 0
    results['Precision'] = 0
    results['F-Measure'] = 0
    
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        start = time.time()
        
        try:
        
            if clf_name == "LOF":
                y_pred = clf.fit_predict(X)
            else:
                clf.fit(X)
                y_pred = clf.predict(X)

            y_pred[y_pred == 1] = 0
            y_pred[y_pred == -1] = 1

            acc = metrics.accuracy_score(Y,y_pred)
            rec = metrics.recall_score(Y,y_pred)
            prec = metrics.precision_score(Y,y_pred)
            f1 = metrics.f1_score(Y,y_pred)
            
        except:
            acc = rec = prec = f1 = 0
        
        end = time.time()
        time_total = round(end - start,3)
    
        results.loc[clf_name,'Time'] = time_total
        results.loc[clf_name,'Accuracy'] = acc
        results.loc[clf_name, 'Recall'] = rec
        results.loc[clf_name, 'Precision'] = prec
        results.loc[clf_name, 'F-Measure'] = f1
        
        if debug:
            print(clf_name)
            print(round(end - start,3), acc, rec, prec, f1)

    return results

def getOverdoseMedications(prescription): 
    prescription = prescription[prescription['target'] > 0]
    MostMedicines = prescription[['target','medication']].groupby(['medication']).agg(['count'])
    MostMedicines = MostMedicines['target']
    return MostMedicines[MostMedicines['count'] > 10]['count'].index


def runParameterSearch(prescription, medications, ep_range, minimum=1000, norm=False):
    max_t = pd.DataFrame()
    max_a = pd.DataFrame()
    max_r = pd.DataFrame()
    max_p = pd.DataFrame()
    max_f = pd.DataFrame()

    for med in medications:

        med_t = pd.DataFrame()    
        med_a = pd.DataFrame() 
        med_r = pd.DataFrame() 
        med_p = pd.DataFrame() 
        med_f = pd.DataFrame() 

        X, Y = getPrescriptions(prescription, med)
        if norm:
            X = normalize(X,norm='l2')

        if len(X) < minimum: 
            continue
            
        sys.stdout.write(med + ', ' + 
                         str(len(X)) + ', ' + 
                         str(len(Y[Y==1])) + ', ' +
#                         str(len(np.unique(X,axis=0))) + ', ' +
#                         str(np.mean(X)) + ', ' +
#                         str(np.std(X)) + ', ' +
#                         str(np.median(X)) + ', ' +
#                         str(np.percentile(X,25)) + ', ' +
#                         str(np.percentile(X,50)) + ', ' +
#                         str(np.percentile(X,75)) + ', ' +
                         
#                         str(np.mean(X[:,0])) + ', ' +
#                         str(np.std(X[:,0])) + ', ' +
#                         str(np.median(X[:,0])) + ', ' +
#                         str(np.percentile(X[:,0],25)) + ', ' +
#                         str(np.percentile(X[:,0],50)) + ', ' +
#                         str(np.percentile(X[:,0],75)) + ', ' +
                         
#                         str(np.mean(X[:,1])) + ', ' +
#                         str(np.std(X[:,1])) + ', ' +
#                         str(np.median(X[:,1])) + ', ' +
#                         str(np.percentile(X[:,1],25)) + ', ' +
#                         str(np.percentile(X[:,1],50)) + ', ' +
#                         str(np.percentile(X[:,1],75)) + ', ' +
                        '')
            
        f_scores = []

        for ep in ep_range:
            p_svm = [ep,4]
            p_cov = [ep]
            p_ift = [ep]
            p_lof = [500,ep]
            p_wpr = [ep]
            p_gmx = [ep]
            results_norms = evaluateMethods(X, Y, p_svm, p_cov, p_ift, p_lof, p_wpr, p_gmx, debug=False)

            #sys.stdout.write(str(ep) +', ')
            for idx in results_norms.index:
                med_t.loc[idx, str(ep)] = results_norms.loc[idx,'Time']
                med_a.loc[idx, str(ep)] = results_norms.loc[idx,'Accuracy']
                med_r.loc[idx, str(ep)] = results_norms.loc[idx,'Recall']
                med_p.loc[idx, str(ep)] = results_norms.loc[idx,'Precision']
                med_f.loc[idx, str(ep)] = results_norms.loc[idx,'F-Measure']

        #print('')
        #sys.stdout.write('Best Params: ')
        for idx in med_f.index:
            max_f.loc[idx,med] = med_f.loc[idx].max()
            ep_max = med_f.loc[idx].idxmax()
            max_a.loc[idx,med] = med_a.loc[idx, ep_max]
            max_r.loc[idx,med] = med_r.loc[idx, ep_max]
            max_p.loc[idx,med] = med_p.loc[idx, ep_max]   
            max_t.loc[idx,med] = med_t.loc[idx, ep_max]
            sys.stdout.write(', (' + str(idx) + '+' + str(ep_max) + '), ' + str(med_f.loc[idx].max()) + ', ')
        print('')

    results = pd.DataFrame()

    results['Time'] = max_t.mean(1)
    results['Accuracy'] = max_a.mean(1)
    results['Recall'] = max_r.mean(1)
    results['Precision'] = max_p.mean(1)
    results['F-Measure'] = max_f.mean(1)
    
    return results