'''
this file is to visualize the results and understand the data
'''

import matplotlib.pyplot as plt
import numpy as np
import time

def lineplot(value, error, label, feature, prop):
    '''
    plot a single feature with all classification methods

    '''
    plt.figure()
    plt.errorbar(range(1 , len(value)+1), value, yerr=error, ls=':', linewidth=2, color='m', ecolor='b', marker='o')
    plt.title(prop + ':' + feature)
    plt.grid()

    #set x axis
    ax = plt.gca()
    ax.set_xticks(np.arange(len(value)+2))
    ax.set_xticklabels([''] + label + [''])
    ax.set_xlabel('Classification Model')

    #set y axis
    ax.set_ylabel('Error')
    #ax.set_ylim(0,1)




def lineplotall(value, error, label, feature, prop):
    '''
    plot all features and classification methods in one graph, for a certain property (HOMO, LUMO, GAP)
    convenient to compare various features
    '''
    plt.errorbar(range(1 , len(value)+1), value, yerr=error, ls=':', linewidth=2, marker='o', label=feature)
    plt.title(prop)
    plt.legend()
    plt.grid()

    #set x axis
    ax = plt.gca()
    ax.set_xticks(np.arange(len(value)+2))
    ax.set_xticklabels([''] + label + [''])
    ax.set_xlabel('Classification Model')

    #set y axis
    ax.set_ylabel('Error')
    #ax.set_ylim(0,1)










# lineplotall([3,6,4],[1,1,1],['a','b','c'],'binary', 'homo')
# plt.show()