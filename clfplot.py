'''
this file is to visualize the results and understand the data
'''

import matplotlib.pyplot as plt
import numpy as np
import time

def lineplot(value, error, label, feature):
    plt.figure()
    plt.errorbar(range(1 , len(value)+1), value, yerr=error, ls=':', linewidth=2, color='m', ecolor='b', marker='o')
    plt.title(feature)
    plt.grid()

    #set x axis
    ax = plt.gca()
    ax.set_xticks(np.arange(len(value)+2))
    ax.set_xticklabels([''] + label + [''])
    ax.set_xlabel('Classification Model')

    #set y axis
    ax.set_ylabel('Error')
    #ax.set_ylim(0,1)












#lineplot([3,6,4],[1,1,1],['a','b','c'],'sdf')
#plt.show()