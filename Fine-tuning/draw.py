import math
import os
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

def find_maxlen(pseu_labels):
    maxlen = 0
    for i in range(len(pseu_labels)):
        maxlen = max(maxlen, pseu_labels[i])
    
    return maxlen

def draw_Evaluated_probe(loss_stats, name):
    plt.clf()
    #plt.title('Evaluate('+name+')')
    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    
    x = [i for i in range(1, 26)]
    
    y = loss_stats
    
    xlabels = []
    
    for i in range(0, 25):
        xlabels.append("L{:}".format(i))
    
    axes.plot(x, y, linestyle='-', color='g', marker='s', linewidth=1.5, label='pesu_Depth')
    
    axes.yaxis.set_minor_locator(MultipleLocator(2.5))
    axes.xaxis.set_minor_locator(MultipleLocator(0.5))
    axes.grid(which='minor', c='lightgrey')
    
    axes.set_ylabel("Score("+name+")")
    axes.set_xlabel("Layer")
    axes.set_xticks([i for i in range(1, 26)])
    axes.set_yticks([i*0.05 for i in range(0, 21)]) 
    axes.set_xticklabels(xlabels, rotation = 30, fontsize = 'small')

    #plt.show()
    plt.savefig(os.path.join('picture', 'Evaluated_probe_cola_{:}'.format(name)), dpi=300)

def draw_Evaluated_normaltrain(loss_stats):
    plt.clf()
    #plt.title('Evaluate('+name+')')
    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    
    x = [i for i in range(1, 11)]
    
    y = loss_stats
    
    xlabels = []
    
    for i in range(0, 10):
        xlabels.append("seed{:}".format(i*4+1))
    
    axes.plot(x, y, linestyle='-', color='g', marker='s', linewidth=1.5, label='pesu_Depth')
    
    axes.yaxis.set_minor_locator(MultipleLocator(2.5))
    axes.xaxis.set_minor_locator(MultipleLocator(0.5))
    axes.grid(which='minor', c='lightgrey')
    
    axes.set_ylabel("Score")
    axes.set_xlabel("Seed")
    axes.set_xticks([i for i in range(1, 11)])
    axes.set_yticks([i*0.05 for i in range(0, 21)]) 
    axes.set_xticklabels(xlabels, rotation = 30, fontsize = 'small')

    #plt.show()
    plt.savefig(os.path.join('picture', 'Evaluated_cola_nomal_train'), dpi=300)

def draw_Evaluated_overmeasure(y, loss_stats, start, name):
    plt.clf()
    plt.title('measure-layer')
    plt.xlabel('Seed')
    plt.ylabel('Score')
    
    xlabels = []
    
    for i in range(0, 10):
        xlabels.append("seed{:}".format(i*4+1))
        
    layer = []

    for i in range(start, start+5):
        x = [i for i in range(1, len(y[i]) + 1)]
        plt.plot(x, y[i], marker='o', markersize=3)
        layer.append('layer{:}'.format(i))
        
    x = [i for i in range(1, len(loss_stats) + 1)]
    plt.plot(x, loss_stats, marker='o', markersize=3)
    layer.append('no-probe')
    
    #plt.set_xticks([i for i in range(1, 11)])
    #plt.set_yticks([i*0.05 for i in range(0, 21)]) 
    #plt.set_xticklabels(xlabels, rotation = 30, fontsize = 'small')
    
    
    plt.legend(layer)

    plt.savefig(os.path.join('picture', name), dpi=300)
    

if __name__ == '__main__':
    draw_Evaluated_probe()
    #draw_Evaluated_normaltrain()
