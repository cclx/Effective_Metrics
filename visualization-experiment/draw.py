import math
import os
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style="darkgrid")
#mpl.rcParams['agg.path.chunksize'] = 10000

def find_maxlen(pseu_labels):
    maxlen = 0
    for i in range(len(pseu_labels)):
        maxlen = max(maxlen, pseu_labels[i])
    
    return maxlen

def draw_probeTree(tokenizer_sentence, pseu_labels, norms, name):
    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    maxlen = find_maxlen(pseu_labels)
    
    x = [i for i in range(1, len(tokenizer_sentence) + 1)]
    
    axes.plot(x, pseu_labels, linestyle='-', color='g', marker='s', linewidth=1.5, label='pesu_Depth')
    
    axes.plot(x, norms, linestyle='-.', color='r', marker='o', linewidth=1.5, label='predict_Norms')
    
    #axes.yaxis.set_minor_locator(MultipleLocator(2.5))
    #axes.xaxis.set_minor_locator(MultipleLocator(0.5))
    #axes.grid(which='minor', c='lightgrey')
    
    #axes.set_ylabel("Norms_Depth")
    #axes.set_xlabel("Sentence")
    #axes.set_xticks([i for i in range(1, len(tokenizer_sentence) + 1)])
    #axes.set_yticks([i for i in range(1, int(maxlen) + 5)]) # 设置刻度
    #axes.set_xticklabels(tokenizer_sentence, rotation = 30, fontsize = 'small')

    
    plt.savefig(os.path.join('picture', name), dpi=300)

def draw_blueprint(y, start, name, sentence):
    plt.clf()
    #plt.title('blueprint: ('+sentence+')')
    #plt.xlabel('depth')
    #plt.ylabel('different-measure')
    layer = []
    
    for i in range(len(y)):
        x = [i for i in range(1, len(y[i]) + 1)]
        plt.plot(x, y[i], marker='o', markersize=3)
        layer.append('layer{:}'.format(start + i))
    
    plt.legend(layer)
    
    plt.savefig(os.path.join('picture', name), dpi=300)

def draw_varprint(y, start, name, max_depth):
    plt.clf()
    plt.title('varprint')
    plt.xlabel('depth')
    plt.ylabel('varmean-measure')
    layer = []
    
    for i in range(start, start+5):
        depth = min(max_depth[i-start], 20)
        x = [j for j in range(1, depth+1)]
        plt.plot(x, y[i-start][:depth], marker='o', markersize=3)
        layer.append('layer{:}'.format(i))
    
    plt.legend(layer)
    
    plt.savefig(os.path.join('picture', name), dpi=300)
    
    
