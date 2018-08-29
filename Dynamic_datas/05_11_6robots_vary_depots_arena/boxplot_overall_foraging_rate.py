import pdb
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy import stats

def get_data_from_file(filename):
    ''' read data from file into a list'''
    f = open(filename)
    filecontents = f.readlines()
    table = [line.strip('\n') for line in filecontents]
    f.close()
    return table
def get_multiple_data(files):
    datas=[]
    for f in files:
        data = get_data_from_file(f)
        forage = compute_overall_forage_data(data)
        datas.append(forage)
    return datas
 


def compute_overall_forage_data(datas):
    words=datas[0].replace(",","").split()
    if words[0]!='tags_collected':
        print "the data may not correct!"
        return
    forage=[]
    for line in datas[1:]:
        words =line.replace(",","").split()
        forage.append(float(words[0]))
    return forage
    #mean = np.mean(forage)
    #std = np.std(forage)
    #return mean, std

def compute_mean_std(fileNames):
    means, stds=[], []
    for fileName in fileNames:
        datas = get_data_from_file(fileName)
        forage = compute_overall_forage_data(datas)
        mean, std = np.mean(forage), np.std(forage)
        means.append(mean)
        stds.append(std)
    return means, stds

def plot_bars(handle, means, stds, Color, counter, width, ind):
    rects= handle.bar(ind+counter*width,np.array(means),width=width, color=Color, yerr=np.array(stds), ecolor='k')
    return rects

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 0.6*height, '%0.1f'%float(height),
                ha='center', va='bottom', color='white')

def linearReg(handle, X, Y, color, outputFile):
    fit = np.polyfit(X, Y,1)
    slope, intercept = np.polyfit(X, Y,1)
    fit_fn = np.poly1d(fit)
    #axarr[0].plot(ind1, random_data[0], 'ro', ind1, fit_fn(ind1), 'k')
    handle.plot(X, fit_fn(X), color)
    slope, intercept, r_value, p_value, stderr = stats.linregress(X, Y)
    outputFile.write(str(slope)+'\t\t'+str(intercept)+'\t\t\t'+str(r_value**2)+'\t\t\t'+str(p_value)+'\t\t'+str(stderr)+'\r')
    return slope, intercept


fileNames = ["random_dynamic_MPFA_n1_r6_tag512_5by5_iAntTagData.txt", "random_dynamic_MPFA_n4_r6_tag512_10by10_iAntTagData.txt", "random_dynamic_MPFA_n16_r6_tag512_20by20_iAntTagData.txt"]
without_comm_datas = get_multiple_data(fileNames)


#fileNames = ["with_random_dynamic_MPFA_n4_r24_tag512_10by10_iAntTagData.txt", "with_random_dynamic_MPFA_n4_r48_tag512_20by20_iAntTagData.txt", "with_random_dynamic_MPFA_n4_r72_tag512_30by30_iAntTagData.txt"]
#with_comm_datas = get_multiple_data(fileNames)


width =0.1
N=3
ind = np.arange(N)  # the x locations for the groups

#colors =['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
colors =['#e41a1c', '#377eb8']


outputFile = open('linearReg.txt', 'w+')

fig, axarr = plt.subplots()
Y= without_comm_datas

rect= axarr.boxplot(Y, notch=True, positions =ind, widths= 0.1)

plt.setp(rect['boxes'], color=colors[0])


slopes=[]
intercepts =[]
#pdb.set_trace()

#for x, y, color in zip(ind, np.array(Y).mean(axis=1), colors):
#pdb.set_trace()
slope, intercept = linearReg(axarr, ind, np.array(Y).mean(axis=1), colors[0], outputFile) 

#axarr.plot(ind, ind*slope+intercept, colors[0])


outputFile.close()

# add some text for labels, title and axes ticks
axarr.set_ylabel('Number of collected resources', fontsize=20)
#ax.set_title('Foraging rate in each model', fontsize=20)
axarr.set_xlim(-0.5, 3)
#axarr.set_xticks(ind+width)
#axarr.set_xticklabels( ('No communication', 'Communication'), fontsize=18)
axarr.set_yticks(np.arange(0, 250, 50))


savefig('overall_forage_rate')

plt.show()
