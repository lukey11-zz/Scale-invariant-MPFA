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
    travel_datas, search_datas=[], []
    for f in files:
        data = get_data_from_file(f)
        travel, search = process_data(data)
        travel_datas.append(travel)
        search_datas.append(search)
    return travel_datas, search_datas
 
def process_data(datas):
    travel, search =[], []
    for line in datas:
        words =line.replace(",","").split()
        travel.append(float(words[0])/60.0)
        search.append(float(words[1])/60.0)
    return travel, search

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


fileNames = ["random_dynamic_MPFA_n1_r6_tag512_5by5_TravelSearchTimeData.txt", "random_dynamic_MPFA_n1_r12_tag512_10by10_TravelSearchTimeData.txt", "random_dynamic_MPFA_n1_r24_tag512_20by20_TravelSearchTimeData.txt"]
travel_datas, search_datas = get_multiple_data(fileNames)


#fileNames = ["with_random_dynamic_MPFA_n4_r24_tag512_10by10_iAntTagData.txt", "with_random_dynamic_MPFA_n4_r48_tag512_20by20_iAntTagData.txt", "with_random_dynamic_MPFA_n4_r72_tag512_30by30_iAntTagData.txt"]
#with_comm_datas = get_multiple_data(fileNames)


width =0.1
N=3
ind = np.arange(N)  # the x locations for the groups

#colors =['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
colors =['#e41a1c', '#377eb8']


outputFile = open('linearReg_travelSearch.txt', 'w+')

fig, axarr = plt.subplots(2,1)
Y= travel_datas

rect= axarr[0].boxplot(Y, notch=True, positions =ind, widths= 0.1)

plt.setp(rect['boxes'], color=colors[0])


slopes=[]
intercepts =[]
#pdb.set_trace()

slope, intercept = linearReg(axarr[0], ind, np.array(Y).mean(axis=1), colors[0], outputFile) 

# add some text for labels, title and axes ticks
axarr.set_ylabel('Travel time', fontsize=20)
axarr.set_xlim(-0.5, 3)
axarr.set_yticks(np.arange(0, 150, 20))


#savefig('overall_travelTime')


#fig, axarr = plt.subplots()
Y= search_datas

rect= axarr[1].boxplot(Y, notch=True, positions =ind, widths= 0.1)

plt.setp(rect['boxes'], color=colors[0])


slopes=[]
intercepts =[]
slope, intercept = linearReg(axarr[1], ind, np.array(Y).mean(axis=1), colors[0], outputFile) 


outputFile.close()

# add some text for labels, title and axes ticks
axarr.set_ylabel('Search time', fontsize=20)
axarr.set_xlim(-0.5, 3)
axarr.set_yticks(np.arange(0, 200, 20))


savefig('overall_searchTime')

plt.show()
