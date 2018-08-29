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
    #fit_fn = np.poly1d(fit)
    #handle.plot(X, fit_fn(X), color)
    slope, intercept, r_value, p_value, stderr = stats.linregress(X, Y)
    outputFile.write(str(slope)+'\t\t'+str(intercept)+'\t\t\t'+str(r_value**2)+'\t\t\t'+str(p_value)+'\t\t'+str(stderr)+'\r')
    return slope, intercept


fileNames = ["random_CPFA_r6_tag128_5by5_iAntTagData.txt",
"random_CPFA_r24_tag512_10by10_iAntTagData.txt",
"random_CPFA_r96_tag2048_20by20_iAntTagData.txt","random_CPFA_r384_tag8192_40by40_iAntTagData.txt","random_CPFA_r1536_tag32768_80by80_iAntTagData.txt"]
random_datas = get_multiple_data(fileNames)

fileNames = ["powerlaw_CPFA_r6_tag128_5by5_iAntTagData.txt",
"powerlaw_CPFA_r24_tag512_10by10_iAntTagData.txt",
"powerlaw_CPFA_r96_tag2048_20by20_iAntTagData.txt","powerlaw_CPFA_r384_tag8192_40by40_iAntTagData.txt","powerlaw_CPFA_r1536_tag32768_80by80_iAntTagData.txt"]
powerlaw_datas = get_multiple_data(fileNames)

fileNames = ["cluster_CPFA_r6_tag128_5by5_iAntTagData.txt",
"cluster_CPFA_r24_tag512_10by10_iAntTagData.txt",
"cluster_CPFA_r96_tag2048_20by20_iAntTagData.txt","cluster_CPFA_r384_tag8192_40by40_iAntTagData.txt","cluster_CPFA_r1536_tag32768_80by80_iAntTagData.txt"]
cluster_datas = get_multiple_data(fileNames)


random_avg, powerlaw_avg, cluster_avg =[], [], []
num_robots = [6.0, 24.0, 96.0, 384.0, 1536.0]

for d, r in zip(random_datas, num_robots):
    random_avg.append(np.array(d)/r)

for d, r in zip(powerlaw_datas, num_robots):
    powerlaw_avg.append(np.array(d)/r)

for d, r in zip(cluster_datas, num_robots):
    cluster_avg.append(np.array(d)/r)

 

#fileNames = ["with_random_dynamic_MPFA_n4_r24_tag512_10by10_iAntTagData.txt", "with_random_dynamic_MPFA_n4_r48_tag512_20by20_iAntTagData.txt", "with_random_dynamic_MPFA_n4_r72_tag512_30by30_iAntTagData.txt"]
#with_comm_datas = get_multiple_data(fileNames)


width =0.1
N=5
#ind = np.arange(N)  # the x locations for the groups
ind = np.array([0, 2, 4, 6, 8])
#colors =['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
colors =['#e41a1c', '#377eb8', '#984ea3']


outputFile = open('linearReg.txt', 'w+')

fig, axarr = plt.subplots()
means = np.array(random_avg).mean(axis=1)
axarr.plot(ind, means, colors[0])
print means
Y= random_avg

rect1= axarr.boxplot(Y, notch=True, positions =ind, patch_artist=True, widths= 0.18)

plt.setp(rect1['boxes'], color=colors[0])

slope, intercept = linearReg(axarr, ind, np.array(Y).mean(axis=1), colors[0], outputFile) 


means = np.array(powerlaw_avg).mean(axis=1) 
axarr.plot(ind+0.3, means, colors[1])
print means
Y= powerlaw_avg

rect2= axarr.boxplot(Y, notch=True, positions =ind+0.3, patch_artist=True, widths= 0.18)

plt.setp(rect2['boxes'], color=colors[1])

slope, intercept = linearReg(axarr, ind+0.3, np.array(Y).mean(axis=1), colors[1], outputFile) 


means = np.array(cluster_avg).mean(axis=1) 
axarr.plot(ind+0.6, means, colors[2])
print means
Y= cluster_avg

rect3= axarr.boxplot(Y, notch=True, positions =ind+0.6, patch_artist=True, widths= 0.18)

plt.setp(rect3['boxes'], color=colors[2])

slope, intercept = linearReg(axarr, ind+0.6, np.array(Y).mean(axis=1), colors[2], outputFile) 



outputFile.close()

# add some text for labels, title and axes ticks
axarr.set_ylabel('Foraging per robot', fontsize=20)
#ax.set_title('Foraging rate in each model', fontsize=20)
axarr.set_xlim(-0.5, 9)
#axarr.set_xticks(ind+width)
axarr.set_xticklabels( ('0', '2','4', '6', '8'), fontsize=12)
axarr.set_xlabel('Log2 of number of depots', fontsize=20)
axarr.set_yticks(np.arange(0, 25, 2))

plt.figtext(0.54, 0.86, '____',
            backgroundcolor=colors[0], color=colors[0], weight='roman',
            size='large') #size='x-small'
plt.figtext(0.6, 0.85, 'Random',
             color='black', weight='roman',
            size='x-large') #size='x-small''large'


plt.figtext(0.54, 0.82, '____',
            backgroundcolor=colors[1], color=colors[1], weight='roman',
            size='large') #size='x-small'
plt.figtext(0.6, 0.81, 'Partially clustered',
             color='black', weight='roman',
            size='x-large') #size='x-small''large'

plt.figtext(0.54, 0.78, '____',
            backgroundcolor=colors[2], color=colors[2], weight='roman',
            size='large') #size='x-small'
plt.figtext(0.6, 0.77, 'Clustered',
             color='black', weight='roman',
            size='x-large') #size='x-small''large'

savefig('forage_rate_per_robot')

plt.show()
