import pdb
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy import stats

arena_width = input('The arena width is ')

num_depots = input('The number of depots is ')


#x = arena_width/5 
x = int(np.sqrt(num_depots))
territory_width = arena_width/x

f = open("depots_xml_conf.txt", "w")
pdb.set_trace()

Xcord, Ycord =[], []
#generate nest positions
count =0
for i in range(x):
    for j in range(x):
        valx = arena_width/2.0 - (1+2*i)*territory_width/2.0
        valy = arena_width/2.0 - (1+2*j)*territory_width/2.0
        Xcord.append(valx)
        Ycord.append(valy)
        f.write("NestPosition_"+str(count)+"=\""+str(valx)+","+str(valy)+"\"\n")  
        count+=1
    
#generate the range for distributing robots
count =0
for i in range(len(Xcord)):
    f.write("<distribute>\n")
    f.write( "<position max=\"" + str(Xcord[i]+0.5) + "," + str(Ycord[i]+0.5) +",  0.0\" method=\"uniform\" min=\"" + str(Xcord[i]-0.5) +", " + str(Ycord[i]-0.5) +",0.0\"/>\n" )
    f.write( "<orientation mean=\"0, 0, 0\" method=\"gaussian\" std_dev=\"360, 0, 0\"/>\n" )
    f.write( "<entity max_trials=\"100\" quantity=\"6\">\n" )
    f.write( "<foot-bot id=\"CPFA_"+ str(i) + "\"><controller config=\"CPFA\"/></foot-bot>\n")
    f.write( "</entity>\n" )
    f.write( "</distribute>\n")
    count+=1


 
count =0
for i in range(x):
    for j in range(x):
       f.write("<cylinder id=\"cyl"+ str(count) + "\" radius=\"0.15\" height=\"0.1\"\n")
       f.write("             movable=\"false\" mass=\"2.5\">\n")
       f.write("<body position=\""+ str(arena_width/2.0 - 2.5-i*5.0)+", "+ str(arena_width/2.0 - 2.5-j*5.0) + ", 0\" orientation=\"45,0,0\" /> \n")
       f.write("</cylinder>\n")
       count+=1

f.close()
