# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:33:16 2020

@author: Soumyajit Saha
"""
import cv2
import numpy as np
import random
#from sklearn.metrics import mean_squared_error as mse
from PIL import ImageChops, Image
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import color
from statsmodels.stats.outliers_influence import variance_inflation_factor
from math import log10, sqrt 
import skimage 
from skimage import filters
import math 
from skimage.metrics import structural_similarity as ssim
from PIL import ImageEnhance

def objective(sol,greyscale_img,uniq):
        
    out=np.zeros(shape=([len(greyscale_img),len(greyscale_img[0])]))
           
        
    for it in range(0,len(uniq)): # creation of new image
        for i in range(0,len(greyscale_img)):
            for j in range(0,len(greyscale_img[0])):
                if uniq[it]==greyscale_img[i][j]:
                    out[i][j]=sol[it]
                    
    entropy = skimage.measure.shannon_entropy(out)  # calculation of entropy
    pv=len(out) # height of the image
    ph=len(out[0]) # weight of the image
    edge_sobel = filters.sobel(out) # egdes obtained from sobel filter
    # edge_sobel = filters.laplace(out)
    edge_sobel_fl=edge_sobel.flatten()
        
    E=0
    ne=np.count_nonzero(edge_sobel_fl)
    for i in range(len(edge_sobel_fl)):
        if edge_sobel_fl[i]>0:
            E+=edge_sobel_fl[i] # calculation of number of edge pixels
                    
        
    freq, x =  np.histogram(out, bins=256) # calculation of frequencies
    h=freq/sum(freq) # calculation of distribution of unique values
        
    Nt=0;
        
    for i in range(0,len(freq)):
        if h[i]>=h.mean(axis=0).mean(): # calulation the number of unique values having higher distribution than mean
            Nt+=1
        
    del_h=np.var(h)
            
    #math.log(entropy * Nt / del_h)    
    Fz=math.log(math.log(E)) * ne * entropy / (pv * ph)  # objective function        
        
        
    return Fz
        
######### Fitness function ##########
    
def fitness(sol,greyscale_img,uniq):
    value=objective(sol,greyscale_img,uniq)
    # value=evaluator(sol)
    if (value >= 0):
        z = 1 / (1 + value)
    else:
        z = 1 + abs(value)
    return z

def distance(a,b,dim):
    o = np.zeros(dim)
    for i in range(0,len(a)):
        o[i] = abs(a[i] - b[i])
    return o

def Levy(d):
    beta=3/2
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u=np.random.randn(d)*sigma
    v=np.random.randn(d)
    step=u/abs(v)**(1/beta)
    o=0.01*step
    return o        


def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                      # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr


for img_no in range(21,25):
    
    print("Image No. " + str(img_no))
    
    sr='D:/Project/Image Enhancement/Kodak/kodim' + str(img_no).zfill(2) + '.png'  # original image from kodak dataset
    source=Image.open(sr)
    dest=ImageEnhance.Contrast(source)
    dest=dest.enhance(0.8)
    ds='D:/Project/Image Enhancement/kodak_20%_red_con/' + str(img_no).zfill(2) + '.png'   # lesser contrast image to be saved here
    dest.save(ds)
    
    
    
    img=cv2.imread(ds)
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gr=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    uniq=np.unique(greyscale_img)
    uniq=np.sort(uniq)
    
    dim=len(uniq)
    
    ub=255
    lb=0
    
    r=(ub-lb)/10
    Delta_max=(ub-lb)/8.5
    
    Food_fitness=0
    Food_pos=np.zeros(dim)
    
    Enemy_fitness=math.inf
    Enemy_pos=np.zeros(dim)
    
    fitness_of_X = np.zeros(10)
    All_fitness = np.zeros(10)
    
    X = np.zeros(shape=(10,dim))
    DeltaX = np.zeros(shape=(10,dim))
    
    for i in range(0,10):
        for j in range(0,dim):
            X[i][j]=int(lb + random.uniform(0,1)*(ub-lb))
        
        X[i] = np.sort(X[i])
    
    i1=random.randint(0,9)
    i2=random.randint(0,9)
    
    while i2==i1:
        i2=random.randint(0,9)
        
    
    ub_del=max(distance(X[i1],X[i2],dim))
    
    
    
    for i in range(0,10):
        for j in range(0,dim):
            DeltaX[i][j]=int(lb + random.uniform(0,1)*(ub_del-lb))
        
        #DeltaX[i] = np.sort(DeltaX[i])
        
    Max_iteration=10
    
    for itr in range(1,Max_iteration+1):
        
        r=(ub_del-lb)/4+((ub_del-lb)*(itr/Max_iteration)*2)
        w=0.9-itr*((0.9-0.4)/Max_iteration)
        my_c=0.1-itr*((0.1-0)/(Max_iteration/2))
        
        if my_c<0:
            my_c=0
        
        s=2*random.random()*my_c
        a=2*random.random()*my_c
        c=2*random.random()*my_c
        f=2*random.random()*my_c
        e=my_c
        
        for i in range(0,10):
            fitness_of_X[i] = fitness(X[i],greyscale_img,uniq)
            All_fitness[i] = fitness_of_X[i]
            
            if fitness_of_X[i] > Food_fitness:
                Food_fitness = fitness_of_X[i]
                Food_pos=X[i]
            
            if fitness_of_X[i] < Enemy_fitness:
                if all((X[i] <= ub)) and all((X[i] >= lb)):
                    Enemy_fitness = fitness_of_X[i]
                    Enemy_pos = X[i]
                    
        for i in range(1,10):
            index=0
            neighbours_no=0
            
            Neighbours_X = np.zeros(shape=(10,dim))
            Neighbours_DeltaX = np.zeros(shape=(10,dim))
            
            for j in range(0,10):
                Dist2Enemy = distance(X[i],X[j],dim)
                if (all(Dist2Enemy<=r) and all(Dist2Enemy!=0)):
                    index=index+1
                    neighbours_no=neighbours_no+1
                    Neighbours_DeltaX[index]=DeltaX[j]
                    Neighbours_X[index]=X[j]
                    
            S=np.zeros(dim)           
            if neighbours_no>1:
                for k in range(0,neighbours_no):
                    S=S+(Neighbours_X[k]-X[i])
                S=-S
            else:
                S=np.zeros(dim)
                
            
            
            if neighbours_no>1:
                A=(sum(Neighbours_DeltaX))/neighbours_no
            else:
                A = DeltaX[i]
            
            
            
            if neighbours_no>1:
                C_temp=(sum(Neighbours_X))/neighbours_no
            else:
                C_temp=X[i]
        
            C=C_temp-X[i]
            
            
            
            Dist2Food=distance(X[i],Food_pos,dim)
                               
            if all(Dist2Food<=r):
                F=Food_pos-X[i]
            else:
                F=np.zeros(dim)
            
            
            
            Dist2Enemy=distance(X[i],Enemy_pos,dim)
                               
            if all(Dist2Enemy<=r):
                Enemy=Enemy_pos-X[i]
            else:
                Enemy=np.zeros(dim)
            
            
            
            for tt in range(0,dim):
                if X[i][tt]>ub:
                    X[i][tt]=ub
                    DeltaX[i][tt]=random.uniform(0,1)*(50-lb)
                    
                if X[i][tt]<lb:
                    X[i][tt]=lb
                    DeltaX[i][tt]=random.uniform(0,1)*(50-lb)
            
            temp=np.zeros(dim)
            Delta_temp=np.zeros(dim)
            
            if any(Dist2Food>r):
                if neighbours_no>1:
                    for j in range(0,dim):
                        Delta_temp[j] = int(w*DeltaX[i][j] + random.random()*A[i][j] + random.random()*C[i][j] + random.random()*S[i][j])
                        if Delta_temp[j]>Delta_max:
                            Delta_temp[j]=Delta_max
                        if Delta_temp[j]<-Delta_max:
                            Delta_temp[j]=-Delta_max
                        temp[j]=X[i][j]+(DeltaX[i][j])
                else:
                    temp=(X[i] + (Levy(dim))*X[i]).astype(int)
                    Delta_temp=0
            
            else:
                for j in range(0,dim):
                    Delta_temp[j] = int((a*A[j] + c*C[j] + s*S[j] + f*F[j] + e*Enemy[j]) + w*DeltaX[i][j])
                    if Delta_temp[j]>Delta_max:
                        Delta_temp[j]=Delta_max
                    if Delta_temp[j]<-Delta_max:
                        Delta_temp[j]=-Delta_max
                    temp[j]=X[i][j]+DeltaX[i][j]
                    
            if(fitness(temp,greyscale_img,uniq)) > fitness_of_X[i]:
                X[i]=temp
                DeltaX[i]=Delta_temp
            
            for j in range(0,dim):
                if X[i][j]<lb: # Bringinging back to search space
                        X[i][j]=lb
                    
                if X[i][j]>ub: # Bringinging back to search space
                    X[i][j]=ub
                    
        Best_score=Food_fitness
        Best_pos=Food_pos
        
        print("Iteration = " + str(itr))
    
    best_sol=Best_pos
     
    for it in range(0,len(uniq)): # creating enhanced image
        for i in range(0,len(greyscale_img)):
            for j in range(0,len(greyscale_img[0])):
                if uniq[it]==greyscale_img[i][j]:
                    gr[i][j]=best_sol[it]
                   
    res='D:/Project/Image Enhancement/results/DA/' + str(img_no).zfill(2)  # folder containing the output
    #res=''
    cv2.imwrite(res + './before.jpg', greyscale_img)
    cv2.imwrite(res + './after.jpg', gr)
    
    
    b=np.ones(len(uniq))
    ck = np.column_stack([best_sol, b])
    vif = [variance_inflation_factor(ck, i) for i in range(ck.shape[1])]
    psnr=PSNR(gr,greyscale_img)
    ssim_val = ssim(gr, greyscale_img)
    
    ######### the parameter values ##########
    print("psnr = "+str(psnr))
    print("vif = "+str(vif))
    print("ssim = "+str(ssim_val))
    
    file1 = open("D:/Project/Image Enhancement/results/DA/results.txt", "a")  
    file1.write(str(img_no).zfill(2) + ". \n\n")
    file1.write("psnr = "+ str(psnr) + "\n")
    file1.write("vif = "+str(vif) + "\n")
    file1.write("ssim = "+str(ssim_val) + "\n\n\n")
     
    file1.close() 
    
    
