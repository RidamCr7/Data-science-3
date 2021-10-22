import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statistics as st
#fiter warning
import warnings
warnings.filterwarnings("ignore")


#3a
#drop class
df2=df2.drop(['class'], axis=1)
#correlation matrix of remaining frame
cormat=df2.corr()
cormat=np.array(cormat)
#eigen values and vectors of correlation matrix
W3a,V3a=np.linalg.eig(cormat)
#rearrange vectors in descending order of eigenvalues
for i in range(0,7):
    for x in range(i+1,8):
        if W3a[x]>W3a[i]:
            W3a[x],W3a[i]=W3a[i],W3a[x]
            for k in range(0,8):
                V3a[x][k],V3a[i][k]=V3a[i][k],V3a[x][k]
#print eigenalues and eigenvectors
print(W3a)
print(V3a)
ev1=V3a[:,0]
ev2=V3a[:,1]
#lists for coordinates of projections
aev1=[]
aev2=[]
for i in range(0,len(df2['pregs'])):
    c=0
    k=0
    #calculating coordinates
    for x in range(0,8):
        c+=ev1[x]*df2[Cols[x]][i]
        k+=ev2[x]*df2[Cols[x]][i]
    aev1.append(c)
    aev2.append(k)
#scatter plot
#give title
plt.title("Coordinates with d=8--> l=2")
plt.scatter(aev1,aev2)
plt.grid()
#show plot
plt.show()
#Print eigenvalue1 and eigenvalue2
print("Eigen Value 1",W3a[0])
print("Eigen Value 2",W3a[1])
print("Variance of projection on eigenvector1",st.variance(aev1))
print("Variance of projection on eigenvector2",st.variance(aev2))

#3b
#give title
plt.title("Eigen values in descending vector")
x=np.arange(1,9)
#line plot
plt.plot(x,W3a,'r',marker='o')
#show plot
plt.show()

#3c
error_record=[]
for i in range(1,9):
    pca = PCA(n_components=i, random_state=50)
    pcar = pca.fit_transform(df2)
    pcap=pca.inverse_transform(pcar)
    total_loss=np.linalg.norm((df2-pcap),None)
    error_record.append(total_loss)
    co3 = np.round(np.matmul(np.transpose(pcar),pcar), decimals=3)
    print(i)
    print(co3)

l2 = [i for i in range(1,9)]
plt.title("Reconstruction Error of Pca",size=22)
plt.plot(l2,error_record,'r',marker='o')
plt.xlabel('No of dimenssion (l)')
plt.ylabel('Euclidan distance')
plt.grid(color='grey', linestyle='-.', linewidth=0.7)
plt.show()

#3d
#datframe of restructured values
df4 = np.dot(df2,V3a)
#new cov values
co3 = np.dot(np.transpose(df4),df4)
#original Covariance matrix
print("Original Matrix")
print(cormat)
#new Covariance matrix
print("New Matrix")
print(co3)
