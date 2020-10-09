import numpy as np
import pandas as pd
import scipy.linalg as la
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/home/shikha/Desktop/Iris.csv")
f = df.iloc[:,1:5]
print(df.columns)


f1 = f.iloc[:, 0:1]
f2 = f.iloc[:, 1:2]
f3 = f.iloc[:, 2:3]
f4 = f.iloc[:, 3:4]
print(f1)
print(len(f1))


mu1 = np.sum(f1)/len(f1)
mu2 = np.sum(f2)/len(f2)
mu3 = np.sum(f3)/len(f3)
mu4 = np.sum(f4)/len(f4)


f11 = 0
f22 = 0
f33 = 0
f44 = 0

for i in range(0, len(f1)):
    f11 += ((f1.iloc[i].sub(mu1))**2)
    f22 += ((f2.iloc[i].sub(mu2))**2)
    f33 += ((f3.iloc[i].sub(mu3))**2)
    f44 += ((f4.iloc[i].sub(mu4))**2)
    
    
var1 = np.sqrt(f11)/len(f1)
print(var1)
var2 = np.sqrt(f22)/len(f2)
print(var2)
var3 = np.sqrt(f33)/len(f3)
print(var3)
var4 = np.sqrt(f44)/len(f4)
print(var4)


df1 = f1.join(f2)
df1 = df1.join(f3)
df1 = df1.join(f4)
print(df1)

sample, features = df1.shape
cov = np.zeros([features, features])


for i in range(0, features):
    for j in range(0, features):
        for k in range(0, len(f1)):
            cov[i][j] += df1.iloc[k,i]*df1.iloc[k,j]
   
print(cov)

eigen_val, eigen_vec = la.eig(cov)
print(eigen_val)  
print(eigen_vec)  # 4 vector represents how much part contribute all features into all pca,  
total = np.sum(eigen_val)
pc = []

for i in range(len(eigen_val)):
    pc.append(eigen_val[i]/total)  # this matrix represent how much part of pca1, pca2 etc, and see pc1 have 96% and pc2 have 3%, so we take these 2 pca for dimention reduction.

print(pc)
eig_vec = eigen_vec[:, :2]  
eig_vec = eig_vec.T
print(eig_vec.shape)
d = np.zeros([sample,2]) 



for i in range(0, sample):
    u = np.matmul(eig_vec, np.array(df1.iloc[i,:]))
    for j in range(len(u)):
        d[i][j] = u[j]
print(d)

df2 = pd.DataFrame(d)
df2 = pd.concat([df2,df['Species']], axis=1)

df2.columns = ['feature1', 'feature2', 'Species']
print(df2)
color_dict = dict({'Iris-setosa':'black', 'Iris-versicolor':'red', 'Iris-virginica':'blue'})
sns.scatterplot(x='feature1', y='feature2', hue='Species', palette=color_dict, data=df2) # by scatter plot we see each cluster sample together.
plt.show()

