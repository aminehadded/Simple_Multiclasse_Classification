


import tensorflow as tf


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn import datasets


# In[21]:


n_pts = 500 #nombre des points en entrée
center = [[-1, 1], [-1, -1], [1, -1]] # les centers de chaque classe 
X, y= datasets.make_blobs(n_samples=n_pts, random_state=123, centers=center, cluster_std=0.4)
#n_sample=n_pts: nombre de pts à utiliser 
#random_state=123: Permets d'obtenir toujours les mêmes nombres aléatoires
#centers=center : les centres de chaque classe
#cluster_std: la deviation dans les classe par rapport aux centres 
#X: data 
#y : label (0, 1, 2) indicateur de chau=que classe 
#print(X)
#print(y)
plt.scatter(X[y==0, 0], X[y==0, 1]) # plot les pts de classe 0
plt.scatter(X[y==1, 0], X[y==1, 1]) # plot les pts de classe 1
plt.scatter(X[y==2, 0], X[y==2, 1]) # plot les pts de classe 2


# In[15]:


y_cat = to_categorical(y, 3)
print(y_cat)


# In[17]:


model = Sequential() 
model.add(Dense(units=3, input_shape=(2,), activation='softmax'))
#units=3 : nombre des neouds de sortie
#input_shape: nombre des neouds en entrée
#activation='softmax' : activation des neouds de sortie (appliquée au niveau de multiclasse classification)
model.compile(Adam(0.1), loss='categorical_crossentropy', metrics=['accuracy'])
#Adam: optimisation avec un learning rate de 0.1
#loss: erreur categorical_crossentropy par rapport aux classes utiliser d'aprés pour l'entrainement de réseauu
#metrics=['accuracy'] on veut afficher accuracy


# In[18]:


model.fit(x=X, y=y_cat, verbose=1, batch_size= 50, epochs=100) 
#feedforward + feedbackward (Entrainement de réseau des neurones)


# In[19]:


def plot_descision_limite(X, y_cat, model):# permet de donner une indication par coleurs les zones de chaque classe
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1) # prendre 50 points linéarement équidistant sur l'axe x horizantale
    y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1) # prendre 50 points linéarement équidistant sur l'axe y verticale
    #print(y_span)
    xx, yy=np.meshgrid(x_span, y_span) 
    # xx est un array de deux dimensions (50, 50) dont tous les lignes sont les memes (x_span)
    # yy est un array de deux dimensions (50, 50) dont tous les colones sont les memes (y_span)
    #print (xx)
    #print (yy)
    xx_, yy_=xx.ravel(), yy.ravel() # conversion en une seule dimension
    #print(yy_)
    #print(xx_)
    grid = np.c_[xx_, yy_] # concatination de deux array chaque 50 elements de xx_ correspond à une valeur de yy-
    #print(grid)
    
    #le but de tous ces fonctions est de préparer un matrice qui contient plusieurs combinaison possibles entre l'axe x et l'axe y 
    #afin de tester les predictions de réseau de neurones 
    
    pred=model.predict_classes(grid) #prediction
    A=pred.reshape(xx.shape) # Redimensionnement en un array de deux dimension 
    print(A)
    plt.contourf(xx, yy, A) # tracage des contours selon les valeurs de prediction 


# In[20]:


plot_descision_limite(X, y, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])

