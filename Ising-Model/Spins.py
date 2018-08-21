
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


#The data corresponds to a matrix in which each row vector corresponds to a 20x20 spin configuration
#at a certain temperature. The number of rows of this data matrix indicates the number of conficurations
#and the number of columns is 20**2

#the data was generated using the code found in https://github.com/tarod13/Monograph

#configurations with mode=0, the initial state is a random  matrix 
with open('spins_COP_NP.p', 'rb') as f:
    data = pickle.load(f)
with open('temperatures_COP_NP.p', 'rb') as f:
    temp = pickle.load(f)
    
#configurations with mode=1, the initial state is a perfectly divided matrix
with open('spins_COP_NP2.p', 'rb') as f:
    data1 = pickle.load(f)
with open('temperatures_COP_NP2.p', 'rb') as f:
    temp1 = pickle.load(f)


# In[59]:


n_config=np.shape(data)[0]
n_spins=len(data[0,:])

print(n_config)
print(n_spins)

M=[] #stores the total (scaled) magnetization of each configuration in terms of temperature
M1=[] #stores the total (scaled) magnetization of each configuration in terms of temperature for mode 1


#calculates magnetization for each configuration
for i in range(0,n_config):
    m=0
    m1=0
    m=np.sum(data[i,:])/n_spins
    m1=np.sum(data1[i,:])/n_spins
    
    M.append(m)
    M1.append(m1)  


# In[60]:


#data visualisation
#fig1, ax1 = plt.subplots(1,2)
#im1 = ax1[0].imshow(data[0,:].reshape(20,20))
#im2 = ax1[1].imshow(data1[0,:].reshape(20,20))
#fig1.colorbar(im1)
#fig1.gca().set_aspect('equal', adjustable='box')


# In[61]:


#plotting magnetization against temperature
#fig2, ax2 = plt.subplots(1,2)
#im2 = ax2[0].scatter(temp,M)
#im2=("M=0, random initial state")
#im2 = ax2[1].scatter(temp1,M1)
#ax2[0].set_title("M=0, random initial state")
#ax2[1].set_title("M=1, Prefectly divided initial state")


# In[62]:


p=np.polyfit(M,temp,4)
m=np.sort(np.asarray(M))
f=p[0]*m**4+p[1]*m**3+p[2]*m**2+p[3]*m+p[4]
plt.plot(m,f,color='r')
plt.scatter(M,temp)


# In[168]:


#separates the values that are outside the curve fit from the rest and produces labels for them
dropT=[]
dropM=[]
dropD=[]

newT=[]
newM=[]
newD=[]

for i in range(0,len(temp)):
    if(temp[i]<min(f)):
        dropT.append(temp[i])
        dropM.append(M[i])
        dropD.append(data[i,:])
    else:
        newT.append(temp[i])
        newM.append(M[i])
        newD.append(data[i,:])
   
labels=np.zeros(len(newM))


#calculates f again
pp=np.polyfit(newM,newT,4)
nm=np.sort(newM)
nf=pp[0]*nm**4+pp[1]*nm**3+pp[2]*nm**2+pp[3]*nm+pp[4]

for i in range(0,len(newM)):
    for k in range(0,len(newM)):
        if(nm[i]==newM[k]):
        #area outside the curve and above the x-axis, labeled as 1
            if((nf[i]<newT[k]) &(newM[k]>=0)):
                labels[k]=1
        #area outside the curve and below the x-axis, labeled as 2
            elif((nf[i]<newT[k]) & (newM[k]<0)):
                labels[k]=2
        #area inside the curve and above the x-axis, labeled as 2
            elif((nf[i]>=newT[k]) & (newM[k]>=0)):
                labels[k]=3
        #area inside the curve and above the x-axis, labeled as 2
            elif((nf[i]>=newT[k]) & (newM[k]<0)):
                labels[k]=4

#this is to check that every single point has been properly labeled
print(0 in labels)


# In[169]:


#checks that the labels are correct, receives the label
def check(l):
    cM=[]
    cT=[]
    for i in range(0,len(newT)):
        if(labels[i]==l):
            cT.append(newT[i])
            cM.append(newM[i])
    plt.scatter(cT,cM)
    plt.plot(f,m,color='red')

print(np.shape(newD))


# In[170]:


check(1)
check(2)
check(3)
check(4)


# In[207]:


#So each label will be asinged to a vector in the configuration matrix data and temp
#lets try with feeding it a matrix consisting of ising configurations remember, this is the
#labeled data so we should use newD
#data will be fed by taking each row as an input into a placeholder of dimensions len(systemsize)

n_labels=4 #there will be one output neuron for each label
hidden=200 #number of hidden units
size=n_spins #system size

#now we have to cast eache label for each configuration into a vector such that for every configuration in newdD
#there exist a corresponding vetor of the form yi=[1,0,0,0] if it is in zone 1, or yi=[0,2,0,0] if zone 2 and so on
#this will be our teacher vector
def build_teach(l):
    teacher=np.zeros(shape=(len(labels),n_labels))
    for i in range(len(labels)): #number of labeled configurations
        if(l[i]==1):
            teacher[i,0]=1.0
        elif(l[i]==2):
            teacher[i,1]=2.0
        elif(l[i]==3):
            teacher[i,2]=3.0
        else:
            teacher[i,3]=4.0
    return teacher

#initializes weights
def weig(shape):
    first= tf.random_normal(shape,stddev=0.2)
    return tf.Variable(first)

#initializes bias
def bi(shape):
    first = tf.constant(np.random.rand(), shape=shape)
    return tf.Variable(first)

#defines the output function of eachneuron in the layer
def layers(x, W,b):
  return tf.nn.sigmoid(tf.matmul(x, W)+b)


# In[208]:


teach=build_teach(labels) #this are the vectors that will be fed to the NN

#Now we create the model with the input vector and teacher vector. This is the input layer
x = tf.placeholder("float", shape=[None, size])
y = tf.placeholder("float", shape=[None, n_labels]) #teacher vector = label


#defining the hidden layer
W_1 = weig([size,hidden])
b_1 = bi([hidden])
lay_1 = layers(x, W_1,b_1)
#Output layer
W_2 = weig([hidden,n_labels])
b_2 = bi([n_labels])
y_hat=layers(lay_1, W_2,b_2) #our predicted value


# In[234]:


#Defining the cost function
error =0
error += (y-y_hat)**2
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

#predictions
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[240]:


out=np.zeros(shape=(len(newT),n_labels))


#starting a session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 500
    
    for i in range(epochs):
        train_accuracy = sess.run(accuracy,feed_dict={ x:newD, y: teach})
        #print("step %d, training accuracy %g"%(i, train_accuracy))
        
        sess.run(train,feed_dict={x:newD, y:teach})#feeding data to the place holders 
    print(train_accuracy)
    
    #now let's try checking the neuron outputs for each temperature
    for i in range(0,len(newT)):
         out=(sess.run(y_hat,feed_dict={x:newD, y:teach}))


# In[241]:


out2=np.sum(out,axis=1)
print(np.shape(out2))
#plt.scatter(newT,out2)
#plt.scatter(newT,labels,marker='X', color='red')
fig1, ax1 = plt.subplots(1,2)
im1 = ax1[0].scatter(newT,out2)
im2 = ax1[1].scatter( newT , labels)

