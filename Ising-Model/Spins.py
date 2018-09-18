
# coding: utf-8

# In[26]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[27]:


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


# In[28]:


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


# In[29]:


#data visualisation
#fig1, ax1 = plt.subplots(1,2)
#im1 = ax1[0].imshow(data[0,:].reshape(20,20))
#im2 = ax1[1].imshow(data1[0,:].reshape(20,20))
#fig1.colorbar(im1)
#fig1.gca().set_aspect('equal', adjustable='box')


# In[30]:


#plotting magnetization against temperature
#fig2, ax2 = plt.subplots(1,2)
#im2 = ax2[0].scatter(temp,M)
#im2=("M=0, random initial state")
#im2 = ax2[1].scatter(temp1,M1)
#ax2[0].set_title("M=0, random initial state")
#ax2[1].set_title("M=1, Prefectly divided initial state")


# In[31]:


p=np.polyfit(M,temp,4)
m=np.sort(np.asarray(M))
f=p[0]*m**4+p[1]*m**3+p[2]*m**2+p[3]*m+p[4]
plt.plot(m,f,color='r')
plt.scatter(M,temp)


# In[32]:


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


#Asigns labels
#for i in range(0,len(newM)):
#    for k in range(0,len(newM)):
#        if(nm[i]==newM[k]):
        #area outside the curve and above the x-axis, labeled as 1
#            if((nf[i]<newT[k]) &(newM[k]>=0)):
#                labels[k]=1
        #area outside the curve and below the x-axis, labeled as 2
#            elif((nf[i]<newT[k]) & (newM[k]<0)):
#                labels[k]=2
        #area inside the curve and above the x-axis, labeled as 2
#            elif((nf[i]>=newT[k]) & (newM[k]>=0)):
#                labels[k]=3
        #area inside the curve and above the x-axis, labeled as 2
#            elif((nf[i]>=newT[k]) & (newM[k]<0)):
#                labels[k]=4

#this is to check that every single point has been properly labeled
#print(0 in labels)


# In[33]:


#checks that the labels are correct, receives the label
#def check(l):
#    cM=[]
#    cT=[]
#    for i in range(0,len(newT)):
#        if(labels[i]==l):
#            cT.append(newT[i])
#            cM.append(newM[i])
#    plt.scatter(cT,cM)
#    plt.plot(f,m,color='red')

#print(np.shape(newD))


# In[34]:


#check(1)
#check(2)
#check(3)
#check(4)


# In[35]:


#So each label will be asinged to a vector in the configuration matrix data and temp
#lets try with feeding it a matrix consisting of ising configurations remember, this is the
#labeled data so we should use newD
#data will be fed by taking each row as an input into a placeholder of dimensions len(systemsize)

#n_labels=4 #there will be one output neuron for each label
#hidden=200 #number of hidden units
#size=n_spins #system size

#now we have to cast eache label for each configuration into a vector such that for every configuration in newdD
#there exist a corresponding vetor of the form yi=[1,0,0,0] if it is in zone 1, or yi=[0,2,0,0] if zone 2 and so on
#this will be our teacher vector
#def build_teach(l):
#    teacher=np.zeros(shape=(len(labels),n_labels))
#    for i in range(len(labels)): #number of labeled configurations
#        if(l[i]==1):
#            teacher[i,0]=1.0
#        elif(l[i]==2):
#            teacher[i,1]=2.0
#        elif(l[i]==3):
#            teacher[i,2]=3.0
#        else:
#            teacher[i,3]=4.0
#    return teacher

#initializes weights
def weig(shape):
    first= tf.random_normal(shape,stddev=0.2)
    return tf.Variable(first)

#initializes bias
def bi(shape):
    first = tf.constant(np.random.rand(), shape=shape)
    return tf.Variable(first)

#defines the output function of each neuron in the layer
def layers(x, W,b):
  return tf.nn.sigmoid(tf.matmul(x, W)+b)


# In[36]:


#teach=build_teach(labels) #this are the vectors that will be fed to the NN

#Now we create the model with the input vector and teacher vector. This is the input layer
#x = tf.placeholder("float", shape=[None, size])
#y = tf.placeholder("float", shape=[None, n_labels]) #teacher vector = label


#defining the hidden layer
#W_1 = weig([size,hidden])
#b_1 = bi([hidden])
#lay_1 = layers(x, W_1,b_1)
#Output layer
#W_2 = weig([hidden,n_labels])
#b_2 = bi([n_labels])
#y_hat=layers(lay_1, W_2,b_2) #our predicted value


# In[12]:


#Defining the cost function
#error =0
#error += (y-y_hat)**2

#optimizer = tf.train.AdamOptimizer(learning_rate=0.001) I have tested the adamoptimizer and it is even worse
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#train = optimizer.minimize(error)


#predictions
#correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[13]:


#out=np.zeros(shape=(len(newT),n_labels))
#outav=np.zeros(shape=(len(newT)))
#outi = np.zeros(shape=(len(newT)))
#starting a session 
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
    
#    sess.run(init)
    
#    epochs = 900
    
#    for i in range(epochs):
#        train_accuracy = sess.run(accuracy,feed_dict={ x:newD, y: teach})
        #print("step %d, training accuracy %g"%(i, train_accuracy))
        
#        sess.run(train,feed_dict={x:newD, y:teach})#feeding data to the place holders 
        #print(train_accuracy)
#    print("Accuracy:", accuracy.eval({ x: newD, y:teach}))

    
    #now let's try checking the neuron outputs for each temperature
#    out=(sess.run(y_hat,feed_dict={x:newD, y:teach}))
#    for i in range(len(newT)):
#        av=0.0
#        res=sess.run(y_hat,feed_dict={x: newD, y: teach})
#        av=av+res 
        #print ii, res  
#        av=av
#    outav=av


# In[14]:


#out2=np.sum(out,axis=1)
#print(np.shape(out2))
#plt.scatter(newT,out2)
#plt.scatter(newT,labels,marker='X', color='red')
#fig1, ax1 = plt.subplots(1,2)
#im1 = ax1[0].scatter(newT,out2)
#im2 = ax1[1].scatter( newT , labels)


# In[15]:


#PROBEMOS QUE PASA SI INTENTO UNA CLASIFICACION BINARIA, OSEA EL VALOR ABS DE LA MAGNETIZACION CON 2 REGIONES
#ANOTHER PROBLEM COULD LIE IN THE WAY WE CALCULATE THE ERRORS, WE WANT TO MINIMIZE THE ERROR BETWEEN COMPONENTS OF
#THE TEACHER AND PREDICTIONS VERCTOR
#PRINT OUTPUT 1-1
#MAYBE THE THING IS IN THIS TRUNCATED NORMAL METHOD BECAUSE INITIALIZING THE WEIGHTS WITH A RANDOM DIST NEAR ZERO 
# IS NOT OPTIMAL, COULD DISTORT THE ACTIVATION FUNCTION
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))∫
#output of neural net
#DEPRONTO LA ACCUARACY ESTA MAL CALCULADA
#neuron1=(outav[:,0])
#neuron2=(outav[:,1])
#plt.scatter(newT,neuron2)


# In[16]:


#Cambiando las labels a temperatura. La idea es que todo antes de la temperatura crítica sea por ejemplo [1,0] y
#todo por encima [0,1]. De nuevo usamos newT, newD y newM

#represents two output neurons [1,0] means ober critical T and [0,1] means under critical T
labels_T=np.zeros((len(newT),2)) 

#asigns temperature with respect to critical temperature
def T_labels(Ma,Te,crit,labels):
    T=np.array(Te)
    M = np.array(Ma)
    ii= T< crit
    jj = T>=crit
    #under Tc
    labels[ii,:]=[0,1]
    labels[jj,:]=[1,0]
    return T[ii], M[ii], T[jj], M[jj]
    
Tc=2.2691853

T_small, M_small, T_big, M_big = T_labels(newM, newT, Tc,labels_T)

#plt.scatter(T_less,M_less)
#plt.scatter(T_big, M_big)
#plt.axvline(x=Tc, color='red')

fig, subp=plt.subplots(2,1,figsize=(8,6)) #crea 2 subplots en una columna
#fig es la entidad figura y subp es un array que en cada posicion guarda el subplot
#subp se puede manipular como un array para modificar individualmente los subplots
subp[0].scatter(T_small,M_small)
subp[0].scatter(T_big, M_big)
subp[0].set_title('Temperature separation')
subp[0].axvline(x=Tc, color='red')
subp[1].scatter(newT,labels_T[:,0], label='High T Neuron')
subp[1].scatter(newT,labels_T[:,1], label='Low T Neuron')
subp[1].set_title('Label output')
subp[1].legend()
subp[1].axvline(x=Tc, color='red')
subp
fig.tight_layout() #ajusta la distancia entre plots


# In[17]:


n_labels=2 #there will be one output neuron for each label
hidden=200 #number of hidden units
size=n_spins #system size


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

#Defining the cost function
#error =0
#error += (y-y_hat)**2
error =0
error += (y-y_hat)**2
#adds an l2 regularization term to with a beta value
beta=0.01
error = tf.reduce_mean(error + beta *( tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)


#predictions
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[18]:


saver= tf.train.Saver()


# In[19]:


#Varable outputs used to access data from the session tensors
out=np.zeros(shape=(len(newT),n_labels))
outav=np.zeros(shape=(len(newT)))
outac = np.zeros(shape=(len(newT)))

#starting a session 
init = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 50000
    #training cycle
    
    for i in range(epochs):
        train_accuracy = sess.run(accuracy,feed_dict={ x:newD, y: labels_T})      
        sess.run(train,feed_dict={x:newD, y:labels_T})#feeding data to the place holders 
        #stops training when accuracy crosses a certain threshold
        if(train_accuracy>=0.95):
            print(train_accuracy)
            print(i)
            break;
    print("Accuracy:", accuracy.eval({ x: newD, y:labels_T}))

    
    #now let's try checking the neuron outputs for each temperature
    out=(sess.run(y_hat,feed_dict={x:newD, y:labels_T}))
    for i in range(len(newT)):
        res=sess.run(y_hat,feed_dict={x: newD, y: labels_T})
        
    outav=res #neuron outputs
    
    saver.save(sess, 'ising.ckpt')


# In[20]:


plt.scatter(newT, outav[:,0])
plt.scatter(newT,outav[:,1])
plt.axvline(Tc, color='red')


# In[21]:


#Loads test data
with open('spins_COP_test.p', 'rb') as f:
    test_D = pickle.load(f)
with open('temperatures_COP_test.p', 'rb') as f:
    test_T = pickle.load(f)


# In[22]:


test_config=np.shape(test_D)[0]
test_spins=len(test_D[0,:])

print(test_config)
print(test_spins)

test_M=[] #stores the total (scaled) magnetization of each configuration in terms of temperature

#calculates magnetization for each configuration
for i in range(0,n_config):
    m=0
    m=np.sum(test_D[i,:])/n_spins  
    test_M.append(m)


# In[23]:


#represents two output neurons [1,0] means ober critical T and [0,1] means under critical T
labels_test=np.zeros((len(test_T),2)) 

T_small2, M_small2, T_big2, M_big2 = T_labels(test_M, test_T, Tc,labels_test)

plt.scatter(T_small2,M_small2)
plt.scatter(T_big2,M_big2)
plt.axvline(Tc, color='red')


# In[24]:


#evaluates the trained network with the new unlabeled data
with tf.Session() as sess:
    #restores the model
    saver.restore(sess, 'ising.ckpt')
    for i in range(len(test_T)):
        output= sess.run(y_hat,feed_dict={x:test_D})
    print('accuracy=',  accuracy.eval({ x: test_D, y:labels_test}))


# In[25]:


print(np.shape(output))
plt.scatter(test_T, output[:,0])
plt.scatter(test_T, output[:,1])

