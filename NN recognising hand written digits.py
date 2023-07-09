import pandas as pd
import numpy as np

datas=pd.read_csv('digit-dataset-train.csv')
datas=np.array(datas)
m,n=datas.shape
np.random.shuffle(datas)

dev_datas=datas[0:1000].T
dev_y=dev_datas[0]
dev_x=dev_datas[1:n]
dev_x=dev_x/255

train_datas=datas[1000:m].T
train_y=train_datas[0]
train_x=train_datas[1:n]
train_x=train_x/255

def init_params():
    W1=np.random.rand(10,784)
    b1=np.random.rand(10,1)
    W2=np.random.rand(10,10)
    b2=np.random.rand(10,1)
    return W1,b1,W2,b2

def leaky_reLU(Z,alpha_=0.01):
    return np.maximum(Z,Z*alpha_)

def softmax(Z,epsilon_=1e-4):
    for i in range(len(Z)):
        Z[i]=np.array(list(map(lambda x: 700 if x>700 else(-700 if x<-700 else (epsilon_ if x>0 and x<epsilon_ else (-epsilon_ if x<0 and x>-epsilon_ else x))),Z[i])))
    return (np.exp(Z)/sum(np.exp(Z)))

def d_leaky_reLU(Z,alpha_=0.01):
    return np.where(Z>0,1,alpha_)

def forward_propagation(W1,b1,W2,b2,X):
    Z1=W1.dot(X)+b1
    A1=leaky_reLU(Z1)
    Z2=W2.dot(A1)+b2
    A2=softmax(Z2)
    return Z1,A1,Z2,A2

def onehot(Y):
    onehot_Y=np.zeros((Y.size,Y.max()+1))
    onehot_Y[np.arange(Y.size),Y]=1
    onehot_Y=onehot_Y.T
    return onehot_Y

def back_propagation(Z1,A1,Z2,A2,W2,X,Y):
    m=Y.size
    Y=onehot(Y)
    dZ2=A2-Y
    dW2=(1/m)*(dZ2.dot(A1.T))
    db2=(1/m)*(np.sum(dZ2))
    dZ1=W2.T.dot(dZ2)*(d_leaky_reLU(Z1))
    dW1=(1/m)*(dZ1.dot(X.T))
    db1=(1/m)*(np.sum(dZ1))
    return dW1,db1,dW2,db2

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,learnign_rate):
    W1=W1-(learnign_rate*dW1)
    b1=b1-(learnign_rate*db1)
    W2=W2-(learnign_rate*dW2)
    b2=b2-(learnign_rate*db2)
    return W1,b1,W2,b2

def get_predictions(X):
    return np.argmax(X,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return (np.sum(predictions==Y)/Y.size)

def lr_decay(learning_rate,iteration,decayRate=0.998):
    if iteration>500 and iteration<800:
        return 1.9
    if iteration>=800 and iteration<1100:
        return 1.8
    if iteration>=1100:
        return 1
    return learning_rate

def gradient_descent(X,Y,iterations,learning_rate):
    W1,b1,W2,b2=init_params()
    for i in range(iterations+1):
        Z1,A1,Z2,A2=forward_propagation(W1,b1,W2,b2,X)
        dW1,db1,dW2,db2=back_propagation(Z1,A1,Z2,A2,W2,X,Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr_decay(learning_rate, i))
        if i%200==0:
            print('Iteration: ',i,' Learning Rate: ',lr_decay(learning_rate,i))
            print('Accuracy: ',(get_accuracy(get_predictions(A2),Y)*100),'%')
    return W1,b1,W2,b2

def make_prediction(X,W1,b1,W2,b2):
    _,_,_,A2=forward_propagation(W1,b1,W2,b2,X)
    predictions=get_predictions(A2)
    return predictions

def Predictions(X,Y,W1,b1,W2,b2):
    predictions=make_prediction(X,W1,b1,W2,b2)
    print('Dev-set Accuracy: ',(get_accuracy(predictions,Y)*100),'%')

W1,b1,W2,b2=gradient_descent(train_x,train_y,1600,2)
Predictions(dev_x,dev_y,W1,b1,W2,b2)
