import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import keras

# fix random seed for reproducibility
seed = 7
k=25
np.random.seed(seed)
UL=[]
# load dataset
dataframe = pd.read_csv("iris.csv", header=None)
dataset = dataframe.values
Y = dataset[:,4]
X = np.array(dataset)[:,0:4].astype(float)
#KRNN=[[]]*len(X)#np.zeros((len(X),len(X)),np.int)
KRNN=[]
for i in range(len(X)):
	KRNN.append([])
print(KRNN)




def dist(i,j):
	return ((X[i,0]-X[j,0])**2+(X[i,1]-X[j,1])**2+(X[i,2]-X[j,2])**2+(X[i,3]-X[j,3])**2)**0.5

DIST=[]

for i in range(len(X)):
	for j in range(len(X)):
		if i != j:
			DIST.append([dist(i,j),j])
	DIST.sort()
	#print(DIST)
	#break
	for j in range(min(k,len(DIST))):
		#print(DIST[j],len(KRNN))
		KRNN[DIST[j][1]].append(i)


def diversity(x,y):
	cnt=0
	for i in range(len(KRNN[x])):
		if KRNN[x][i] in KRNN[y]:
			cnt+=1
	return 1-((1+cnt)/((1+len(KRNN[x]))*(1+len(KRNN[y])))**0.5) 



	#print(len(KRNN[i]))
#for i in range(len(KRNN)):
#	print(len(KRNN[i]))
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(dummy_y)
for i in range(len(dataset)):
	UL.append(i)
#dataset= pd.concat([dataset[:,0:4],dummy_y],axis =1)
L=[]

L.append(UL[0])
del UL[0]

acc=0
while(acc<95 or len(L)<100):
	X_tr=np.zeros((len(L),4),dtype=np.float64)
	Y_tr=np.zeros((len(L),3),dtype=np.float64)
	for i in range(len(L)):
		#print(X[L[i]])
		X_tr[i]=X[L[i]]
		Y_tr[i]=dummy_y[L[i]]

	X_UL=np.zeros((len(UL),4),dtype=np.float64)
	for i in range(len(UL)):
		#print(X[L[i]])
		X_UL[i]=X[UL[i]]
		#Y_[i]=dummy_y[L[i]]

	# encode class values as integers


	# define baseline model
		# create model
	model.fit(X_tr,Y_tr,epochs=20,batch_size=5)
	pred=model.predict(X_UL)
	uncrt=[]
	for i in range(len(pred)):
		pred[i].sort()
		uncrt.append(1-(pred[i,2]-pred[i,1]))
	
	density=[]
	for i in range(len(UL)):
		cnt=0
		for j in range(len(KRNN[UL[i]])):
			if KRNN[UL[i]][j] in L:
				cnt+=1
		#print(len(KRNN[UL[i]]))
		density.append((1+len(KRNN[UL[i]])-cnt)/(1+cnt))
	print(density)



	break
	



#estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)

#kfold = KFold(n_splits=min(len(X_tr)+1,10), shuffle=True, random_state=seed)

#results = cross_val_score(estimator, X_tr, Y_tr, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))










