import sys
import numpy as np
import matplotlib.pyplot as plt
def plotDataPointsAndClassifier(XList1, YList):
	plt.plot(XList1, YList[0], color = 'black', label='Cancer Voted Perceptron')
	plt.plot(XList1, YList[1], color = 'green', label='Cancer Vanilla Perceptron')
	plt.plot(XList1, YList[2], color = 'blue', label='Iono Voted Perceptron')
	plt.plot(XList1, YList[3], color = 'red', label='Iono Vanilla Perceptron')
	plt.plot(XList1, YList[0],'ro',XList1, YList[1],'ro',XList1, YList[2],'ro',XList1, YList[3],'ro')

	plt.axis([0, 50, 80, 100])
	plt.legend(loc='best') 
	plt.xlabel('Epochs', fontsize=18)
	plt.ylabel('Accuracy', fontsize=18)
	plt.show()

X_plot=[]
Y_plot=[]
for mode in ["breast", "iono"]:
	file_name=""

	if mode=="breast":
		file_name="../dataset/breast-cancer-wisconsin.data"
	elif mode == "iono":
		file_name="../dataset/ionosphere.data"
	x=[]
	y=[]



	with open(file_name) as f:
		for line in f:
			if mode=="breast":
				if '?' in line:
					continue
				line_parts=line.split(',')
				for i in range(0,len(line_parts)):
					line_parts[i] = float(line_parts[i].split('\n')[0])
				x.append(line_parts[1:10])
				if line_parts[10]==2:
					y.append(1)
				if line_parts[10]==4:
					y.append(-1)
		
			elif mode=="iono":
				line_parts=line.split(',')
				for i in range(0,len(line_parts)-1):
					line_parts[i] = float(line_parts[i].split('\n')[0])

				x.append(line_parts[0:34])
				if line_parts[34]=='g\n':
					y.append(1)
				if line_parts[34]=='b\n':
					y.append(-1)
			else:
				print("enter the right mode")
				exit(0)

	chunk_size = int(len(x)/10)
	epochs_catch = [10,15,20,25,30,35,40,45,50]
	# epochs_catch = [10,15,20,25]
	num_epochs = max(epochs_catch)
	accuracy_matrix=[]
	for cross_val in range(0,10):

		accuracy_array = []
		#ranges for the test set
		start_ind = cross_val*chunk_size
		end_ind = (cross_val+1)*chunk_size 
		if cross_val == 9:
			end_ind = len(x)

		x_test=[]
		y_test=[]
		x_train=[]
		y_train=[]
		
		for i in range(start_ind,end_ind):
			x_test.append(x[i])
			y_test.append(y[i])
		
		for i in range(0,start_ind):
			x_train.append(x[i])
			y_train.append(y[i])
		
		for i in range(end_ind,len(x)):
			x_train.append(x[i])
			y_train.append(y[i])


		x_train=np.array(x_train)
		y_train=np.array(y_train)
		x_test=np.array(x_test)
		y_test=np.array(y_test)
		# n = length of output - 1 

		n=0
		output=[]
		output.append([np.zeros(len(x_train[0])),0,1])
		c1=0
		c2=0
		for iter in range(0,num_epochs):
			for i in range(0,len(x_train)):
				if y_train[i]*(output[n][1] + np.dot(output[n][0],x_train[i])) <= 0:
					output.append(output[n][:])
					n+=1
					output[n][0] = output[n][0] + y_train[i]*x_train[i] #w
					output[n][1]+=y_train[i] #b
					output[n][2]=1 #c
					c1+=1
				else:
					c2+=1
					output[n][2]+=1

			if (iter+1) in epochs_catch:
				# print(iter+1,c1,c2)
				correct=0
				incorrect=0
				correct_v=0
				incorrect_v=0
				for ind in range(0,len(x_test)):
					s=0
					sig=1
					for i in range(0,n+1):
						if (np.dot(output[i][0],x_test[ind]) + output[i][1]) > 0:
							sig=+1
						else:
							sig=-1
						s+= output[i][2]*sig
					if sig == y_test[ind]:
						correct_v+=1
					else:
						incorrect_v+=1

					voted_pred=0
					if s>0:
						voted_pred=1
					else:
						voted_pred=-1

					if voted_pred == y_test[ind]:
						correct+=1
					else:
						incorrect+=1
				accuracy_array.append((float(correct/(correct+incorrect)),\
					(float(correct_v/(correct_v+incorrect_v)))))
		accuracy_matrix.append(accuracy_array)




	# print(accuracy_matrix)
	print('epoch','\t','Voted','\t','Vanila')
	y_voted=[]
	y_vanilla=[]
	X_plot = (epochs_catch)
	for j in range(0,len(epochs_catch)):
		sum=0
		sum_v=0
		for i in range(0,10):
			sum+=accuracy_matrix[i][j][0]
			sum_v+=accuracy_matrix[i][j][1]
		sum=sum*10
		sum_v=sum_v*10
		print(epochs_catch[j],'\t',round(sum,2), '\t', round(sum_v,2))
		y_voted.append(round(sum,2))
		y_vanilla.append(round(sum_v,2))
		
	Y_plot.append(y_voted)
	Y_plot.append(y_vanilla)

plotDataPointsAndClassifier(X_plot, Y_plot)

