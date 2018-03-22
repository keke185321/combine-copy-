# load and plot dataset
import numpy
#import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pylab
from matplotlib.pyplot import show
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Tkinter
look_back = 200
class PulsePre:
	def __init__(self,root):
		self.plt=plt
		self.plt.subplots_adjust(left=0.5, bottom=0.8, right=0.9, top=0.9, wspace=0.9, hspace=0.9)
		self.plt.ion()
		self.model=[]
		self.oripredict=[]
		self.oriplot=[]
		self.x1=0
		self.t,self.v,self.v2,self.v3=[],[],[],[]
		#self.plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
		frame = Tkinter.Frame(root)
		self.fig = Figure()
		#self.fig.canvas.set_window_title('Predicted Heart Rate vs Real time Heart Rate')
		self.ax1 = self.fig.add_subplot(221)
		self.ax1.set_title("Predicted Heart Rate")
		self.ax2 = self.fig.add_subplot(222)
		self.ax2.set_title("Real time Heart Rate")
		self.ax3 = self.fig.add_subplot(223)
		self.ax3.set_title("Difference between Heart Rate")
		self.ax3.set_position([0.1,0.05, 0.5, 0.35])
		#self.ax3.subplots_adjust(left=0.5, bottom=0.8, right=0.9, top=0.9, wspace=0.9, hspace=0.9)
		self.line, = self.ax1.plot(self.t,self.v, linestyle="-", color="r")
		self.lineone, = self.ax2.plot(self.t,self.v2, linestyle="-", color="r")
		self.linetwo, = self.ax3.plot(self.t,self.v3, linestyle="-", color="r")
		self.canvas = FigureCanvasTkAgg(self.fig,master=root)
		self.canvas.show()
		self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
		frame.pack()
		self.score=0
	def floatconv(self,pulse):
		for i in pulse:
			if i !='':
				float(i)
			else:
				del i
		return pulse

	def create_dataset(self,dataset, look_back):
		dataX, dataY = [], []
		for i in range(len(dataset)-look_back-1):
			a = dataset[i:(i+look_back), 0]
			dataX.append(a)
			dataY.append(dataset[i + look_back, 0])
		return numpy.array(dataX), numpy.array(dataY)


	def rewrite(self):
		fo  = open('pulse.txt','r')
		n = open('pulse.txt', 'a')
		pulse=fo.read().split(' ')
		rm=open('pulse.txt', 'w').close()
		del pulse[-1],pulse[0:999]
		for i in pulse:
			n.write(i+' ')



	def updateData(self,num):
		#del firstdata
		fo  = open('pulse.txt','r')
		pulse=fo.read().split(' ')
		num=-num-200
		del pulse[-1],pulse[0:num]
		pulse=self.floatconv(pulse)
		pulselast=pulse[-1]
	        #print 'lat: ',pulselast
		#time=[]
		a=0
		'''for i in pulse:
			time.append(round(a,1))
			a+=0.3'''
		pulse=numpy.array(pulse)
		pulse= numpy.reshape(pulse, (-1, 1))
		scaler = MinMaxScaler(feature_range=(0, 1))
		dataset = scaler.fit_transform(pulse)
		
		# split into train and test sets
		train_size = int(len(dataset) * 0.67)
		test_size = len(dataset) - train_size
		train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
		# reshape into X=t and Y=t+1
		#look_back = 200#predict for future 1 minute
		trainX, trainY = self.create_dataset(train, look_back)
		testX, testY = self.create_dataset(test, look_back)
		datasetX, datasetY = self.create_dataset(dataset, look_back)
		oritestX=datasetX
		trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
		testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
		return trainX,trainY,testX,testY,oritestX,scaler,pulselast

	def trainmodel(self,trainX,trainY,testX,testY,scaler):
		# create and fit the LSTM network
		model = Sequential()
		model.add(LSTM(4, input_shape=(1, look_back)))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

		# make predictions
		trainPredict = model.predict(trainX)
		testPredict = model.predict(testX)
		#scaler = MinMaxScaler(feature_range=(0, 1))
		# invert predictions
		trainPredict = scaler.inverse_transform(trainPredict)
		trainY = scaler.inverse_transform([trainY])
		testPredict = scaler.inverse_transform(testPredict)
		testY = scaler.inverse_transform([testY])

		# calculate root mean squared error
		trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
		#print('Train Score: %.2f RMSE' % (trainScore))
		testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
		#print('Test Score: %.2f RMSE' % (testScore))
		return model


	def predictapp(self,num,oritestX,model,scaler):
		futureX=oritestX
		futureX = numpy.reshape(futureX, (futureX.shape[0], 1, futureX.shape[1]))
		#print futureX
		futurePredict=model.predict(futureX)
		futurePredict=scaler.inverse_transform(futurePredict)
		futurePredict=futurePredict.ravel()
		return futurePredict
	
	def p(self,a, b,oritestX,pulselast):
	    self.t.append(a)
	    self.v.append(b)
	    self.v2.append(numpy.array(float(pulselast)))
	    self.v3.append(abs(float(pulselast)-b))
	    #print 'v ',self.v
	    #print 'v2 ',self.v2
	    self.ax1.set_xlim(min(self.t), max(self.t) + 1)
	    self.ax1.set_ylim(0, 100)
  	    self.ax2.set_xlim(min(self.t), max(self.t) + 1)
	    self.ax2.set_ylim(0,100)
	    self.ax3.set_xlim(min(self.t), max(self.t) + 1)
	    self.ax3.set_ylim(0,20)
	    self.line.set_data(self.t, self.v)
	    #self.plt.pause(0.001)
	    #self.ax1.figure.canvas.draw()
  	    self.lineone.set_data(self.t, self.v2)
	    self.linetwo.set_data(self.t, self.v3)
	    self.canvas.draw()
	    #self.ax2.figure.canvas.draw()
	# shift train predictions for plotting
	def accuracy(self):
	    self.score = math.sqrt(mean_squared_error(self.v, self.v2))
	    return self.score
	def plotpulse(self,num):
		trainX,trainY,testX,testY,oritestX,scaler,pulselast=self.updateData(num)
		if num==-201: 
			self.model.append(self.trainmodel(trainX,trainY,testX,testY,scaler))
		if (num+201)%(-100)==0 and num!=-201: #update accurary every 30s
			score=self.accuracy()
		futurePredict=self.predictapp(num,oritestX[-200:],self.model[0],scaler)
		self.p(self.x1,futurePredict[-200],oritestX,pulselast)
		self.x1=self.x1+0.005
		return self.score


