from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import Tkinter as Tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#---------End of imports

fig = plt.Figure()

x = np.arange(0, 2*np.pi, 0.01)        # x-array
xaxis=range(0,10)
yaxis=range(10,20)
xList = []
yList = []
def update(i):
	a,b=xaxis[i],yaxis[i]
	xList.append(xaxis[i])
        yList.append(yaxis[i])
for i in range(0,10):update(i)	
def animate(i):
    
    line.set_data(xList,yList) 
    return line, 
root = Tk.Tk()

label = Tk.Label(root,text="SHM Simulation").grid(column=0, row=0)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(column=0,row=1)

ax = fig.add_subplot(111)
line, = ax.plot(xList,yList)

ani = animation.FuncAnimation(fig, animate, interval=25)

Tk.mainloop()
