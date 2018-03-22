'''import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (5,3)

fig = plt.figure()
fig.add_subplot(121)

ax = fig.add_subplot(122)
ax.set_title("subplots")


plt.show()'''
import Tkinter
import pulsepre as pu
root = Tkinter.Tk()
pul=pu.PulsePre(root)

a=range(-201,-502,-1)
for i in a:
	pul.plotpulse(i)
root.mainloop()
