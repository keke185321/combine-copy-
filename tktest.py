import Tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from matplotlib import pyplot as plt
#camera = Camera(camera=0)
width, height = 400, 300
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()
w, h = 300, 200
canvas = tk.Canvas(root, width=w, height=h)
canvas.pack()
#lmain1 = tk.Label(root)
#lmain1.pack(side=tk.LEFT)
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    #lmain.after(10, show_frame)
    #lmain1.imgtk = imgtk
    #lmain1.configure(image=imgtk)
    #lmain1.after(10, show_frame)
show_frame()
X = np.linspace(0, 2 * np.pi, 50)
Y = np.sin(X)

# Create the figure we desire to add to an existing canvas
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax.plot(X, Y)

# Keep this handle alive, or else figure will disappear
fig_x, fig_y = 100, 100
fig_photo = draw_figure(canvas, fig, loc=(fig_x, fig_y))
fig_w, fig_h = fig_photo.width(), fig_photo.height()

# Add more elements to the canvas, potentially on top of the figure
canvas.create_line(200, 50, fig_x + fig_w / 2, fig_y + fig_h / 2)
canvas.create_text(200, 50, text="Zero-crossing", anchor="s")

#lmain1.after(10, show_frame)
root.mainloop()
