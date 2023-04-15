import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

my_w = tk.Tk()
my_w.geometry("700x700")  # Size of the window 
my_w.title('CatHacks IX')
my_font1=('Comic Sans MS', 18, 'bold')
my_font2=('Comic Sans MS', 12)
my_font3=('Comic Sans MS', 14)

epochs = '2'
accuracy = '99.6%'

l1 = tk.Label(my_w,text='Upload Files & Train Models',width=30,font=my_font1)  
l2 = tk.Label(my_w,text='Current Model:\nEpochs:' + epochs + '\nAccuracy:' + accuracy +'%', font=my_font2)  

b1 = tk.Button(my_w, text='Upload File', 
    width=20, font = my_font2, command = lambda:upload_file())

b2 = tk.Button(my_w, text='Train Classification', 
    width=20,font = my_font2, command = lambda:classify())

b3 = tk.Button(my_w, text='Go!', 
    width=20,font = my_font3, command = lambda:go())

l1.grid(row=0,column=1,columnspan=4)
l2.grid(row=2,column=4)
b1.grid(row=4,column=1)
b2.grid(row=3,column=4)
b3.grid(row=1, column=1)

filename = 'Images/sample.png'
img=Image.open(filename) # read the image file
img=img.resize((200,200)) # new width & height
img=ImageTk.PhotoImage(img)
e1 = tk.Label(my_w)
e1.grid(row=2,column=1)
e1.image = img # keep a reference! by attaching it to a widget attribute
e1['image']=img # Show Image  

def upload_file():
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png')]   # type of files to select 
    filename = tk.filedialog.askopenfilename(filetypes=f_types)
    col=1 # start from column 1
    row=2 # start from row 3 
    img=Image.open(filename) # read the image file
    img=img.resize((200,200)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 =tk.Label(my_w)
    e1.grid(row=row,column=col)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image     

def upscale():
    # Execute tkinter
    root = Tk()
    # Adjust size
    root.geometry("400x400")
    b = tk.Button(root, text='Done', 
        width=20, font = my_font2, command = lambda:quit(root))
    b.grid(row=2,column=1)

def classify():
    # Execute tkinter
    root = Tk()
    # Adjust size
    root.geometry("400x400")
    b = tk.Button(root, text='Done', 
        width=20, font = my_font2, command = lambda:quit(root))
    b.grid(row=2,column=1)

def go():
    # Execute tkinter
    root = Tk()
    # Adjust size
    root.geometry("600x600")
    root.title('CatHacks IX')
    my_font1=('Comic Sans MS', 18, 'bold')
    my_font2=('Comic Sans MS', 12)
    my_font3=('Comic Sans MS', 14)

    b = tk.Button(root, text='Back to Main', 
        width=20, font = my_font2, command = lambda:quit(root))
    b.grid(row=1,column=1)


def quit(root):
    root.destroy()

my_w.mainloop()  # Keep the window open