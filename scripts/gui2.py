import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import os,sys
import nn
import script
import test
import imageResize
import game

def newmain():
    my_w = tk.Tk()
    my_w.geometry("1400x800")  # Size of the window 
    my_w.configure(bg='blue')
    my_w.title('CatHacks IX')
    my_font1=('Comic Sans MS', 18, 'bold')
    my_font2=('Comic Sans MS', 12)

    # robot
    filename = './images/IMG_2144.png'
    img=Image.open(filename) # read the image file
    img=img.resize((550,800)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 = tk.Label(my_w, borderwidth=0)
    e1.grid(row=2,column=1)
    e1.place(relx=0.3, rely =0.5, anchor= CENTER)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image 


    l1 = tk.Label(my_w,text='SpaceGPT',width=30,font=my_font1, bg ='blue') 
    l1.grid(row=0,column=1,columnspan=4)
    l1.place(relx = 0.75, rely = 0.1, anchor= CENTER) 

    filename = './images/IMG_2147.png'
    img=Image.open(filename) # read the image file
    img=img.resize((300,200)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 = tk.Label(my_w, borderwidth=0)
    e1.grid(row=2,column=1)
    e1.place(relx=0.75, rely =0.4, anchor= CENTER)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image 

    b1 = tk.Button(my_w, text='Discover', bg = 'green', height = 1,
        width=8, font = my_font2, command = lambda:main())
    b1.grid(row=4,column=1)
    b1.place(relx = 0.75, rely = 0.4, anchor= CENTER) 

    filename = './images/IMG_2147.png'
    img=Image.open(filename) # read the image file
    img=img.resize((300,200)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 = tk.Label(my_w, borderwidth=0)
    e1.grid(row=2,column=1)
    e1.place(relx=0.75, rely =0.65, anchor= CENTER)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image 

    b3 = tk.Button(my_w, text='Play', bg = 'green', height = 1,
        width=8,font = my_font2, command = lambda:play_game())
    b3.grid(row=1, column=1)
    b3.place(relx = 0.75, rely = 0.65, anchor= CENTER) 

    my_w.mainloop()  # Keep the window open 

def main():
    my_w = tk.Toplevel()
    my_w.geometry("1400x800")  # Size of the window 
    my_w.configure(bg='blue')
    my_w.title('CatHacks IX')
    my_font1=('Comic Sans MS', 18, 'bold')
    my_font2=('Comic Sans MS', 12)

    filename = './images/IMG_2146.png' # train button
    img=Image.open(filename) # read the image file
    img=img.resize((250,150)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 = tk.Label(my_w, borderwidth=0)
    e1.grid(row=2,column=1)
    e1.place(relx=0.5, rely =0.8, anchor= CENTER)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image 

    b1 = tk.Button(my_w, text='Upload File', bg = 'red', height = 1,
        width=10, font = my_font2, command = lambda:upload_file(my_w))
    b1.grid(row=4,column=1)
    b1.place(relx = 0.5, rely = 0.8, anchor= CENTER) 
    
    filename = './images/IMG_2147.png'
    img=Image.open(filename) # read the image file
    img=img.resize((250,150)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 = tk.Label(my_w, borderwidth=0)
    e1.grid(row=2,column=1)
    e1.place(relx=0.5, rely =0.2, anchor= CENTER)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image 

    b3 = tk.Button(my_w, text='Go!', bg = 'green', height = 1,
        width=3,font = my_font2, command = lambda:go())
    b3.grid(row=1, column=1)
    b3.place(relx = 0.5, rely = 0.2, anchor= CENTER) 

    filename = '../GUI/Images/image_icon.png'
    img=Image.open(filename) # read the image file
    img=img.resize((50,50)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 = tk.Label(my_w)
    e1.grid(row=2,column=1)
    e1.place(relx=0.5, rely =0.5, anchor= CENTER)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image 
    
    filename = './images/IMG_2146.png' # train button
    img=Image.open(filename) # read the image file
    img=img.resize((250,150)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 = tk.Label(my_w, borderwidth=0)
    e1.grid(row=2,column=1)
    e1.place(relx=0.8, rely =0.5, anchor= CENTER)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image 

    b2 = tk.Button(my_w, text='Train Classification', height = 1,
        width=15,font = my_font2, bg = 'red', command = lambda:classify())
    b2.grid(row=3,column=4)
    b2.place(relx = 0.8, rely = 0.5, anchor= CENTER) 

    # my_w.mainloop()  # Keep the window open 

def play_game():
    my_font1=('Comic Sans MS', 18, 'bold')
    my_font2=('Comic Sans MS', 12)
    my_font3=('Comic Sans MS', 14)
    # Execute tkinter
    root = tk.Toplevel()
    root.configure(bg='blue')
   
    # Adjust size
    root.geometry("1000x700")

    lbl1 = tk.Label(root, text = "Question: ")
    lbl1.pack(side = "top", padx=50, pady=100)

    inputtxt = tk.Text(root,
                   height = 1,
                   width = 20)
    inputtxt.pack(side="top",padx=50, pady=50)

    lbl = tk.Label(root, text = "", bg="blue")
    

    def getInput():
        inp = inputtxt.get(1.0, "end-1c")
        ques = lbl1.cget("text")
        ans = game.check(ques, inp)
        if (ans):
            lbl.config(text = "Correct!")
        else:
            lbl.config(text = "Incorrect!")

    def ask():
        ques = game.getQuestion()
        lbl1.config(text = "Question: " + ques)
        lbl.config(text = "")

    printButton = tk.Button(root,
                        text = "Check", 
                        command = getInput)
   
    printButton.pack(side="top", padx=50, pady=50)

    askButton = tk.Button(root,
                        text = "Ask", 
                        command = ask)
    
    askButton.pack(side = "top", padx=50, pady=50)
    
    lbl.pack(side = "top",padx=50, pady=50)

    root.mainloop()
    ask()

def upload_file(my_w):
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png')]   # type of files to select 
    filename = tk.filedialog.askopenfilename(filetypes=f_types)
    col=1 # start from column 1
    row=2 # start from row 3 
    img=Image.open(filename) # read the image file
    img=img.resize((256*2,144*2)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 =tk.Label(my_w)
    e1.grid(row=row,column=col)
    e1.place(relx=0.5, rely =0.5, anchor= CENTER)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image   
    global url
    url=filename

def classify():
    my_font1=('Comic Sans MS', 18, 'bold')
    my_font2=('Comic Sans MS', 12)
    my_font3=('Comic Sans MS', 14)
    # Execute tkinter
    root = tk.Toplevel()
    root.configure(bg='blue')
   
    # Adjust size
    root.geometry("1000x700")

    b = tk.Button(root, text='Done', 
        width=20, font = my_font2, command = lambda:quit(root))
    root.update_idletasks()
    nn.main()
    b.grid(row=2,column=1)
    b.place(x=0,y=600)

def go():
    # Execute tkinter
    root = tk.Toplevel()
    root.configure(bg='blue')
    # Adjust size
    root.geometry("1400x800")
    root.title('CatHacks IX')
    my_font1=('Comic Sans MS', 18, 'bold')
    my_font2=('Comic Sans MS', 12)
    my_font3=('Comic Sans MS', 14)

    b = tk.Button(root, text='Back to Main', bg = 'red',
        width=20, font = my_font2, command = lambda:quit(root))
    b.grid(row=1,column=1)
    b.place(x=0,y=700)

    imageResize.upscale_image(url)
    imageResize.resize_image("./images/upscaled.jpg")
    filename = "./images/upscaled.jpg"
    img=Image.open(filename) # read the image file
    img=img.resize((256*3,144*3)) # new width & height
    img=ImageTk.PhotoImage(img)
    e1 = tk.Label(root)
    e1.grid(row=2,column=1)
    e1.place(relx=0.5, rely =0.4, anchor= CENTER)
    e1.image = img # keep a reference! by attaching it to a widget attribute
    e1['image']=img # Show Image 
    root.update_idletasks()
    # message = script.main("Mars")
    # print("Fun Facts!\n {}".format(message['content']))

    url2 = "./images/resized.jpg"
    pred = test.test_image(url2)
    print("Prediction: {}".format(pred))
    message = script.main(pred)
    print("Fun Facts!\n {}".format(message['content']))

    l5 = tk.Label(root,text=pred,width=30,font=my_font1) 
    l5.grid(row=0,column=1,columnspan=4)
    l5.place(relx = 0.5, rely = 0.6, anchor= CENTER) 
    
    l6 = tk.Label(root,text=message['content'],width=200,font=my_font2) 
    l6.grid(row=0,column=1,columnspan=20)
    l6.place(relx = 0.5, rely = 0.7, anchor= CENTER) 

    root.mainloop()
   
def quit(root):
    root.destroy()

if __name__ == '__main__':
    newmain()