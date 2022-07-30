from tkinter import *
from tkinter.filedialog import asksaveasfile
import speech_recognition as sr
from timeit import default_timer
from threading import *
import pyttsx3
import time
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont 
from torchvision.utils import make_grid
import sys
import os
import glob 
import winsound
import pickle as pkl
import json
import torch
import torch as th
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.append(".\\utils\\")
from text2face import text2face
from siamese import *

engine = pyttsx3.init()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

back_imgs=[]
cur_img=None
imgToInsert = None
imgWithInfo = None
frwd_imgs=[]
threads=[]
nfi=0
nbi=0
state=0
caption=""
toggle=False
monitor_state={
         "button1":[],
         "button2":[],
         "button3":[],
         "button4":[],
         "button5":[],
         "button6":[],
         "text_area":[]
         }

def save():
   global imgWithInfo
   file = asksaveasfile(mode="w", defaultextension = ".png")
   img = ImageTk.getimage(imgWithInfo)
   if file:
       path = os.path.abspath(file.name)
       img.save(path)

def info():
    global cur_img
    global imgToInsert
    global imgWithInfo
    image = ImageTk.getimage(imgToInsert)
    imageDraw = ImageDraw.Draw(image)
    with open("./database/profiles.json","r") as prfls:
       profiles = json.load(prfls)
    
    image_id=cur_img[0].split("\\")[-1].split(".")[0]
    idx=50
    data=profiles[image_id]
    font = ImageFont.truetype('arial.ttf', 80) 
    for info in data:
       imageDraw.text((10,idx),f"{info.title()} : {data[info]}")
       idx+=20
    imgWithInfo=ImageTk.PhotoImage(image)
    text_area.delete("1.0",END)
    text_area.image_create(END,image=imgWithInfo)
    text_area.update_idletasks()
    
def clear():
    global caption
    caption=""
    text_area.delete("1.0",END)
    text_area.update_idletasks()

def save_state():
    monitor_state["button1"].append({i:button1[i] for i in button1.keys()})
    monitor_state["button2"].append({i:button2[i] for i in button2.keys()})
    monitor_state["button3"].append({i:button3[i] for i in button3.keys()})
    monitor_state["button4"].append({i:button4[i] for i in button4.keys()})
    monitor_state["button5"].append({i:button5[i] for i in button5.keys()})
    monitor_state["button6"].append({i:button6[i] for i in button6.keys()})
    monitor_state["text_area"].append([text_area.get("1.0","end-1c"),imgToInsert])

def update_state():
    global state
    button1.update_idletasks()
    button2.update_idletasks()
    if state==0:
        button3.place(x=650)
        button6.place(x=521)
    button4.place(x=680)
    button5.place(relx=0.45)
    button3.update_idletasks()
    button4.update_idletasks()
    button5.update_idletasks()
    button6.update_idletasks()
    text_area.update_idletasks()
    
def back():
    global state
    global caption
    global imgToInsert
    state-=1
    for i in button1.keys():
        button1[i]=monitor_state["button1"][-1][i]
        button2[i]=monitor_state["button2"][-1][i]
        button3[i]=monitor_state["button3"][-1][i]
        button4[i]=monitor_state["button4"][-1][i]
        button5[i]=monitor_state["button5"][-1][i]
        button6[i]=monitor_state["button6"][-1][i]
    text = monitor_state["text_area"][-1][0].strip()
    img = monitor_state["text_area"][-1][1]
    imgToInsert = img
    if state==0:
        caption = text if text else caption
        text_area.delete("1.0",END)
        text_area.insert(INSERT,caption+" ")
    elif img:
        text_area.delete("1.0",END)
        text_area.image_create(END,image=imgToInsert)
    monitor_state["button1"].pop(-1)
    monitor_state["button2"].pop(-1)
    monitor_state["button3"].pop(-1)
    monitor_state["button4"].pop(-1)
    monitor_state["button5"].pop(-1)
    monitor_state["button6"].pop(-1)
    monitor_state["text_area"].pop(-1)
    update_state()

def forward():
    global nfi
    global nbi
    global back_imgs
    global frwd_imgs
    global cur_img
    global imgToInsert
    
    if nfi>=1:
      img=frwd_imgs.pop(0)
      back_imgs.append(cur_img)
      cur_img=img
      nfi-=1
      nbi+=1
      imgFile=Image.open(cur_img[0])
      imageDraw = ImageDraw.Draw(imgFile)
      similarity = f"{((1-cur_img[1].detach()[0])*100).round()}% matched"
      imageDraw.text((5, 10),similarity , (255, 255, 255))
      image = imgFile.resize((450, 360), Image.ANTIALIAS)
      imgToInsert=ImageTk.PhotoImage(image)
      text_area.delete("1.0",END)
      text_area.image_create(END,image=imgToInsert)
      text_area.update_idletasks()
      if button1["state"]==DISABLED:
         button1.config(image=button_image1,text="",state=NORMAL,command=backward)
    if nfi==0:
       button1.config(image=button_image1,text="",state=NORMAL,command=backward)
       button2.config(image=button_image2,text="",state=DISABLED,command=forward)        
       button1.update_idletasks()
       button2.update_idletasks()
       
def backward():
    global nfi
    global nbi
    global back_imgs
    global frwd_imgs
    global cur_img
    global imgToInsert
    
    if nbi>=1:
       img=back_imgs.pop(-1)
       frwd_imgs=[cur_img]+frwd_imgs
       cur_img=img
       nfi+=1
       nbi-=1
       imgFile=Image.open(cur_img[0])
       imageDraw = ImageDraw.Draw(imgFile)
       similarity = f"{((1-cur_img[1].detach()[0])*100).round()}% matched"
       imageDraw.text((5, 10),similarity , (255, 255, 255))
       image = imgFile.resize((450, 360), Image.ANTIALIAS)
       imgToInsert=ImageTk.PhotoImage(image)
       text_area.delete("1.0",END)
       text_area.image_create(END,image=imgToInsert)
       text_area.update_idletasks()
       if button2["state"]==DISABLED:
         button2.config(image=button_image2,text="",state=NORMAL,command=forward)
    if nbi==0:
       button1.config(image=button_image1,text="",state=DISABLED,command=backward)
       button2.config(image=button_image2,text="",state=NORMAL,command=forward)        
       button1.update_idletasks()
       button2.update_idletasks()

def message(msg="Welcome! to Criminal Identity Detector."):
    engine.say(msg)
    engine.runAndWait()

def toggle_state(reset=False):
    global nfi
    global nbi
    if reset:
       button1.config(image='',text="Draw",command=draw)
       button2.config(image='',text="Search",state=DISABLED,command=search)
    else:
       button1.config(image=button_image1,text="",state=DISABLED,command=backward)
       button2.config(image=button_image2,text="",command=forward)
    if nfi==0 and nbi==0 or (nfi+nbi==0):
        button1.config(state=DISABLED)
        button2.config(state=DISABLED)
    button1.update_idletasks()
    button2.update_idletasks()
       
def search():
    global back_imgs
    global frwd_imgs
    global cur_img
    global nfi
    global nbi
    global imgToInsert
    global state
    back_imgs=[]
    frwd_imgs=[]
    nfi=0
    nbi=0
    save_state()
    state+=1
    trans = transforms.Compose([
                    transforms.Resize(100),
                    transforms.CenterCrop(100),
                    transforms.ToTensor()
                    ])
    fake_img = Image.open("./images/tmp.png")
    fake_img = trans(ImageOps.grayscale(fake_img)).reshape(1,1,100,100)
    images=glob.glob(".\\database\\images\\*jpg")
    path=".\\database\\images\\"
    model = SiameseNetwork()
    model.load_state_dict(torch.load('./models/siamese/netSm_epoch_last.pth', map_location=torch.device('cpu')))
    model = model.to(device)
    with open ("./database/hit_list.pkl","rb") as hitl:
       hit_list = pkl.load(hitl)
    with open ("./database/ordinary_list.pkl","rb") as ordl:
       ord_list = pkl.load(ordl)
    catches=[]
    location="Jhansi"
    opt_list = hit_list+ord_list[location] if location else []
    idx=0
    for i in opt_list:
       opt_list[idx]=path+str(i)+".jpg"
       images.remove(opt_list[idx])
       idx+=1
    c=0
    for data in [opt_list,images]:
        for image in data:
            img = Image.open(image)
            img = trans(ImageOps.grayscale(img)).reshape(1,1,100,100)
            out = oneshot(model,Variable(fake_img),Variable(img))
            if out[0]:
                catches.append((image,out[1]))
                c+=1
            if c>=5:
                    break
    catches.sort(key=lambda x:x[1])
    ncth=len(catches)
    if ncth:
        cur_img=catches.pop(0)
        frwd_imgs=catches
        imgFile=Image.open(cur_img[0])
        imageDraw = ImageDraw.Draw(imgFile)
        similarity = f"{((1-cur_img[1].detach()[0])*100).round()}% matched"
        imageDraw.text((5, 10),similarity , (255, 255, 255))
        button4.place(x=522,y=2)
        button4.update_idletasks()
        button5.config(image=button_image7,command=info,state=NORMAL)
    else:
        imgFile=Image.open('./images/imagenotfound.png')
        button5.place(relx=550)
        
    image = imgFile.resize((450, 360), Image.ANTIALIAS)
    imgToInsert=ImageTk.PhotoImage(image)
    text_area.delete("1.0",END)
    text_area.image_create(END,image=imgToInsert)
    text_area.update_idletasks()
    nfi=len(frwd_imgs)
    nbi=len(back_imgs)
    toggle_state()
    button5.update_idletasks()
    
def draw():
    global imgToInsert
    global state
    global caption
    text = caption if caption else text_area.get("1.0","end-1c")
    if len(text.strip())==0:
        message("Please, give the criminal's face description")
        return
    if not caption or state==0:
        save_state()
        state+=1
    caption = text
    text_area.delete("1.0",END)
    fake = text2face(text)
    grid = make_grid(fake.detach().cpu(), nrow=8, normalize=True).mul(255).clamp(0, 255).byte().permute(1,2,0).numpy()
    im = Image.fromarray(grid)
    im = im.resize((128,128), Image.NEAREST)
    im.save('./images/tmp.png')
    imgFile=Image.open('./images/tmp.png')
    image = imgFile.resize((450, 360), Image.ANTIALIAS)
    imgToInsert=ImageTk.PhotoImage(image)
    text_area.image_create(END,image=imgToInsert)
    text_area.update_idletasks()
    button2["state"] = NORMAL
    button1["text"] = "Redraw"
    button5.config(state=DISABLED)
    button6.place(x=660)
    button6.update_idletasks()
    button3.place(x=3,y=3)
    button3.update_idletasks()

def speech():
    # create the recognizer 
    r = sr.Recognizer() 
    
    # define the microphone 
    mic = sr.Microphone()

    # handling text insertion 
    def callback(audio_input):
        def inner_callback(text):
            text_area.insert(INSERT,text+" ")
            text_area.update_idletasks()
        try:
          text=r.recognize_google(audio_input, language='en-IN')
          thread = Thread(target=inner_callback,args=(text,))
          thread.start()
        except sr.UnknownValueError:
          engine.say("Your voice is not clear!")
        
    # record your speech
    with mic as source:
        r.adjust_for_ambient_noise(source) 
        captured_audio = r.listen(source=mic)
        thread = Thread(target=callback,args=(captured_audio,))
        thread.start()

def micro():
   global toggle
   global threads

   if toggle:
      for thread in threads[::-1]:
        thread.join()
      threads=[]
      button_image5 = PhotoImage(file='./images/microphone.png')
      button5.config(image = button_image5)
      button5.image = button_image5
      button5.update_idletasks()
      toggle = False if toggle else True
      return
   else:
      button_image5 = PhotoImage(file='./images/mute-microphone.png')
      button5.config(image = button_image5)
      button5.image = button_image5
      button5.update_idletasks()
      toggle = False if toggle else True
      thread = Thread(target=listen)
      threads.append(thread)
      thread.start()
   
   
def listen():
    global threads
    
    def handler():
        start_time = default_timer()
        while True:
           thread = Thread(target=speech)
           threads.append(thread)
           thread.start()
           time.sleep(3)
           last_time = default_timer()
           if last_time-start_time>30:
                break
    thread = Thread(target=handler)
    threads.append(thread)
    thread.start()
        
HEIGHT = 300
WIDTH  = 550

root = Tk()
root.minsize(300, 550)
root.title('Criminal Identity Detector')

canvas = Canvas(root,height=HEIGHT,width=WIDTH)
canvas.pack()


background_image = PhotoImage(file='./images/detective.png')
background_label = Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

frame = Frame(root,bg='#80c1ff', bd=10)
frame.place(relx=0.5, rely=0.15, relwidth=0.82, relheight=0.65, anchor='n')

lower_frame = Frame(root, bg="#80c1ff",bd=5)
lower_frame.place(relx=0.5, rely=0.85, relwidth=0.82, relheight=0.1,anchor='n')

button1 = Button(lower_frame, text="Draw", font=40, command=draw)
button1.place(relx=0.01,relwidth=0.3, relheight=1)

button2 = Button(lower_frame, text="Search",state=DISABLED, font=40, command=search)
button2.place(relx=0.7,relwidth=0.3, relheight=1)

button_image1 = PhotoImage(file='./images/backward.png')
button_image2 = PhotoImage(file='./images/forward.png')
button_image3 = PhotoImage(file='./images/back.png')
button_image4 = PhotoImage(file='./images/save.png')
button_image6 = PhotoImage(file='./images/refresh.png')
button_image7 = PhotoImage(file='./images/info.png')

button3 = Button(root,image=button_image3, command=back)

button4 = Button(root, bg='black',image=button_image4, command=save)

label = Label(frame)
label.place(relwidth=1,relheight=1)

text_area = Text(label,height=25)
text_area.pack()

button_image5 = PhotoImage(file='./images/microphone.png')

button5 = Button(lower_frame,image=button_image5,command=micro)
button5.place(relx=0.45,relwidth=0.1, relheight=1)

button6 = Button(root,image=button_image6,command=clear)
button6.place(x=521,y=3)


root.after(3000, message)
root.iconbitmap('./cid.ico')
root.mainloop()
