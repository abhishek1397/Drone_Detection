import numpy as np # importing numpy
import cv2         # importing cv2 for using yolov3
import glob        # importing glob that is used to retrieve files/pathnames matching a specified pattern
import random
import tkinter as tk   #importing tkinter for GUI
from PIL import Image,ImageTk  # importing background image for tkinter window
import matplotlib.pyplot as plt


# initializing tkinter ooject
root = tk.Tk()


root.geometry("700x380")
root.title("Drone Detector")

# for loading background image on tkinter window
image1 = ImageTk.PhotoImage(Image.open(r'C:\Users\Abhishek Sharma\Downloads\Image18_Quadcopter-drone_Caban_022819-Hero-1000x715.jpg'))
root.geometry("%dx%d+0+0" % (700,380))
panel1 = tk.Label(root,image=image1)
panel1.place(x=0, y=0, relwidth=1, relheight=1)
panel1.image = image1
    
# label for Gui welcome window
label1 = tk.Label(root,text=" Welcome to Drone Detector ", bg="orange red", fg="white" , font=("Algerian",25,"bold"))
label1.grid(row=0,column=3, pady=5,padx=5)



def detect_drone(accuracys=0,no_images=0):    
#load yolo
    net=cv2.dnn.readNet("D:\\Machine Learning Model\\weights\\yolo-drone.weights","D:\\Machine Learning Model\\cfg\\yolo-drone.cfg")
    
#name of object 
    classes_drone=["drone","no_drone"] 
    
#images path
    imag_path = glob.glob("D:\\Machine Learning Model\\images\\*.jpg")
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes_drone), 3))
    
#for shuffling images 
    random.shuffle(imag_path)
    
#variable to keep track of epoches
    var=0
    no_image=no_images
    confidence_plot=[]
    var_graph=[]
    if(no_images ==0):
        no_images=2500
    
#loop through all the images
    for drone_image_path in imag_path:
        var = var+1
#Loading image
        drone_image = cv2.imread(drone_image_path)
        drone_image = cv2.resize(drone_image, None, fx=0.7, fy=0.7)
        height, width, channels = drone_image.shape
        print(var)
        var_graph.append(var)
        
        
        if(var<=no_images):      
                
#Detecting objects
            blob = cv2.dnn.blobFromImage(drone_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
            net.setInput(blob)
            outs = net.forward(output_layers)

#Showing informations on the screen
            class_ids = []
            confidences = []
            img_boxes = []
            for out in outs:
                for detect_drone in out:
                    class_scores = detect_drone[5:]
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id]
                    if(confidence > 0.9):
                        confidence_plot.append(confidence)
                    if confidence > 0.3:
                        # Object detected
                        #print(class_id)
                        #print("inner_lppo:%d",var)
                        center_x = int(detect_drone[0] * width)
                        center_y = int(detect_drone[1] * height)
                        w = int(detect_drone[2] * width)
                        h = int(detect_drone[3] * height)
        
#Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
        
                        img_boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
            indexes = cv2.dnn.NMSBoxes(img_boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(img_boxes)):
                if i in indexes:
                    x, y, w, h = img_boxes[i]
                    label = str(classes_drone[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(drone_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(drone_image, label, (x, y + 30), font, 3, color, 2)
             
            if(no_image!=0):
                cv2.imshow("Image", drone_image) 
                cv2.waitKey(0)
                
                    
            
        else:
            break            
        
    if(accuracys==1):
           #print(len(confidence_plot_not))
           data = {'Drone': len(confidence_plot), 'Not_Drone': no_images-len(confidence_plot)}
           names = list(data.keys())
           values = list(data.values())
           
           fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
           axs[0].bar(names, values)
           axs[1].scatter(names, values)
        
    
    cv2.destroyAllWindows()

# to destroy tkinter window  
def quit():
    root.destroy()
    
# function to enter no of images to view        
def fxn1(image_perm=0):
    label3 = tk.Label(root,text="How many images you wish to view:", fg="black" , font=("Times New Roman",12,""))
    label3.grid(row=5,column=3, pady=2,padx=2)

    no_image=tk.Entry(root)
    no_image.grid(row=6,column=3, pady=2,padx=2)
    button3=tk.Button(root,text="  Enter ", bg = "pink", font=("Times New Roman",12,""),command=lambda: fxn2(no_images=int(no_image.get())))
    button3.grid(row=7,column=3,pady=5,padx=5)
    
# function to ask to view graph    
#label and button for asking to view accuracy graph
def fxn2(no_images=0):
    label2 = tk.Label(root,text="Do you want to view accuracy graph : ", fg="black" , font=("Times New Roman",12,""))
    label2.grid(row=8,column=3, pady=2,padx=2)
    button3=tk.Button(root,text="  Yes  ", bg = "pink", font=("Times New Roman",12,""),command=lambda:fxn3(accuracy=1,no_image=no_images))
    button3.grid(row=9,column=2,pady=5,padx=5)
    button3=tk.Button(root,text="  No  ", bg = "pink", font=("Times New Roman",12,""),command=lambda:fxn3(accuracy=0,no_image=no_images))
    button3.grid(row=9,column=4,pady=5,padx=5)
    
# button to detect drone and launch detecting algo    
def fxn3(accuracy=0,image_perms=0,no_image=0):
    button1=tk.Button(root,text="  Detect Drone ", bg = "pink",font=("Times New Roman",12,""), command=lambda:detect_drone(accuracys=accuracy,no_images=no_image))
    button1.grid(row=10,column=3,pady=5,padx=5)
    

#label and button for asking to view detected drone images
choice1=tk.IntVar()
label2 = tk.Label(root,text="Do you want to view drone detected images:", fg="black" , font=("Times New Roman",12,""))
label2.grid(row=1,column=3, pady=2,padx=2)
button4=tk.Button(root,text="  Yes  ", bg = "pink", font=("Times New Roman",12,""),command=lambda:fxn1(image_perm=1))
button4.grid(row=2,column=2,pady=5,padx=5)
button5=tk.Button(root,text="  no  ", bg = "pink", font=("Times New Roman",12,""),command=lambda:fxn2(no_images=0))
button5.grid(row=2,column=4,pady=5,padx=5)

#button for quitting gui and stop algo
button3=tk.Button(root,text="  Exit  ", bg = "pink", font=("Times New Roman",12,""), command=quit)
button3.grid(row=15,column=3,pady=5,padx=5)

root.mainloop() 