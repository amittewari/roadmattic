
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


subjects = ["", "Good", "Potholes","Cracks"]


# In[3]:


#function to detect face using OpenCV
def detect_surface(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('C:\\Users\\amit\\AppData\\Local\\Continuum\\anaconda3\\pkgs\\opencv-3.3.1-py36h20b85fd_1\\Library\\etc\\lbpcascades\\lbpcascade_frontalface.xml')
 
    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    surfaces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1);
 
    #if no faces are detected then return original img
    if (len(surfaces) == 0):
        return None, None
 
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = surfaces[0]
 
    #return only the surface part of the image
    return gray[y:y+w, x:x+h], surfaces[0]


# In[4]:


#this function will read all training images, detect surface from each image
#and will return two lists of exactly same size, one list 
# of surfaces and another list of labels for each surface
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all surfaces
    surfaces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            
            #detect face
            surface, rect = detect_surface(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if surface is not None:
                #add face to list of faces
                surfaces.append(surface)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return surfaces, labels


# In[ ]:


#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces
#and the other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("C://amit//per//surface//training-data")
print("Data prepared")
 
#print total faces and labels
print("Total surfaces: ", len(faces))
print("Total labels: ", len(labels))


# In[7]:


surface_recognizer = cv2.face.LBPHFaceRecognizer_create()


# In[8]:


#train our surface recognizer of our training faces
surface_recognizer.train(surfaces, np.array(labels))


# In[9]:


#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# In[10]:


#this function recognizes the surface in image passed
#and draws a rectangle around detected surface with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detect face from the image
    surface, rect = detect_surface(img)

    #predict the image using our surface recognizer 
    label= surface_recognizer.predict(face)
    #get name of respective label returned by surface recognizer
    label_text = subjects[label[0]]
    
    #draw a rectangle around surface detected
    draw_rectangle(img, rect)
    conf=100-float(label[1])
    print(conf)
    if(conf>40):
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    else:    
        draw_text(img, 'unknown', rect[0], rect[1]-5)
    return img

