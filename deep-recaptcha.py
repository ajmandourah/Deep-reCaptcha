from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import os
import requests
from PIL import Image
from io import BytesIO
import random
import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Threshold for accepting a choice other than the predicted, adjust if many false positive results. default is 0.6
threshold = 0.5

#suppress tensorflow logs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
# check if data dir is present 
if os.path.exists(os.path.join(os.getcwd(),"data")):
    data_dir = os.path.join(os.getcwd(),"data")
else:
    os.mkdir(os.path.join(os.getcwd(),"data"))
    data_dir = os.path.join(os.getcwd(),"data")

#A list of possible classes

CLASSES  = [
    "bicycle",
    "bridge",
    "bus",
    "car",
    "chimney",
    "crosswalk",
    "hydrant",
    "motorcycle",
    "other",
    "palm",
    "stairs",
    "traffic"
]    
    


# main functions

def getModel():
    f = tf.keras.utils.get_file(
                                fname="model.h5",
                                origin="https://github.com/ajmandourah/Deep-reCaptcha/releases/download/model/efficentM2.h5",
                                cache_dir=data_dir,
                                cache_subdir='model')
    model = keras.models.load_model(f, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=["accuracy"])
    return model

def sliceImage(path):
    from PIL import Image
    img = np.array(Image.open(path))

    ys = img.shape[0] // 3
    xs = img.shape[1] // 3

    topLeft = img[0:ys,0:xs]
    topMid = img[0:ys,xs:xs*2]
    topRight = img[0:ys,xs*2:xs*3]
    midLeft = img[ys:ys*2,0:xs]
    midmid = img[ys:ys*2,xs:xs*2]
    midRight = img[ys:ys*2,xs*2:xs*3]
    bottomLeft = img[ys*2:ys*3,0:xs]
    bottomMid = img[ys*2:ys*3,xs:xs*2]
    bottomRight = img[ys*2:ys*3,xs*2:xs*3]
    
    return [topLeft, topMid, topRight, midLeft, midmid, midRight, bottomLeft, bottomMid, bottomRight]



def predictTile(tile, model):    
    #resize the image

    i = img_to_array(tile)
    to_predict = i.reshape((-1,224,224,3))
    prediction = model.predict(to_predict)
    #return a list of the prediction array, the class name with highest probability and its index 
    return [ prediction, CLASSES[np.argmax(prediction)], np.argmax(prediction)  ]


#getting the model from github
model = getModel()

#Selenium workflow
captcha_url = "https://www.google.com/recaptcha/api2/demo"
driver = webdriver.Firefox()
driver.get(captcha_url)

recapcha_frame = driver.find_element(By.XPATH, "//iframe[@title='reCAPTCHA']")

driver.switch_to.frame(recapcha_frame)

driver.find_element(By.CLASS_NAME, "recaptcha-checkbox-border").click()
sleep(4)
driver.switch_to.default_content()
WebDriverWait(driver, 20).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH,"//iframe[@title='recaptcha challenge expires in two minutes']")))


#The main loop

while True:
    reload = driver.find_element(By.ID, "recaptcha-reload-button")
    title_wrapper = driver.find_element(By.ID, 'rc-imageselect')
    
    #Skipping if it is a 4x4 captcha...will be implementing it later on .
    if "squares" in title_wrapper.text:
        print("Square captcha found....skipping")
        reload = driver.find_element(By.ID, "recaptcha-reload-button")
        reload.click()
        continue
    
    if "none" in title_wrapper.text:
        print("found a 3x3 dynamic captcha")
        dynamic_captcha = True
    else:
        print("found a 3x3 one time selection captcha")
        dynamic_captcha = False

    #Get the object of the captcha where we suppose to look for
    captcha_object = title_wrapper.find_element(By.TAG_NAME, 'strong')
    print("The object to look for is ", captcha_object.text)
    
    #get the class index of the captcha object if found.
    for i in CLASSES:
        if i in captcha_object.text:
            class_index = CLASSES.index(i)
            print("class index is ", str(class_index))    
    
     
    #first run of solving the captcha       
    check = []
    for i in range(9):
        xpath = "//td[contains(@tabindex, '" + str(i+4)+ "')]"
        matched_tile = driver.find_element(By.XPATH, xpath)
        matched_tile.screenshot(os.path.join(data_dir,"tile.jpg"))
        img = Image.open(os.path.join(data_dir,"tile.jpg")).convert('RGB')
        img = img.resize(size=(224,224))
                    
        result = predictTile(img, model)
        current_object_probability = result[0][0][class_index]
        compare_probability = result[2] * threshold
        print("The AI predicted tile to be ", result[1], "and probability is",current_object_probability)
        
        '''
        Two methods for predictioin here, The simple matching of the text was first implemented but false negative/positive results
        was seen.
        To compromise getting the probability of the current captcha object and assigning a thresold seems to yeild a better results
        '''
        if result[1] in captcha_object.text:
            print("found a match clicking tile ", str(i+1))
            #tabindex="4"
            matched_tile.click()
            check.append("found")
            sleep(3)
        elif current_object_probability > compare_probability :
            print("found a match clicking tile ", str(i+1))
            #tabindex="4"
            matched_tile.click()
            check.append("found")
            sleep(3)
            
            
        else:
            print(" not a match .. skipping!")
            sleep(.1)
            continue
        
    if dynamic_captcha:
        if len(check) <1:    
            verify = driver.find_element(By.ID, "recaptcha-verify-button").click()
            sleep(5)
            break
        else:
            while True:
                #Loop untill no results found 
                
                check = []
                for i in range(9):
                    xpath = "//td[contains(@tabindex, '" + str(i+4)+ "')]"
                    matched_tile = driver.find_element(By.XPATH, xpath)
                    matched_tile.screenshot(os.path.join(data_dir,"tile.jpg"))
                    img = Image.open(os.path.join(data_dir,"tile.jpg")).convert('RGB')
                    img = img.resize(size=(224,224)) 
                                            
                    result = predictTile(img, model)
                    current_object_probability = result[0][0][class_index]
                    compare_probability = result[2] * threshold

                    print("The AI predicted tile to be", result[1], "and probability is",current_object_probability)
                    
                    if result[1] in captcha_object.text:
                        print("found a match clicking tile ", str(i+1))
                        #tabindex="4"
                        matched_tile.click()
                        check.append("found")
                        sleep(3)
                    
                    if current_object_probability > compare_probability :
                        print("found a match clicking tile ", str(i+1))
                        #tabindex="4"
                        matched_tile.click()
                        check.append("found")
                        sleep(3)
                    else:
                        print(" not a match .. skipping!")
                        sleep(.1)
                        continue

                if len(check) <1:    
                    verify = driver.find_element(By.ID, "recaptcha-verify-button").click()
                    sleep(5)
                    break   
        
        
    #else if this were a non dynamic captcha      
    else:   
        verify = driver.find_element(By.ID, "recaptcha-verify-button").click()
        break
    break
driver.close()