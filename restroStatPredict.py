#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 04:13:13 2017

@author: sanketagarwal
"""

"""
restroStatPredict.py
Description: This module is creating a user interface framework for the
             restaurant status prediction application.
"""
#Importing the library and packages used
from tkinter import Entry
from tkinter import Label
from tkinter import Button
from tkinter import Toplevel
from tkinter import StringVar
from tkinter import OptionMenu
from tkinter import Tk
from tkinter import*
import pickle
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#class restroStatPredict 
class restroStatPredict: 
    #constructor of the class guiProj
    def __init__(self,master):
        self.master=master
        master.title("RIT Dining Advisor System")
          
        self.labelmain = Label(master, text="Enter The Details ")
        self.labelmain.grid(columnspan=5, sticky=W)
        
        self.labelrestro = Label(master, text="Restraunt Name")
        self.labelrestro.grid(row= 3, column=2)
       
        self.labelsession = Label(master, text="Session")
        self.labelsession.grid(row=3, column=3)
       
        self.labeldow = Label(master, text="Day of Week")
        self.labeldow.grid(row=3, column=4)
        
        self.labeltimeInt = Label(master, text="Time Interval")
        self.labeltimeInt.grid(row=3, column=5)
       
        
        #Drop down menu for restaurants field        
        self.varrestro = StringVar(root)
        self.varrestro.set('Select' )
        self.restrochoices = {'Select':'', 'Beanz':10053, 'Fieldhouse Concessions':10056, 'Ben and Jerrys':10060, 'Nathans Soup':10071,'Midnight Oil':10081,'Global Village Market':10066,'CROSSROADS CAFE & MARKET':10028,'DINING COMMONS':10022,'BRICK CITY CAFE':10010,'GV CANTINA & GRILLE':10065,'RITZ SPORTS ZONE':10012,'SOLS UNDERGROUND':10020,'CTRL ALT DELI':10016,'VENDING':10034,'ARTESANO':10063,'BYTES ON THE RUN':10062,'THE COLLEGE GRIND':10024,'CORNER STORE':10018,'GRACIES(GRACE WATSON)':10026,'(JAVA)UGRYD ON CAMPUS':10102,'FRESHENS CATALYST':10082}
        self.restrooption = OptionMenu(root, self.varrestro, *self.restrochoices.keys())
        #self.varrestro = StringVar(root)
        
        #Drop down menu for semester field
        self.varsession = StringVar(root)
        self.varsession.set('Select' )
        self.sessionchoices = {'Select':'', 'spring':3,'Fall':2,'new_stud_orient':1,'finals(fall)':4,'finals(spr)':5,'finals(sum)':6,'finals(intersession)':7,'summer':8,'intersession':9,'college_closed':0}
        self.sessionoption = OptionMenu(root, self.varsession, *self.sessionchoices.keys())
        
        #Drop Down menu for week-day field
        self.vardow = StringVar(root)
        self.vardow.set('Select' )
        self.dowchoices = {'Select':'', 'Friday':6,'Sunday':1,'Monday':2,'Tuesday':3,'Wednesday':4,'Thursday':5,'Saturday':7}
        self.dowoption = OptionMenu(root, self.vardow, *self.dowchoices.keys())
        
        #Drop Down menu for time-interval field
        self.varTI = StringVar(root)
        self.varTI.set('Select' )
        self.TIchoices = {'Select':'', '[5am-6am)':1,'[6am-7am)':2,'[7am-8am)':3,'[8am-9am)':4,'[9am-10am)':5,'[10am-11am)':6,'[11am-12pm)':7,'[12pm-1pm)':8,'[1pm-2pm)':9,'[2pm-3pm)':10,'[3pm-4pm)':11,'[4pm-5pm)':12,'[5pm-6pm)':13,'[6pm-7pm)':14,'[7pm-8pm)':15,'[8pm-9pm)':16,'[9pm-10pm)':17,'[10pm-11pm)':18,'[11pm-12am)':19,'[12am-1am)':20,'[1am-2am)':21,'[2am-3am)':22,'[3am-4am)':23,'[4am-5am)':24}
        self.TIoption = OptionMenu(root, self.varTI, *self.TIchoices.keys())
    
        
        #Assignment of values provided by user for restaurant
        self.u1restro = self.restrooption
        self.u1restro.grid(row=5, column =2)
        
        
        #Assignment of values provided by user for session
        self.u1session = self.sessionoption
        self.u1session.grid(row=5, column =3)       
        
        #Assignment of values provided by user for day of week
        self.u1dow = self.dowoption
        self.u1dow.grid(row=5, column =4)    
        
        
        #Assignment of values provided by user for time interval
        self.u1TI = self.TIoption
        self.u1TI.grid(row=5, column =5) 
        
       
        
        #submit button : it will call recommend method upon a mouse click 
        self.submit_button = Button(master, text="Recommend ", command=self.recommend)
        self.submit_button.grid(row=6, column=3)
       
    
    #This function is called upon the mouse click by the user after the deatils were entered   
    def recommend(self):
        #A new window will be openend upon a mouse click to the button
        self.top = Toplevel()
        
        # This new window will fetch the information to recommend if its good to go to the restaurant
        self.top.title("Status")
        
        #fetching the values of restaunts provided by the user
        u1restro=self.restrochoices[self.varrestro.get()]  
        
        #fetching the values of session provided by the user
        u1session=self.sessionchoices[self.varsession.get()]
        
        
        #fetching the values of dow provided by the user
        u1dow=self.dowchoices[self.vardow.get()]
        
         #fetching the values of TI provided by the user
        u1TI=self.TIchoices[self.varTI.get()]
       
        #the values entered are saved as a dataframe which will act as the testng dataframe for the prediction
        table=[u1restro,u1session,u1dow,u1TI]
        
        testTable=[[u1restro,u1session,u1dow,u1TI,'']]

        
        #cols will store names of attributes in the dataset
        cols=['locgrpid','semester_ident','day_of_week_ident','time_inter_desc_ident','target']
        df=pd.DataFrame(testTable,columns=cols)
        
        #FullDS will store the dataset provided on the given path
        fullDS=pd.read_csv('../restaurantStatusPrediction/dining_index_tab_targ_1.csv')
        train=fullDS
        
        #tesing file will be the dataframe of details given by the user
        testDS = df
        
        
        # Feature will have the list of features for modelling
        #Fragmenting the data into two parts: training set and validation set
        msk = np.random.rand(len(fullDS)) < 0.75
        Train = fullDS[msk]
        validate = fullDS[~msk]

       #Genrating the modle based on the feature list and target variable
        features=['locgrpid','semester_ident','day_of_week_ident','time_inter_desc_ident']
        x_train = Train[list(features)].values
        y_train = Train["target"].values
        x_validate = validate[list(features)].values
        y_validate = validate["target"].values
        x_test= testDS[list(features)].values
        #this will generate a random forest model on the provided data
        
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(x_train, y_train)
        print("check")
        status_new = rf.predict(x_validate)
        print(sk.metrics.classification_report(y_validate,status_new))
            
        #final_status_new = rf.predict(x_test)
        #testDS["targ"]=final_status_new            
        #testDS.to_csv('/Users/sanketagarwal/Downloads/targetData.csv')
            
        #x_test=testDS[list(features)].values
  
        
        #predicting the value of target variable for the testing dataset.
        final_status = rf.predict(x_test)
        print(final_status)
     
        #output will have the prediction for the restaurant status
        output=final_status
    
        #outputstring=output.to_string(index=False)
        if(output==[1]):
            outputString= "NO LINE! TAKE YOUR TIME!"
            outputcolr="green"
        elif(output==[2]):
            outputString= "BRACE YOURSELF FOR A LINE"
            outputcolr = "yellow"
        else:
            outputString= "CONSIDER ANOTHER OPTION!"
            outputcolr = "red"
            

        self.labelmain = Label(self.top, text=outputString, bg="black",fg=outputcolr,width=25,height = 10)
        self.labelmain.pack()
      
      
#Tkinter root creation
root = Tk()
my_prediction = restroStatPredict(root)
root.mainloop()
