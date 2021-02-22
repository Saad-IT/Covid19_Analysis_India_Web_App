
import requests
import json
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from PIL import Image
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import plotly.graph_objects as go

import tensorflow as tf
from keras.preprocessing import image as ge
import argparse
import os
import time



response = requests.get("https://api.apify.com/v2/key-value-stores/toDWvRj1JpTXiM8FF/records/LATEST?disableRedirect=true")
data=response.json()
#b=json.dumps(data)
#print(b)
#print(data)
#a=data["regionData"][0::]
#print(a)
df=pd.DataFrame(data['regionData'],columns=['region','totalInfected','recovered','deceased'])
#print(df)

#@st.cache(persist=True)
def load_data():
    pd_data = df
    lowercase = lambda x: str(x).lower()
    pd_data.rename(lowercase, axis='columns', inplace=True)
    pd_data.rename(columns={'region':'State','totalinfected':'Active Cases','recovered':'Recovered','deceased':'Deaths'},inplace=True)
    return pd_data

pd_data =load_data()
#pd_data.to_csv("C:\\Users\\micro\\Desktop\\Data_Project\\pd_data_file.csv")
original_data = pd_data[:]
count=0
for label,rows in pd_data.iterrows():
    pd_data.loc[label,'Confirmed']= (int(pd_data['Active Cases'].tolist()[count]) + int(pd_data['Recovered'].tolist()[count]) + int(pd_data['Deaths'].tolist()[count]))
    count+=1



#pd_data index labels
pd_data.index=['AN','AP','AR','AS','BR','CH','CT','DN','DL','GA','GJ','HR','HP','JK','JH','KA','KL','LD','LDP','MP','MH','MN','ML','MZ','NL','OR','PY','PB','RJ','SK','TN','TG','TR','UT','UP','WB']

#sidebar details
st.sidebar.title("Covid-19 Analysis India")



#menu
menu=st.sidebar.radio("",["Home",'Symptoms of Covid-19','Prevention from Covid-19','Help Line & Support','About Us','Covid-19 Detector'])
if menu=='Home':
    st.title("Covid-19 Analysis India 🔍📊")
    image=Image.open("WebApp_Top_Banner.jpg")
    st.image(image,use_column_width=True)
    
    menu2=st.sidebar.radio("Select option below: ",['Show All India Data','Show Statewise Data','Tests Done'])

    
    if menu2=='Show All India Data':
    
        #TotalIndia
        def get_total_dataframe():
            total_dataframe=pd.DataFrame({

            "Status":["Confirmed","Recovered","Active Cases","Deaths"],
            'Number of Cases in India':(data['totalCases'],
            data['recovered'],
            data['activeCases'],
            data['deaths'])})
            return total_dataframe

        state_total=get_total_dataframe()

        state_total_graph=px.bar(
            state_total,
            x='Status',
            y='Number of Cases in India',
            color="Status"
        )
        st.plotly_chart(state_total_graph)


        st.write(pd_data)
        #a=int(state_data['Active Cases'][state_data.index])
        #b=int(state_data['Recovered'][state_data.index])
        #c=int(state_data['Deaths'][state_data.index])


    

        if st.checkbox("Show data in Tabular Form",False,key=1):
            st.table(pd_data)


        response2 = requests.get("https://api.covid19india.org/data.json")
        data2=response2.json()
        


        



        df=pd.DataFrame(data2['cases_time_series'])
        #df.to_csv("C:\\Users\\micro\\Desktop\\Data_Project\\df_file.csv")
        row,column=df.shape
        id=list(range(1,row+1))
        df['id']=id
        
        st.subheader("Covid-19 India Time Series Data from 'Jan 31 2020'")

        fig = px.line(df, x='dateymd', y=["totalconfirmed","totalrecovered",'totaldeceased'],labels={'dateymd':"Dates",'value':"Cases",'variable':"Status"})

        st.plotly_chart(fig)

        #Preparing Data
        x=np.array(df['id']).reshape(-1,1)
        y=np.array(df['totalconfirmed']).reshape(-1,1)

        polyFeat = PolynomialFeatures(degree=3)
        x=polyFeat.fit_transform(x)

        #training
        model= linear_model.LinearRegression()
        model.fit(x,y)
        accuracy=model.score(x,y)
        #print(f"Accuracy: {round(accuracy*100,3)}%")


        #prediction
        days=30
        cases=[]
        upcoming_days=list(range(1,days+1))
        for i in range(1,days+1):
            cases.append((int(model.predict(polyFeat.fit_transform([[row+i]])))))
        #print(cases)



        agree = st.button('**Click to see upcoming 30 days cases prediction**')
        if agree:
            st.bar_chart(cases)


   


    #sidebar
    if menu2=='Show Statewise Data':
        select=st.sidebar.selectbox('Select a State',pd_data['State'])

        state_data=pd.DataFrame(pd_data[pd_data['State']==select])
        
        st.title("%s Analysis"%(select))
             
        st.table(state_data)


        a=int(state_data['Active Cases'][state_data.index])
        b=int(state_data['Recovered'][state_data.index])
        c=int(state_data['Deaths'][state_data.index])
        d=a+b+c  #confirmed cases
        



        def get_state_dataframe():
            state_dataframe=pd.DataFrame({

            "Status":["Confirmed","Recovered","Active Cases","Deaths"],
            'Number of Cases in %s'%(select):(d,b,a,c)})
            return state_dataframe

        Ind_state_total=get_state_dataframe()
        #st.write(Ind_state_total)


        Ind_state_total_graph=px.bar(
        Ind_state_total,
        x='Status',
        y='Number of Cases in %s'%(select),
        color="Status"
    
        )
        st.plotly_chart(Ind_state_total_graph)

               
        
        #Pie_Chart
        comparison_values=[c,b,a]
        names=['Deaths','Recovered','Active Cases']
        fig5 = go.Figure(data=[go.Pie(labels=names, values=comparison_values)])
        st.plotly_chart(fig5)



        #Which States have More Cases

        Sorted_Data=pd_data.sort_values("Confirmed",axis=0,ascending=False,na_position='last')
        if st.checkbox("Show top most affected states data",False,key=1):
            a=st.slider("Select",min_value=1,max_value=35,value=10)
            fig6 = go.Figure(data=[go.Pie(labels=Sorted_Data['State'][0:a], values=Sorted_Data['Confirmed'][0:a])])
            st.plotly_chart(fig6)
            st.subheader("Top 10 States with highest number of Covid-19 Cases")
            st.table(Sorted_Data.iloc[:10,:])



        #compare states
        if st.checkbox("Compare States",False,key=1):
            S1,S2=st.beta_columns(2)
            State1=S1.selectbox('Select State 1',pd_data['State'])
            State2=S2.selectbox('Select State 2',pd_data['State'])

            state1_data=pd.DataFrame(pd_data[pd_data['State']==State1])
            state2_data=pd.DataFrame(pd_data[pd_data['State']==State2])

            a=int(state1_data['Active Cases'][state1_data.index])
            b=int(state1_data['Recovered'][state1_data.index])
            c=int(state1_data['Deaths'][state1_data.index])
            d=int(state1_data['Confirmed'][state1_data.index])


            e=int(state2_data['Active Cases'][state2_data.index])
            f=int(state2_data['Recovered'][state2_data.index])
            g=int(state2_data['Deaths'][state2_data.index])
            h=int(state2_data['Confirmed'][state2_data.index])
        

            fig7 = go.Figure(data=[
                go.Bar(name=State1, x=['Active Cases','Recovered','Deaths','Confirmed'], y=[a,b,c,d]),
                go.Bar(name=State2, x=['Active Cases','Recovered','Deaths','Confirmed'], y=[e,f,g,h])
            ])
            fig7.update_layout(barmode='group')
            st.plotly_chart(fig7)


        
    if menu2=='Tests Done':
        
        response2 = requests.get("https://api.covid19india.org/data.json")
        data2=response2.json()
        #tested samples
        
        tested_df=pd.DataFrame(data2['tested'][-1:-2:-1])
        sample_reported_today=int(tested_df['samplereportedtoday'])
        total_sample_tested=int(tested_df['totalsamplestested'])
        st.header("🧪 Covid Samples Tested Today :    "+str(sample_reported_today))
        st.header("Total Samples Done :    "+str(total_sample_tested))
        #st.subheader(tested_df['testedasof'][-1:-2:-1].tolist()[0])
        
        

        
        figtest = px.line(data2['tested'],x='testedasof',y='samplereportedtoday')

        st.plotly_chart(figtest)


if menu=='Symptoms of Covid-19':
    st.title("Symptoms of Covid-19")
    image=Image.open("symptoms_of_Covid19.png")
    st.image(image,use_column_width=True)
    st.subheader("**Most common symptoms:**")
    st.markdown("**◼️ fever**")
    st.markdown("**◼️ dry cough**")
    st.markdown("**◼️ tiredness**")

    st.subheader("**Less common symptoms:**")
    st.markdown("**◼️ aches and pains**")
    st.markdown("**◼️ sore throat**")
    st.markdown("**◼️ diarrhoea**")
    st.markdown("**◼️ conjunctivitis**")
    st.markdown("**◼️ headache**")
    st.markdown("**◼️ loss of taste or smell**")
    st.markdown("**◼️ a rash on skin, or discolouration of fingers or toes**")

    st.subheader("**Serious symptoms:**")
    st.markdown("**◼️ difficulty breathing or shortness of breath**")
    st.markdown("**◼️ chest pain or pressure**")
    st.markdown("**◼️ loss of speech or movement**")
    st.markdown("**Seek immediate medical attention if you have serious symptoms. Always call before visiting your doctor or health facility.**")
    st.markdown("**People with mild symptoms who are otherwise healthy should manage their symptoms at home.**")
    st.markdown("**On average it takes 5–6 days from when someone is infected with the virus for symptoms to show, however it can take up to 14 days.**")
    st.video('https://youtu.be/oBSkHZPu2xU') 


if menu=="Prevention from Covid-19":
    st.title("Prevention from Covid-19")

    image=Image.open("Prevention_from_Covid19.png")
    st.image(image,use_column_width=True)
    st.markdown("**Protect yourself and others around you by knowing the facts and taking appropriate precautions. Follow advice provided by your local health authority.**")
    st.subheader("**To prevent the spread of COVID-19:**")
    st.markdown("**◼️ Clean your hands often. Use soap and water, or an alcohol-based hand rub.**")
    st.markdown("**◼️ Maintain a safe distance from anyone who is coughing or sneezing**")
    st.markdown("**◼️ Wear a mask when physical distancing is not possible.**")
    st.markdown("**◼️ Don’t touch your eyes, nose or mouth**")
    st.markdown("**◼️ Cover your nose and mouth with your bent elbow or a tissue when you cough or sneeze.**")
    st.markdown("**◼️ Stay home if you feel unwell.**")
    st.markdown("**◼️ If you have a fever, cough and difficulty breathing, seek medical attention.**")
    st.markdown("**Calling in advance allows your healthcare provider to quickly direct you to the right health facility. This protects you, and prevents the spread of viruses and other infections**")
    st.video('https://youtu.be/1APwq1df6Mw')
if menu=='Help Line & Support':
    st.title("Help Line & Support")
    st.info("**Live HelpDesk :  https://wa.me/919013151515  \nCenter Helpline Number :  +91-11-23978046  \nToll Free : 1075  \n Helpline Email ID : ncov2019@gov.in**")
    st.subheader("For Statewise Help Line & Support: ")
    StateHelpDF=pd.read_csv("State_HelpLine.csv")
    select2=st.selectbox('Select Your State',StateHelpDF['Name_of_State'])
    state_data_Helpline= (StateHelpDF[StateHelpDF['Name_of_State']==select2])
    
    Helpline_Number=pd.DataFrame(state_data_Helpline['Helpline_Number'])
    Helpline_List=state_data_Helpline['Helpline_Number'].tolist()[0]
    State_Service=pd.DataFrame(state_data_Helpline['State_Service_Link'])
    Service_List=state_data_Helpline['State_Service_Link'].tolist()[0]
    st.subheader("📞 Help Line Number:    "+Helpline_List)
    
    st.subheader("📎 Service Link:    "+Service_List)


if menu=='About Us':
    st.title("**About Us: **")
    image=Image.open("MGM_College_Logo.jpg")
    st.image(image,use_column_width=False)
    st.info("**Syed Saad Ali  \nInformation Technology (B.Tech)  \n MGM's College of Engineering, Nanded**")
    st.info("**Rastogi Raghav  \nInformation Technology (B.Tech)  \n MGM's College of Engineering, Nanded**")
    st.balloons()




#Covid-19 Detector Model UI
if menu=="Covid-19 Detector":
    st.title("Covid-19 Detector")


    def file_selector(folder_path=''):

        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return os.path.join(folder_path, selected_filename)



    # Select a file
    try:
        folder_path = ''
  
        folder_path = st.text_input('Enter folder path', '')
        if folder_path!=None:
            filename = file_selector(folder_path=folder_path)
            st.write('You selected `%s`' % filename)
    except:
        st.error("Select an appropriate folder")

   

    try:

    
        model = tf.keras.models.load_model('Covid_model.h5')
        from keras.applications.vgg16 import preprocess_input
        img = ge.load_img(filename, target_size=(224, 224)) #insert a random covid-19 x-ray image
        
        image = Image.open(filename)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        x = ge.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        classes = model.predict(img_data)
        New_pred = np.argmax(classes, axis=1)
        if st.button("Predict"):
            progress=st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            if New_pred==[1]:
                st.success('Great!It is looking Normal')
                image=Image.open("CovidNegative_Happy.png")
                st.image(image,use_column_width=True)
            else:
                st.info('OOPS! You may have Covid-19')
                image=Image.open("CovidPositive_Sad.png")
                st.image(image,use_column_width=True)
    except:
        st.error("Select Data of type image only; select chest x-ray image")

    
    
  
  
