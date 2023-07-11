import streamlit as st
import pandas as pd
import numpy as np
import os
import asyncio
import json
import nats
import cv2
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import time  
import plotly.express as px  

##########

dataset_url = "https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv"

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_data()

##########

st.title('Hackathon Team Error404')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded_images/',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0


async def messages():
    # Connect to NATS!
    nc = await nats.connect("nats://10.148.129.96:4222")
    sub = await nc.subscribe("rpi.status")

    while True:
        # Process a message
        msg = await sub.next_msg()
        st.write("Received:", msg.data)

    # Close NATS connection
    # await nc.close()

def json_decode():
    # code
    msg = 'trying'
    return msg

# All the new code 
task = st.sidebar.selectbox("Select Task: ", ("Homepage", "Live", "Dashboard", "Messages"))
def get_model(task):
    if task == "Homepage":
        uploaded_file = st.file_uploader("Upload Disease Image")
        if uploaded_file is not None:
            if save_uploaded_file(uploaded_file): 
                # upload model
                st.image(uploaded_file)

    if task == "Live":
        st.write(task)
        # uploaded_file = st.file_uploader("Choose a file")
        # st.image(uploaded_file)
        # capture = cv2.VideoCapture("rtsp://10.148.129.96:8554/webcam")
        # # FRAME_WINDOW=st.image([])
        # # video_file = open('myvideo.mp4', 'rb')
        # video_bytes = capture.read()
        # # x, frame = capture.read()
        # st.video(video_bytes)
        # ### GRAB NEW IMAGE
        # # x, frame = capture.read()
        # # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # # time.sleep(0.025)


        vf = cv2.VideoCapture("rtsp://10.148.129.96:8554/webcam")
        stframe = st.empty()

        while vf.isOpened():
            ret, frame = vf.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(gray)

    if task == "Dashboard":
        st.write(task)
        # top-level filters
        job_filter = st.selectbox("Select the Job", pd.unique(df["job"]))

        # creating a single-element container
        placeholder = st.empty()

        # dataframe filter
        df = df[df["job"] == job_filter]

        # near real-time / live feed simulation
        for seconds in range(200):

            df["age_new"] = df["age"] * np.random.choice(range(1, 5))
            df["balance_new"] = df["balance"] * np.random.choice(range(1, 5))

            # creating KPIs
            avg_age = np.mean(df["age_new"])

            count_married = int(
                df[(df["marital"] == "married")]["marital"].count()
                + np.random.choice(range(1, 30))
            )

            balance = np.mean(df["balance_new"])

            with placeholder.container():

                # create three columns
                kpi1, kpi2, kpi3 = st.columns(3)

                # fill in those three columns with respective metrics or KPIs
                kpi1.metric(
                    label="Age ⏳",
                    value=round(avg_age),
                    delta=round(avg_age) - 10,
                )
                
                kpi2.metric(
                    label="Married Count 💍",
                    value=int(count_married),
                    delta=-10 + count_married,
                )
                
                kpi3.metric(
                    label="A/C Balance ＄",
                    value=f"$ {round(balance,2)} ",
                    delta=-round(balance / count_married) * 100,
                )

                # create two columns for charts
                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.markdown("### First Chart")
                    fig = px.density_heatmap(
                        data_frame=df, y="age_new", x="marital"
                    )
                    st.write(fig)
                    
                with fig_col2:
                    st.markdown("### Second Chart")
                    fig2 = px.histogram(data_frame=df, x="age_new")
                    st.write(fig2)

                st.markdown("### Detailed Data View")
                st.dataframe(df)
                time.sleep(1)
        
    
    if task == "Messages":
        st.write(task)
        asyncio.run(messages())

get_model(task)        