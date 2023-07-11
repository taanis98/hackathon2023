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
task = st.sidebar.selectbox("Select Task: ", ("Homepage", "Dashboard", "Messages"))
def get_model(task):
    if task == "Homepage":
        uploaded_file = st.file_uploader("Upload Disease Image")
        if uploaded_file is not None:
            if save_uploaded_file(uploaded_file): 
                # upload model
                st.image(uploaded_file)

    if task == "Dashboard":
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

    if task == "Messages":
        st.write(task)
        asyncio.run(messages())

get_model(task)        