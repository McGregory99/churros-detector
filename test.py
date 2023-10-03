import streamlit as st
from fastai.vision.all import *
import torch
from PIL import Image




#Labeling function required for load_learner to work
def GetLabel(fileName):
  return fileName.split('-')[0]

def predict(img_path='\img\0.jpg'):
    img = Image.open('0.jpg')
    label,_,probs = learn.predict(img)
    print(f'{label} ({torch.max(probs).item()*100:.0f}%)')

learn = torch.load('export.pkl')
print(learn)
#predict()