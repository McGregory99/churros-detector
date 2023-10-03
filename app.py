import streamlit as st
from fastai.vision.all import *
import torch

def main():
    st.title("Detector de Churros")

    columns = st.columns(2)
    with columns[0]:
        with st.container():
            uploaded_file = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)
                st.write("")
                st.write("Clasificación de la imagen o cualquier procesamiento adicional aquí.")
    
    with columns[1]:
        with st.container():
            if st.button("¿Hay churros?"):
                st.text("CHURROSS")
                text = predict(uploaded_file)
                st.text(text)

def predict(img):
    #img = PILImage.create()
    label,_,probs = learn.predict(img)
    return f'{label} ({torch.max(probs).item()*100:.0f}%)'

if __name__ == "__main__":
    try:
        learn = load_learner('export.pkl')
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        exit()
        
    main()
