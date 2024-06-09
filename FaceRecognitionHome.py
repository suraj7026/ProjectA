import streamlit as st


st.set_page_config(page_title='Attendance App', layout='wide')
st.title("Attendance App Project A")

with st.spinner('Loading Data...'):
    import face_rec

st.success("Loaded")