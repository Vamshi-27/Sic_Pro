import streamlit as st
from generate import generate_story

st.title("AI Story Generator")
st.write("Click the button below to generate a unique story.")

if st.button("Generate Story"):
    story = generate_story()
    st.write(story)
