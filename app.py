import streamlit as st
import datetime

# Title and description
st.title("Form for the Users")
st.write("Here, you can answer some questions in this form.")

# User inputs
user_id = st.text_input("ID", value="Your ID", max_chars=7)
info = st.text_area(
    "Share some information about you",
    "Put information here",
    help="You can write about your hobbies or family.",
)
age = st.number_input("Age", min_value=18, max_value=100, step=1)
birth_date = st.date_input(
    "Date of Birth",
    min_value=datetime.date(1921, 1, 1),
    max_value=datetime.date(2003, 12, 31),
)
smoke = st.checkbox("Do you smoke?")
genre = st.radio(
    "Which movie genre do you like?", options=["horror", "adventure", "romantic"]
)
weight = st.slider("Choose your weight", min_value=40.0, max_value=150.0, step=0.5)
physical_form = st.selectbox(
    "Select your level of physical condition", options=["Bad", "Normal", "Good"]
)
colors = st.multiselect(
    "What are your favorite colors?", options=["Green", "Yellow", "Red", "Blue", "Pink"]
)

# Image uploader
image = st.file_uploader("Upload your photo", type=["jpg", "png"])

# Display columns with images
col1, col2 = st.columns(2)  # Updated the deprecated beta_columns to columns
with col1:
    st.image("https://static.streamlit.io/examples/cat.jpg", width=300)
    st.button("Like cats")
with col2:
    st.image("https://static.streamlit.io/examples/dog.jpg", width=355)
    st.button("Like dogs")

# Submit button
submit = st.button("Submit")
if submit:
    st.write("You submitted the form")

# Sidebar button
click = st.sidebar.button("Click me!")
if click:
    st.sidebar.write("You clicked the button")