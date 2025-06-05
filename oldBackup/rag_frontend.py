# The below frontend code is provided by AWS and Streamlit. I have only modified it to make it look attractive.
import streamlit as st 
import rag_backend as demo ### replace rag_backend with your backend filename

st.set_page_config(page_title="AI driven Customer Engagement") ### Modify Heading

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">AI driven Customer Engagement 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True) ### Modify Title

if 'vector_index' not in st.session_state: 
    with st.spinner("uploading file and indexing...Please wait for a moment:-⌛⌛⌛"): ###spinner message
        st.session_state.vector_index = demo.hr_index() ### Your Index Function name from Backend File

input_text = st.text_area("Input text", label_visibility="collapsed") 
go_button = st.button("⚽ Get insights", type="secondary") ### Button Name

if go_button: 

    with st.spinner("📢🕰️Wait for magic...Fetching reliable answer:-✨✨✨"): ### Spinner message
        response_content = demo.hr_rag_response(index=st.session_state.vector_index, question=input_text) ### replace with RAG Function from backend file
        st.write(response_content) 