import streamlit as st
from nott.sdfg import data_loading, data_store, retrive_answer, final_result

st.title("YouTube Video RAG System")

st.header("Step 1: Provide a YouTube Video URL")
youtube_url = st.text_input("Enter YouTube video link:")

if st.button("Download & Process Video"):
    with st.spinner("Processing video..."):
        metadata_vid, output_folder = data_loading(youtube_url)
        index=data_store(output_folder)
        st.session_state['index'] = index
        st.session_state['metadata_vid'] = metadata_vid
        st.session_state['output_folder'] = output_folder
        st.success("Video processed and data stored!")


st.header("Step 2: Ask a Question About the Video")
user_question = st.text_input("Type your question here:")
if st.button("Submit Question"):
    if all(k in st.session_state for k in ('index', 'metadata_vid', 'output_folder')):
        index = st.session_state['index']
        metadata_vid = st.session_state['metadata_vid']
        output_folder = st.session_state['output_folder']
        image_documents, context_str = retrive_answer(index, user_question, metadata_vid, output_folder)
        response = final_result(metadata_vid, context_str, user_question, image_documents)
        st.markdown(f"**Answer:** {response}")
    else:
        st.error("Please process a video first (Step 1) before asking questions.")
    # image_documents, context_str=retrive_answer(index, user_question, metadata_vid,output_folder)
    # response=final_result(metadata_vid, context_str, user_question, image_documents)
    # st.markdown(f"**Answer:** {response}")

