import streamlit as st

st.set_page_config(page_title='Chat With ArXiv Papers', page_icon='📄')

import os
from dotenv import load_dotenv
import google.generativeai as gai
from Arxiv_call import fetch_and_store_arxiv_papers, load_conversation_chain
from langchain.schema import HumanMessage, AIMessage

load_dotenv()
gai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']


    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

def main():
    
    st.title('📄 Chat With ArXiv Papers')
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "initial_message_displayed" not in st.session_state:
        st.session_state.initial_message_displayed = False
    with st.sidebar:
        st.header('Retrieve Papers')
        keyword = st.text_input("Enter search keyword")
        num_papers = st.number_input("Number of papers", min_value=1, max_value=10, value=3)
        if st.button("Fetch & Process"):
            with st.spinner("Processing papers..."):
                success = fetch_and_store_arxiv_papers(keyword, num_papers)
                if success:
                    st.session_state.conversation = load_conversation_chain()
                    st.success("Papers processed and stored!")
                else:
                    st.error("No relevant papers found or failed to process.")
    if "initial_message" in st.session_state and not st.session_state.initial_message_displayed:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.initial_message)
        st.session_state.initial_message_displayed = True
        
    user_question = st.chat_input("Ask your question here")
    if user_question:
        if st.session_state.conversation:
            handle_user_input(user_question)
        else:
            st.warning("Please fetch and process papers first.")
        #get_answer_from_faiss(user_question)

if __name__ == "__main__":
    main()