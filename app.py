import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

load_dotenv()
import google.generativeai as gai
gai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

from Arxiv_call import fetch_and_store_arxiv_papers

def get_conversation_chain():
    prompt_template = """"You are an AI assistant called 'ArXiv Assist' that has a conversation with the user and answers to questions. "
                     "Try looking into the research papers content provided to you to respond back. If you could not find any relevant information there, mention something like 'I do not have enough information form the research papers. However, this is what I know...' and then try to formulate a response by your own. "
                     "There could be cases when user does not ask a question, but it is just a statement. Just reply back normally and accordingly to have a good conversation (e.g. 'You're welcome' if the input is 'Thanks'). "
                     "If you mention the name of a paper, provide an arxiv link to it. "
                     "Be polite, friendly, and format your response well (e.g., use bullet points, bold text, etc.). "
                     "Below are relevant excerpts from the research papers:
Context: {context}

Question: {question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def get_answer_from_faiss(user_question, index_path="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain.run(input_documents=docs, question=user_question)
    st.write("Answer: ", response)

def main():
    st.set_page_config(page_title='Chat With ArXiv Papers', page_icon='📄')
    st.title('📄 Chat With ArXiv Papers')

    with st.sidebar:
        st.header('Retrieve Papers')
        keyword = st.text_input("Enter search keyword")
        num_papers = st.number_input("Number of papers", min_value=1, max_value=10, value=3)
        if st.button("Fetch & Process"):
            with st.spinner("Processing papers..."):
                success = fetch_and_store_arxiv_papers(keyword, num_papers)
                if success:
                    st.success("Papers processed and stored!")
                else:
                    st.error("No relevant papers found or failed to process.")

    user_question = st.text_input("Ask your question")
    if user_question:
        get_answer_from_faiss(user_question)

if __name__ == "__main__":
    main()