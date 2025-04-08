import streamlit as st
from io import BytesIO
from base64 import b64decode
from PIL import Image
import google.generativeai as genai
import dotenv
import os
from MultimodalRAG import fetch_and_summarize_arxiv_papers, retriever

st.set_page_config(page_title="Multimodal ArXiv Chat", page_icon="📄")

dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

def display_base64_image(base64_code):
    image_data = b64decode(base64_code)
    image = Image.open(BytesIO(image_data))
    st.image(image)

def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(context_dict, user_question):
    context_text = "".join([
        text.page_content if hasattr(text, 'page_content') else str(text)
        for text in context_dict["texts"]
    ])

    prompt = f"""
    Answer the question based only on the following context, which can include text.
    Be detailed if the user asks to explain in detail, concise if user asks for a brief answer, and include image context if requested.

    Context:
    {context_text}

    Question:
    {user_question}
    """

    contents = [
        {"role": "user", "parts": [{"text": prompt}]}
    ]

    for image_b64 in context_dict["images"]:
        contents[0]["parts"].append({
            "image": {"base64": image_b64}
        })

    return contents


def get_rag_answer(question):
    docs = retriever.invoke(question)
    parsed_context = parse_docs(docs)
    gemini_prompt = build_prompt(parsed_context, question)
    response = chat_model.generate_content(contents=gemini_prompt)
    return response.text.strip(), parsed_context

def main():
    st.title("📄 Multimodal ArXiv Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Retrieve Papers")
        keyword = st.text_input("Enter search keyword")
        num_papers = st.number_input("Number of papers", min_value=1, max_value=10, value=2)
        if st.button("Fetch & Process"):
            with st.spinner("Processing papers..."):
                result = fetch_and_summarize_arxiv_papers(keyword, num_papers)
                if result:
                    st.session_state.chat_history = []
                    st.success("Papers processed and stored!")
                    with st.expander("🧾 Downloaded Papers"):
                        for idx, (title, abstract) in enumerate(zip(result['titles'], result['abstracts']), start=1):
                            st.markdown(f"**{idx}. {title}**\n\n*Abstract:* {abstract}\n\n")
                else:
                    st.error("No relevant papers found.")

    user_question = st.chat_input("Ask your question here")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.spinner("Retrieving answer..."):
            answer, context = get_rag_answer(user_question)
        with st.chat_message("assistant"):
            st.markdown(answer)
            if context["images"]:
                st.markdown("**Retrieved Images:**")
                for img_b64 in context["images"]:
                    display_base64_image(img_b64)
        st.session_state.chat_history.append((user_question, answer))

if __name__ == "__main__":
    main()
