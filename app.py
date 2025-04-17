# app.py

import warnings
warnings.filterwarnings("ignore")

# --- Standard Imports ---
import streamlit as st
from dotenv import load_dotenv
import os
import base64
from base64 import b64decode
import io 
from PIL import Image
import re
import logging

# --- LLM / Embedding Imports ---
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Qdrant / Langchain Imports ---
from qdrant_client import QdrantClient, models
from langchain_qdrant import Qdrant
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain.schema.document import Document

# --- Custom Modules ---
from arxiv_call import fetch_arxiv_papers_metadata, process_and_store_gemini
from chatgpt_pipeline import process_and_store_openai

# --- Logging Configuration ---
# Suppress noisy loggers
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('unstructured').setLevel(logging.ERROR)
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('qdrant_client').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.WARNING)


# --- Initialization ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')

# --- App config ---
st.set_page_config(page_title="Multimodal ArXiv Chatbot", page_icon="📄", layout="wide")
st.title("📄 Multimodal ArXiv Chatbot")
st.write("Search arXiv papers, process with Gemini or OpenAI, and chat about their content.")

# --- API Key Checks ---
keys_ok = True
if not GOOGLE_API_KEY: st.warning("⚠️ Google API Key missing. Gemini unavailable.", icon="🔑"); keys_ok = False
if not OPENAI_API_KEY: st.warning("⚠️ OpenAI API Key missing. OpenAI unavailable.", icon="🔑"); keys_ok = False
if not QDRANT_API_KEY or not QDRANT_URL: st.error("❌ Qdrant Key/URL missing. Cannot continue."); st.stop()

# --- Model/Embedding Definitions ---
gemini_chat_model = None
gemini_embeddings = None
gemini_embedding_dim = 768
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_chat_model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Gemini models/embeddings initialized.")
    except Exception as e: st.error(f"Gemini init failed: {e}"); GOOGLE_API_KEY = None

openai_chat_model = None
openai_embeddings = None
openai_embedding_dim = 1536
if OPENAI_API_KEY:
    try:
        openai_chat_model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        print("OpenAI models/embeddings initialized.")
    except Exception as e: st.error(f"OpenAI init failed: {e}"); OPENAI_API_KEY = None

# --- Qdrant Client Setup ---
collection_name_gemini = "multimodal_arxiv_rag_gemini_v3"
collection_name_openai = "multimodal_arxiv_rag_openai_v3"
id_key = "doc_id"

try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    qdrant_client.get_collections() 
    print("Qdrant client connected.")
except Exception as e: st.error(f"❌ Qdrant connection failed: {e}"); st.stop()

# --- Session State Initialization ---
if "docstore" not in st.session_state:
    # Holds originals for the currently selected paper
    st.session_state.docstore = InMemoryStore()
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "papers_metadata" not in st.session_state: st.session_state.papers_metadata = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "selected_paper_meta" not in st.session_state: st.session_state.selected_paper_meta = None
if "last_selected_paper_id" not in st.session_state: st.session_state.last_selected_paper_id = None
if "processing_done" not in st.session_state: st.session_state.processing_done = False
if "processing_method" not in st.session_state: st.session_state.processing_method = None


# --- RAG Setup Helper ---
def setup_rag_components(method):
    """Initializes vectorstore wrapper and retriever based on chosen method."""
    global gemini_embeddings, openai_embeddings

    current_vectorstore = None
    current_embeddings = None
    current_collection = None

    if method == 'gemini':
        if not GOOGLE_API_KEY or not gemini_embeddings: return False
        current_embeddings = gemini_embeddings
        current_collection = collection_name_gemini
    elif method == 'openai':
        if not OPENAI_API_KEY or not openai_embeddings: return False
        current_embeddings = openai_embeddings
        current_collection = collection_name_openai
    else: return False

    try:
        print(f"Setting up RAG: Method={method}, Collection={current_collection}")
        current_vectorstore = Qdrant(
            client=qdrant_client, collection_name=current_collection, embeddings=current_embeddings,
        )
        # Use the session docstore, which was updated by the processing function
        st.session_state.retriever = MultiVectorRetriever(
            vectorstore=current_vectorstore,
            docstore=st.session_state.docstore,
            id_key=id_key, search_kwargs={'k': 5}
        )
        print("RAG components set up successfully.")
        return True
    except Exception as e:
        st.error(f"Error setting up RAG components for {method}: {e}")
        st.session_state.retriever = None
        return False

# --- UI Sections ---
with st.sidebar:
    st.header("Paper Selection")
    if not st.session_state.selected_paper_meta:
        st.subheader("1. Find Papers")
        keyword = st.text_input("Search keyword", key="search_kw")
        num_papers = st.number_input("Max papers", 1, 10, 3, key="search_num")
        if st.button("🔍 Fetch Papers", key="fetch_button"):
             if not keyword: st.warning("Enter keyword.")
             else:
                 with st.spinner("Searching ArXiv..."):
                     success, details = fetch_arxiv_papers_metadata(keyword, num_papers)
                     if success and details: st.session_state.papers_metadata = details; st.success(f"Found {len(details)} papers.")
                     else: st.session_state.papers_metadata = []; st.error("No relevant papers found.")
                 st.session_state.selected_paper_meta = None; st.session_state.processing_done = False
                 st.session_state.processing_method = None; st.session_state.chat_history = []
                 st.session_state.retriever = None; st.session_state.docstore = InMemoryStore()
                 st.rerun()

        if st.session_state.papers_metadata:
            st.markdown("---")
            st.subheader("2. Select & Process Paper")

            # Create dummy vectorstore instances needed for passing to processing functions
            # The processing functions will re-initialize these internally.
            dummy_vs_gemini = None
            if gemini_embeddings: dummy_vs_gemini = Qdrant(client=qdrant_client, collection_name="dummy", embeddings=gemini_embeddings)
            dummy_vs_openai = None
            if openai_embeddings: dummy_vs_openai = Qdrant(client=qdrant_client, collection_name="dummy", embeddings=openai_embeddings)

            for i, paper_meta in enumerate(st.session_state.papers_metadata):
                st.markdown(f"**{i+1}. {paper_meta['title'][:60]}...**")
                with st.expander("Details"):
                    st.caption(f"Authors: {paper_meta.get('authors', 'N/A')}")
                    st.write(f"Summary: {paper_meta.get('summary', 'N/A')}")
                    arxiv_id = paper_meta.get('link', 'N/A'); pdf_link = paper_meta.get('pdf_link', '#')
                    arxiv_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id != 'N/A' else '#'
                    st.write(f"Links: [arXiv]({arxiv_link}) | [PDF]({pdf_link})")
                    st.write(f"Published: {paper_meta.get('published', 'N/A')} | Updated: {paper_meta.get('updated', 'N/A')}")

                col1, col2 = st.columns(2)
                # --- Gemini Button ---
                with col1:
                    if st.button(f"Process Gemini", key=f"gemini_{i}", disabled=not GOOGLE_API_KEY or not dummy_vs_gemini, help="Uses Google Gemini"):
                         st.session_state.processing_method = 'gemini'
                         current_paper_id = paper_meta['link']
                         if st.session_state.last_selected_paper_id != current_paper_id or st.session_state.processing_method != 'gemini':
                             st.session_state.chat_history = []

                         with st.spinner(f"Processing with Gemini..."):
                             # Call specific Gemini function from arxiv_call
                             success, updated_docstore = process_and_store_gemini(
                                 paper_meta, qdrant_client, dummy_vs_gemini, st.session_state.docstore,
                                 collection_name_gemini, gemini_embeddings, gemini_embedding_dim, id_key
                             )
                         if success:
                             st.session_state.docstore = updated_docstore 
                             st.session_state.selected_paper_meta = paper_meta
                             st.session_state.last_selected_paper_id = current_paper_id
                             st.session_state.processing_done = setup_rag_components('gemini')
                             if st.session_state.processing_done: st.success("✅ Processed with Gemini!"); st.rerun()
                             else: st.error("❌ Processed, but RAG setup failed.")
                         else: st.error(f"❌ Gemini processing failed."); st.session_state.processing_done = False
                # --- OpenAI Button ---
                with col2:
                    if st.button(f"Process OpenAI", key=f"openai_{i}", disabled=not OPENAI_API_KEY or not dummy_vs_openai, help="Uses OpenAI GPT"):
                         st.session_state.processing_method = 'openai'
                         current_paper_id = paper_meta['link']
                         if st.session_state.last_selected_paper_id != current_paper_id or st.session_state.processing_method != 'openai':
                             st.session_state.chat_history = []

                         with st.spinner(f"Processing with OpenAI..."):
                             # Call specific OpenAI function from chatgpt_pipeline
                             success, updated_docstore = process_and_store_openai(
                                 paper_meta, qdrant_client, dummy_vs_openai, st.session_state.docstore,
                                 collection_name_openai, openai_embeddings, openai_embedding_dim,
                                 OPENAI_API_KEY, id_key # Pass key
                             )
                         if success:
                             st.session_state.docstore = updated_docstore
                             st.session_state.selected_paper_meta = paper_meta
                             st.session_state.last_selected_paper_id = current_paper_id
                             st.session_state.processing_done = setup_rag_components('openai')
                             if st.session_state.processing_done: st.success("✅ Processed with OpenAI!"); st.rerun()
                             else: st.error("❌ Processed, but RAG setup failed.")
                         else: st.error(f"❌ OpenAI processing failed."); st.session_state.processing_done = False
                st.divider()

    elif st.session_state.selected_paper_meta:
        st.subheader("Current Paper")
        st.info(f"**{st.session_state.selected_paper_meta['title']}**")
        st.caption(f"Processed using: **{st.session_state.processing_method.upper()}**")
        if st.button("🔙 Change Paper / Re-process"):
            st.session_state.selected_paper_meta = None; st.session_state.last_selected_paper_id = None
            st.session_state.chat_history = []; st.session_state.processing_done = False
            st.session_state.processing_method = None; st.session_state.retriever = None
            st.session_state.docstore = InMemoryStore()
            st.rerun()

# --- Main Chat Area ---
st.header("💬 Ask Questions About the Paper")

if not st.session_state.selected_paper_meta:
    st.info("Select and process a paper using the sidebar.")
elif not st.session_state.processing_done or not st.session_state.retriever:
    st.warning("Paper processing selected but RAG components not ready.")
    if st.button("Retry RAG Setup"):
        if setup_rag_components(st.session_state.processing_method): st.rerun()
        else: st.error("Failed RAG setup.")
else:
    # --- RAG Helper Functions ---
    def display_base64_image(base64_string):
        try: st.image(Image.open(io.BytesIO(b64decode(base64_string))), use_container_width=True)
        except Exception as e: st.error(f"Err display img: {e}")

    def parse_docs_from_retriever(docs):
        """ Parses docs (expects full content from MultiVectorRetriever). """
        parsed = {"images": [], "texts": [], "tables": []}
        for doc in docs:
            content = doc.page_content if isinstance(doc, Document) else doc
            metadata = doc.metadata if isinstance(doc, Document) else {}
            doc_type = metadata.get("type", None)
            if isinstance(content, str):
                is_image = False
                try:
                    if len(content) > 100 and len(content) % 4 == 0 and re.match(r"^[A-Za-z0-9+/]*={0,2}$", content):
                        b64decode(content); is_image = True
                except: is_image = False
                if is_image or doc_type == "image": parsed["images"].append(content)
                elif doc_type == "table": parsed["tables"].append(content)
                else: parsed["texts"].append(content)
        return parsed

    def build_rag_prompt_dynamic(context_dict, user_question, method):
        """ Builds prompt for Gemini or OpenAI. """
        context_text = "\n---\n".join(context_dict.get("texts", []))
        context_text += "\n--- TABLES ---\n" + "\n---\n".join(context_dict.get("tables", []))
        base_prompt = f"Use the context (text, tables, images) to answer the question.\nCONTEXT:\n{context_text}\nQUESTION: {user_question}\nANSWER:"
        prompt_parts, image_refs = [base_prompt], []
        img_list = context_dict.get("images", [])

        if method == 'gemini':
            for i, img_b64 in enumerate(img_list):
                 try: prompt_parts.append(Image.open(io.BytesIO(base64.b64decode(img_b64)))); image_refs.append({"id": f"img_{i}", "base64": img_b64})
                 except Exception as e: print(f"Skip invalid gemini img: {e}")
        elif method == 'openai':
            for i, img_b64 in enumerate(img_list):
                 if isinstance(img_b64, str) and len(img_b64) > 100: image_refs.append({"id": f"img_{i}", "base64": img_b64})
        return prompt_parts, image_refs

    def get_rag_answer_dynamic(question, retriever, method):
        """ Generates RAG answer using the appropriate model. """
        global gemini_chat_model, openai_chat_model
        try:
            if not retriever: return "Retriever not available.", [], []
            retrieved_docs = retriever.invoke(question)
            if not retrieved_docs: return "No relevant context found.", [], []

            parsed_context = parse_docs_from_retriever(retrieved_docs)
            prompt_content, image_refs = build_rag_prompt_dynamic(parsed_context, question, method)
            response_text = "Error generating response."

            if method == 'gemini' and gemini_chat_model:
                 response = gemini_chat_model.generate_content(prompt_content) 
                 response_text = response.text.strip()
            elif method == 'openai' and openai_chat_model:
                 # Construct OpenAI message format
                 messages = [HumanMessage(content=[{"type": "text", "text": prompt_content[0]}])] 
                 for img_ref in image_refs: 
                      messages[0].content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_ref['base64']}"}})
                 response = openai_chat_model.invoke(messages)
                 response_text = response.content.strip()
            else: response_text = f"Error: {method} chat model unavailable."

            return response_text, image_refs, retrieved_docs
        except Exception as e:
            st.error(f"RAG Error ({method}): {e}"); import traceback; traceback.print_exc()
            return f"RAG Error ({method}).", [], []

    # --- Chat Interface Logic ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("images"):
                 st.write("*(Context Images)*")
                 cols = st.columns(min(len(msg["images"]), 4))
                 for i, img_ref in enumerate(msg["images"]):
                      with cols[i % 4]: display_base64_image(img_ref["base64"])

    user_question = st.chat_input(f"Ask about '{st.session_state.selected_paper_meta['title']}'...")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"): st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner(f"Asking {st.session_state.processing_method.upper()}..."):
                answer_text, answer_images, retrieved = get_rag_answer_dynamic(
                    user_question, st.session_state.retriever, st.session_state.processing_method
                )
                st.markdown(answer_text)
                if answer_images:
                     st.write("*(Context Images)*")
                     cols = st.columns(min(len(answer_images), 4))
                     for i, img_ref in enumerate(answer_images):
                         with cols[i % 4]: display_base64_image(img_ref["base64"])
                st.session_state.chat_history.append({
                    "role": "assistant", "content": answer_text, "images": answer_images
                })