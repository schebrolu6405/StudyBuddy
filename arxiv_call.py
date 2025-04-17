# arxiv_call.py

import os
import io
import re
import requests
import arxiv
import tempfile
import uuid
import base64
import fitz
from PIL import Image
from tqdm import tqdm
from fuzzywuzzy import fuzz
from unstructured.partition.pdf import partition_pdf
from collections import defaultdict
import asyncio 

# --- Langchain / Qdrant ---
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

# --- LLM / Embeddings ---
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from asyncio_throttle import Throttler
import nest_asyncio

# --- Logging ---
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# --- Load API Keys ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 

# --- Initialize Models (Conditionally) ---
model_text_gemini = None
model_vision_gemini = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model_text_gemini = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        model_vision_gemini = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        print("Gemini models initialized.")
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini models: {e}")
else:
    print("Warning: Google API Key not found. Gemini functions disabled.")

# --- Helper Functions ---
def sanitize_filename(title):
    """Removes characters problematic for filenames."""
    return re.sub(r'[<>:"/\\|?*]', '_', title)

def is_relevant_paper(title, abstract, keyword, threshold=50):
    """Checks if a paper title/abstract matches the keyword."""
    keyword_words = keyword.lower().split()
    title_words = title.lower().split()
    abstract_words = abstract.lower().split()
    match_count = sum(1 for word in keyword_words if word in title_words or word in abstract_words)
    fuzzy_score = max(
        fuzz.partial_ratio(keyword.lower(), title.lower()),
        fuzz.partial_ratio(keyword.lower(), abstract.lower())
    )
    return match_count / len(keyword_words) > 0.5 or fuzzy_score > threshold

# --- ArXiv Metadata Fetching ---
def fetch_arxiv_papers_metadata(query, num_papers):
    """Fetches metadata for relevant ArXiv papers."""
    print(f"Searching ArXiv for: {query}")
    search_query = f"ti:{query}"
    search = arxiv.Search(query=search_query, max_results=num_papers * 2, sort_by=arxiv.SortCriterion.Relevance)
    client = arxiv.Client()
    try:
        results = list(client.results(search))
    except Exception as e:
        print(f"Error fetching from ArXiv: {e}")
        return False, []

    relevant_papers_meta = []
    count = 0
    for res in results:
        if count >= num_papers: break
        entry_id = res.entry_id.split('/')[-1]
        pdf_link = res.pdf_url
        if not pdf_link: continue
        if is_relevant_paper(res.title, res.summary, query):
            relevant_papers_meta.append({
                "title": res.title, "summary": res.summary, "link": entry_id,
                "pdf_link": pdf_link, "published": res.published.strftime('%Y-%m-%d') if res.published else "N/A",
                "updated": res.updated.strftime('%Y-%m-%d') if res.updated else "N/A",
                "authors": ", ".join(author.name for author in res.authors)
            })
            count += 1
    if not relevant_papers_meta: print(f"No relevant papers found for: {query}"); return False, []
    print(f"Found {len(relevant_papers_meta)} relevant papers.")
    return True, relevant_papers_meta

# --- Gemini Specific Summarization / Extraction Functions ---
def get_images_gemini(chunks):
    """Extracts image base64 from Unstructured chunks (for Gemini path)."""
    images = []
    for chunk in chunks:
        elements_to_check = []
        if "CompositeElement" in str(type(chunk)) and hasattr(chunk.metadata, 'orig_elements'):
             elements_to_check = chunk.metadata.orig_elements or []
        elif "Image" in str(type(chunk)):
             elements_to_check = [chunk] 

        for el in elements_to_check:
            if "Image" in str(type(el)) and hasattr(el.metadata, 'image_base64') and el.metadata.image_base64:
                images.append({"base64": el.metadata.image_base64})
    # Simple deduplication
    unique_images = {img['base64']: img for img in images}.values()
    return list(unique_images)

def summarize_text_element_gemini(text):
    """Summarizes text chunks using the Gemini text model."""
    if not model_text_gemini: return f"Error: Gemini Text Model unavailable. Text: {text[:100]}..."
    prompt_text = f"""Summarize this text chunk from a research paper concisely: {text}"""
    try:
        response = model_text_gemini.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        print(f"Error summarizing text with Gemini: {e}")
        return f"Error summarizing: {text[:100]}..."

def summarize_image_base64_gemini(base64_code):
    """Summarizes image content using the Gemini vision model."""
    if not model_vision_gemini: return "Error: Gemini Vision Model unavailable."
    try:
        image_data = base64.b64decode(base64_code)
        img = Image.open(io.BytesIO(image_data))
        prompt = "Describe this image from a research paper. Focus on diagrams, plots, or key visual elements."
        response = model_vision_gemini.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        print(f"Error summarizing image with Gemini: {e}")
        return "Error summarizing image."

def summarize_table_html_gemini(html, caption=None):
    """Summarizes HTML table content using Gemini."""
    if not model_text_gemini: return f"Error: Gemini Text Model unavailable. Table: {html[:100]}..."
    prompt = "Summarize this table from a scientific paper."
    if caption: prompt += f"\nCaption: \"{caption}\""
    prompt += f"\n\nTable HTML:\n{html}"
    try:
        response = model_text_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error summarizing table with Gemini: {e}")
        return f"Error summarizing table: {html[:100]}..."

# --- Gemini Specific PDF Processing Function ---
def process_pdf_gemini(paper_meta, output_path="./output_gemini"):
    """Downloads, partitions, and summarizes using Gemini models."""
    pdf_url = paper_meta["pdf_link"]
    print(f"--- Gemini Processor: Processing PDF: {paper_meta['title']} ---")
    os.makedirs(output_path, exist_ok=True)
    pdf_path = None
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf.write(response.content); tmp_pdf.flush(); pdf_path = tmp_pdf.name

        print(f"--- Gemini Processor: Partitioning PDF {pdf_path} ---")
        chunks = partition_pdf(
            filename=pdf_path, infer_table_structure=True, strategy="hi_res",
            extract_image_block_types=["Image", "Table"], extract_image_block_to_payload=True,
            chunking_strategy="by_title", max_characters=4000, combine_text_under_n_chars=1000,
            new_after_n_chars=3000, include_page_breaks=True
        )
        print(f"--- Gemini Processor: Partitioned into {len(chunks)} chunks ---")

        texts = [c.text for c in chunks if "CompositeElement" in str(type(c)) and c.text]
        images_meta = get_images_gemini(chunks) 
        tables_html = [c.metadata.text_as_html for c in chunks if "Table" in str(type(c)) and hasattr(c.metadata, 'text_as_html')]
        print(f"--- Gemini Processor: Extracted {len(texts)} texts, {len(images_meta)} images, {len(tables_html)} tables ---")

        print("--- Gemini Processor: Summarizing elements... ---")
        text_summaries = [summarize_text_element_gemini(t) for t in tqdm(texts, desc="Gemini Texts")]
        image_summaries = [summarize_image_base64_gemini(m["base64"]) for m in tqdm(images_meta, desc="Gemini Images")]
        table_summaries = [summarize_table_html_gemini(h) for h in tqdm(tables_html, desc="Gemini Tables")]

        print(f"--- Gemini Processor: Finished summarizing {paper_meta['title']} ---")
        return texts, tables_html, images_meta, text_summaries, table_summaries, image_summaries

    except Exception as e:
        print(f"--- Gemini Processor: Error processing {paper_meta['title']}: {e} ---")
        import traceback; traceback.print_exc(); return [], [], [], [], [], []
    finally:
        if pdf_path and os.path.exists(pdf_path):
            try: os.remove(pdf_path)
            except OSError as e: print(f"Error removing temp file {pdf_path}: {e}")


# --- START: OpenAI Specific Functions ---
def extract_pdf_content_openai(pdf_url, output_path="./output_openai"):
    print(f"--- (Internal) OpenAI Extractor: Downloading {pdf_url} ---")
    os.makedirs(output_path, exist_ok=True); pdf_path = None
    try:
        response = requests.get(pdf_url); response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf.write(response.content); tmp_pdf.flush(); pdf_path = tmp_pdf.name
        print(f"--- (Internal) OpenAI Extractor: Partitioning {pdf_path} ---")
        chunks = partition_pdf(filename=pdf_path, infer_table_structure=True, strategy="hi_res", extract_image_block_types=["Image", "Table"], extract_image_block_to_payload=True, chunking_strategy="by_title", max_characters=10000)
        texts, images_base64, tables_html = [], [], []
        for chunk in chunks:
            if "Table" in str(type(chunk)): tables_html.append(getattr(chunk.metadata, 'text_as_html', chunk.text))
            elif "CompositeElement" in str(type(chunk)):
                texts.append(chunk.text)
                if hasattr(chunk.metadata, 'orig_elements'):
                    for el in chunk.metadata.orig_elements:
                        if "Image" in str(type(el)) and hasattr(el.metadata, 'image_base64'): images_base64.append(el.metadata.image_base64)
            elif "Image" in str(type(chunk)) and hasattr(chunk.metadata, 'image_base64'): images_base64.append(chunk.metadata.image_base64)
        images_base64 = list(set(images_base64))
        print(f"--- (Internal) OpenAI Extractor: Extracted {len(texts)} texts, {len(tables_html)} tables, {len(images_base64)} images ---")
        return texts, tables_html, images_base64, [] 
    except Exception as e: print(f"--- (Internal) OpenAI Extractor Error: {e} ---"); return [], [], [], []
    finally:
        if pdf_path and os.path.exists(pdf_path):
            try: os.remove(pdf_path)
            except OSError as e: print(f"Error removing temp file {pdf_path}: {e}")

def summarize_elements_openai(texts, tables, openai_api_key, model_name="gpt-4o"):
    if not openai_api_key: return [f"Err" for t in texts], [f"Err" for t in tables]
    print(f"--- (Internal) OpenAI Summarizer: {len(texts)} texts, {len(tables)} tables ---")
    prompt = ChatPromptTemplate.from_template("Summarize: {element}")
    try:
        model = ChatOpenAI(temperature=0, model=model_name, openai_api_key=openai_api_key, max_retries=1)
        chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        txt_s = [chain.invoke(t) for t in texts]
        tbl_s = [chain.invoke(t) for t in tables]
        return txt_s, tbl_s
    except Exception as e: print(f"--- (Internal) OpenAI Summarizer Error: {e} ---"); return [f"Err" for t in texts], [f"Err" for t in tables]

async def summarize_images_openai(images_base64, openai_api_key, model_name="gpt-4o-mini"):
    if not images_base64 or not openai_api_key: return ["Err"] * len(images_base64)
    print(f"--- (Internal) OpenAI Image Summarizer: {len(images_base64)} images ---")
    throttler = Throttler(rate_limit=5, period=1.0)
    tasks = [describe_image_openai_async(img, openai_api_key, model_name, throttler) for img in images_base64]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [res if not isinstance(res, Exception) else "Err" for res in results]

async def describe_image_openai_async(image_base64: str, openai_api_key: str, model="gpt-4o-mini", throttler=None) -> str:
    if not image_base64 or not openai_api_key: return "Err"
    prompt_tmpl = "Describe image."
    async def _call():
        try:
            prompt = ChatPromptTemplate.from_messages([("user", [{"type": "text", "text": prompt_tmpl}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}])])
            chain = prompt | ChatOpenAI(model=model, max_retries=1, openai_api_key=openai_api_key) | StrOutputParser()
            return await chain.ainvoke({})
        except Exception: return "Err"
    if throttler:
        async with throttler: return await _call()
    else: return await _call()
# --- END: OpenAI Specific Functions ---


# --- Generic Storage Function  ---
def store_elements_to_vector_and_docstore(
    paper_meta,
    texts, tables_content, images_base64, 
    text_summaries, table_summaries, image_summaries, 
    vectorstore, 
    docstore, 
    id_key="doc_id"
):
    """ Stores original content in docstore and summaries in the vectorstore. """
    print(f"--- Storage: Storing data for {paper_meta['title']} ---")
    def _store(originals, summaries, type_):
        if not originals: print(f"--- Storage: No {type_} to store."); return 0
        safe_summaries = summaries if len(summaries)==len(originals) else ["Err"]*len(originals)
        ids = [str(uuid.uuid4()) for _ in originals]
        try:
            docs = [Document(page_content=safe_summaries[i], metadata={id_key: ids[i], "type": type_, "title": paper_meta['title']}) for i in range(len(originals)) if safe_summaries[i] != "Err"]
            if docs: vectorstore.add_documents(docs)
            docstore.mset(list(zip(ids, originals))) 
            print(f"--- Storage: Stored {len(ids)} {type_} elements.")
            return len(ids)
        except Exception as e: print(f"--- Storage Error ({type_}): {e} ---"); return 0

    count = _store(texts, text_summaries, "text")
    count += _store(tables_content, table_summaries, "table")
    count += _store(images_base64, image_summaries, "image") 
    print(f"--- Storage: Finished. Total stored: {count} ---")
    return count > 0


# --- Unified Processing Function (Internal Use / Called by Wrappers) ---
def process_and_store_single_paper(
    paper_meta,
    processing_method, 
    qdrant_client,
    vectorstore_instance, 
    docstore_instance,    
    collection_name,
    embeddings_function,
    embedding_dim,
    id_key="doc_id"
):
    """ Processes paper using specified method and stores results. Returns success, new_docstore """
    print(f"--- Unified Processor: Starting for: {paper_meta['title']} | Method: {processing_method} ---")
    try:
        print(f"Recreating Qdrant collection '{collection_name}' (Dim: {embedding_dim})...")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
        )
        print("Re-initializing Vectorstore and creating new Docstore...")
        vectorstore_instance.__init__(client=qdrant_client, collection_name=collection_name, embeddings=embeddings_function)
        current_docstore = InMemoryStore() 

        texts, tables_content, images_base64_list = [], [], []
        text_summaries, table_summaries, image_summaries = [], [], []
        success = False

        if processing_method == 'gemini':
            if not GOOGLE_API_KEY: print("Error: Gemini key missing."); return False, current_docstore
            texts, tables_content, images_meta, text_summaries, table_summaries, image_summaries = process_pdf_gemini(paper_meta)
            images_base64_list = [img['base64'] for img in images_meta] 

        elif processing_method == 'openai':
            if not OPENAI_API_KEY: print("Error: OpenAI key missing."); return False, current_docstore
            texts, tables_content, images_base64_list, _ = extract_pdf_content_openai(paper_meta['pdf_link'])
            if texts or tables_content or images_base64_list:
                 text_summaries, table_summaries = summarize_elements_openai(texts, tables_content, OPENAI_API_KEY)
                 image_summaries = asyncio.run(summarize_images_openai(images_base64_list, OPENAI_API_KEY))

        else:
            print(f"Error: Unknown processing method '{processing_method}'")
            return False, current_docstore

        if texts or tables_content or images_base64_list:
             success = store_elements_to_vector_and_docstore(
                 paper_meta, texts, tables_content, images_base64_list,
                 text_summaries, table_summaries, image_summaries,
                 vectorstore_instance, current_docstore, id_key 
             )
        else:
             print(f"--- Unified Processor ({processing_method}): No content to store. ---"); success = False

        print(f"--- Unified Processor Finished: {paper_meta['title']} ({processing_method}). Success: {success} ---")
        return success, current_docstore 

    except Exception as e:
        print(f"Critical error during unified processing for {paper_meta['title']} ({processing_method}): {e}")
        import traceback; traceback.print_exc()
        return False, locals().get('current_docstore', docstore_instance)


# --- Gemini Specific Processing Wrapper ---
def process_and_store_gemini(
    paper_meta,
    qdrant_client,
    vectorstore_instance, 
    docstore_instance,    
    collection_name,      
    embeddings_function,  
    embedding_dim,
    id_key="doc_id"
):
    """ Wraps the unified processor for the Gemini path. Returns success, new_docstore """
    print(f"--- Calling unified processor for GEMINI method ---")
    return process_and_store_single_paper(
        paper_meta, 'gemini', qdrant_client, vectorstore_instance, docstore_instance,
        collection_name, embeddings_function, embedding_dim, id_key
    )