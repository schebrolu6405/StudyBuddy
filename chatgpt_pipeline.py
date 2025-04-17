# chatgpt_pipeline.py

import os
import re
import requests
import tempfile
import fitz 
import base64
import asyncio
from asyncio_throttle import Throttler
import nest_asyncio
import uuid
from tqdm import tqdm 

# --- Langchain / OpenAI ---
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore

# --- Unstructured ---
from unstructured.partition.pdf import partition_pdf

# --- Qdrant ---
from qdrant_client import QdrantClient, models

# --- Logging ---
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- IMPORT Generic Storage Function from arxiv_call ---
try:
    from arxiv_call import store_elements_to_vector_and_docstore
    print("Successfully imported storage function from arxiv_call.")
except ImportError:
    print("ERROR: Failed to import storage function from arxiv_call. Check file paths and circular dependencies.")
    def store_elements_to_vector_and_docstore(*args, **kwargs):
        print("CRITICAL ERROR: Using dummy storage function. Import failed.")
        return False


nest_asyncio.apply()

# --- OpenAI Extraction Function ---
def extract_pdf_content_openai(pdf_url, output_path="./output_openai"):
    """Downloads PDF, extracts text, tables (as HTML), images (base64)."""
    print(f"--- OpenAI Pipeline: Downloading {pdf_url} ---")
    os.makedirs(output_path, exist_ok=True); pdf_path = None
    try:
        response = requests.get(pdf_url); response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf.write(response.content); tmp_pdf.flush(); pdf_path = tmp_pdf.name

        print(f"--- OpenAI Pipeline: Partitioning PDF {pdf_path} ---")
        chunks = partition_pdf(
            filename=pdf_path, infer_table_structure=True, strategy="hi_res",
            extract_image_block_types=["Image", "Table"], image_output_dir_path=output_path,
            extract_image_block_to_payload=True, chunking_strategy="by_title", max_characters=10000,
            combine_text_under_n_chars=2000, new_after_n_chars=6000, include_page_breaks=True
        )
        print(f"--- OpenAI Pipeline: Partitioned into {len(chunks)} chunks ---")

        texts, images_base64, tables_html = [], [], []
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                 tables_html.append(getattr(chunk.metadata, 'text_as_html', chunk.text))
            elif "CompositeElement" in str(type(chunk)):
                texts.append(chunk.text)
                if hasattr(chunk.metadata, 'orig_elements'):
                    for el in chunk.metadata.orig_elements:
                        if "Image" in str(type(el)) and hasattr(el.metadata, 'image_base64'):
                            images_base64.append(el.metadata.image_base64)
            elif "Image" in str(type(chunk)) and hasattr(chunk.metadata, 'image_base64'):
                 images_base64.append(chunk.metadata.image_base64)

        images_base64 = list(set(images_base64)) 
        print(f"--- OpenAI Pipeline: Extracted {len(texts)} texts, {len(tables_html)} tables, {len(images_base64)} images ---")
        return texts, tables_html, images_base64

    except Exception as e:
        print(f"--- OpenAI Pipeline: Error processing PDF {pdf_url}: {e} ---")
        import traceback; traceback.print_exc(); return [], [], []
    finally:
        if pdf_path and os.path.exists(pdf_path):
            try: os.remove(pdf_path); print(f"--- OpenAI Pipeline: Removed temp file {pdf_path} ---")
            except OSError as e: print(f"--- OpenAI Pipeline: Error removing temp file {pdf_path}: {e} ---")

# --- OpenAI Summarization Functions ---
def summarize_elements_openai(texts, tables, openai_api_key, model_name="gpt-4o"):
    """Summarizes text and table elements using OpenAI (Sequential)."""
    if not openai_api_key:
        print("--- OpenAI Pipeline: Warning - API Key missing for text/table summary.")
        return [f"Summ Err Key" for _ in texts], [f"Summ Err Key" for _ in tables]
    print(f"--- OpenAI Pipeline: Summarizing {len(texts)} texts, {len(tables)} tables using {model_name} (Seq) ---")
    prompt_text = """Summarize this text or table chunk from a research paper concisely: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    try:
        model = ChatOpenAI(temperature=0, model=model_name, openai_api_key=openai_api_key, max_retries=1)
        chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    except Exception as e:
        print(f"--- OpenAI Pipeline: Error initializing OpenAI model ({model_name}): {e} ---")
        return [f"Summ Err Init" for _ in texts], [f"Summ Err Init" for _ in tables]

    text_summaries = [chain.invoke(t) if t else "Empty chunk" for t in tqdm(texts, desc="OpenAI Text Summ")]
    table_summaries = [chain.invoke(tb) if tb else "Empty chunk" for tb in tqdm(tables, desc="OpenAI Table Summ")]

    print(f"--- OpenAI Pipeline: Finished summarizing texts and tables ---")
    return text_summaries, table_summaries

async def describe_image_openai_async(image_base64: str, openai_api_key: str, model="gpt-4o-mini", throttler=None) -> str:
    """Describes a single image using OpenAI vision model with throttling."""
    if not openai_api_key: return "Summ Err Key"
    if not image_base64: return "Empty Image"
    prompt_template = "Describe this image from a research paper. Focus on diagrams, plots, architectures, or tables."
    async def _call():
        try:
            prompt = ChatPromptTemplate.from_messages([("user", [{"type": "text", "text": prompt_template}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}])])
            chain = prompt | ChatOpenAI(model=model, max_retries=2, openai_api_key=openai_api_key) | StrOutputParser()
            return await chain.ainvoke({})
        except Exception as e: print(f"Img Summ Err: {e}"); return "Summ Err API"
    if throttler:
        async with throttler: return await _call()
    else: return await _call()

async def summarize_images_openai(images_base64, openai_api_key, model_name="gpt-4o-mini", rate_limit=5, period=1.0):
    """Summarizes a list of base64 images asynchronously using OpenAI."""
    if not images_base64: return []
    if not openai_api_key: print("--- OpenAI Pipeline: Warning - API Key missing for image summary."); return ["Summ Err Key"] * len(images_base64)
    print(f"--- OpenAI Pipeline: Summarizing {len(images_base64)} images using {model_name} ---")
    throttler = Throttler(rate_limit=rate_limit, period=period)
    tasks = [describe_image_openai_async(img, openai_api_key, model_name, throttler) for img in images_base64]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    final = [res if not isinstance(res, Exception) else "Summ Err Async" for res in results]
    print(f"--- OpenAI Pipeline: Finished summarizing images ---")
    return final


# --- Main OpenAI Processing Function ---
def process_and_store_openai(
    paper_meta,
    qdrant_client,          
    vectorstore_instance,  
    docstore_instance,      
    collection_name,        
    embeddings_function,    
    embedding_dim,         
    openai_api_key,         
    id_key="doc_id"
):
    """ Processes paper using OpenAI pipeline, stores results via imported function. Returns success, new_docstore """
    print(f"--- Starting OpenAI Pipeline for: {paper_meta['title']} ---")
    print(f"--- Collection: {collection_name} | Embedding Dim: {embedding_dim} ---")
    if not openai_api_key: print("--- OpenAI Pipeline: Error - API Key missing."); return False, docstore_instance

    try:
        print(f"Recreating Qdrant collection '{collection_name}'...")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
        )
        print("Re-initializing Vectorstore and creating new Docstore...")
        vectorstore_instance.__init__(client=qdrant_client, collection_name=collection_name, embeddings=embeddings_function)
        current_docstore = InMemoryStore() 
        print("Vectorstore re-initialized. New empty Docstore created.")

        texts, tables_html, images_base64 = extract_pdf_content_openai(paper_meta['pdf_link'])
        if not (texts or tables_html or images_base64):
            print("--- OpenAI Pipeline: No content extracted."); return False, current_docstore

        text_summaries, table_summaries = summarize_elements_openai(texts, tables_html, openai_api_key)
        image_summaries = asyncio.run(summarize_images_openai(images_base64, openai_api_key))

        print(f"--- OpenAI Pipeline: Calling generic storage function... ---")
        success = store_elements_to_vector_and_docstore(
            paper_meta, texts, tables_html, images_base64, 
            text_summaries, table_summaries, image_summaries,
            vectorstore_instance, current_docstore, id_key 
        )

        print(f"--- Finished OpenAI Pipeline for '{paper_meta['title']}'. Success: {success} ---")
        return success, current_docstore 

    except Exception as e:
        print(f"--- OpenAI Pipeline: Critical error for {paper_meta['title']}: {e} ---")
        import traceback; traceback.print_exc()
        return False, locals().get('current_docstore', docstore_instance)