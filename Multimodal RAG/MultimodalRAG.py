import os
import re
import io
import base64
import requests
import tempfile
import uuid
import dotenv
import fitz  
from PIL import Image
from IPython.display import display, Image as IPImage, HTML
from unstructured.partition.pdf import partition_pdf
import google.generativeai as genai
from collections import defaultdict

from langchain.storage import InMemoryStore
from langchain.vectorstores import Qdrant
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
from qdrant_client.http import models


dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model_text = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
model_vision = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

qdrant_client = QdrantClient(
    url="https://3da9da71-c9fd-4e77-a222-5e9fee093b8a.us-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key=QDRANT_API_KEY,
)

collection_name = "multimodal_rag_chunks"

qdrant_client.delete_collection(collection_name=collection_name)
collection_config = models.VectorParams(size=768, distance=Distance.COSINE)
qdrant_client.create_collection(collection_name=collection_name, vectors_config=collection_config)

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding_function,
)

docstore = InMemoryStore()
id_key = "doc_id"
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=docstore, id_key=id_key)


def get_images(chunks):
    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images.append({
                        "base64": el.metadata.image_base64
                    })
    return images

def display_base64_image(base64_code):
    image_data = base64.b64decode(base64_code)
    try:
        display(IPImage(data=image_data))
    except Exception:
        with open("temp_image.jpg", "wb") as f:
            f.write(image_data)
        print("Image saved to 'temp_image.jpg'")

def summarize_text_element(text):
    prompt_text = f"""You are an assistant tasked with summarizing tables and text. Give a concise summary of the table or text. Respond only with the summary, no additionnal comment. Do not start your message by saying "Here is a summary" or anything like that. Just give the summary as it is.
    Table or text chunk: {text} """
    response = model_text.generate_content(prompt_text)
    return response.text.strip()

def summarize_image_base64(base64_code):
    image_data = base64.b64decode(base64_code)
    img = Image.open(io.BytesIO(image_data))
    prompt = f"""You are a research assistant tasked with analyzing an image extracted from a scientific research paper. Please describe the image in detail. If the image contains a workflow diagram, model architecture, experimental setup, or data flow, explain each component clearly and concisely. 
    Focus on the structure, flow, and relationships between elements. If the image includes any labels, blocks, arrows, or layers, describe their roles and how they connect. Avoid vague descriptions and assume the image holds key information related to the research methodology or findings."""
    response = model_vision.generate_content([prompt, img])
    return response.text.strip()

def summarize_table_html(html, caption=None):
    if caption:
        prompt = f"You are a scientific research assistant. Below is a table extracted from a research paper.\nCaption: \"{caption}\"\nPlease summarize the table's key contents and what it reveals."
    else:
        prompt = "You are a research assistant. Summarize the following table from a scientific paper."
    response = model_text.generate_content([prompt, html])
    return response.text.strip()

def process_pdf_url(pdf_url, output_path="./output"):
    os.makedirs(output_path, exist_ok=True)
    response = requests.get(pdf_url)
    response.raise_for_status()  # Raise error for bad responses

    # Save the PDF temporarily
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        tmp_pdf.write(response.content)
        tmp_pdf.flush()
        pdf_path = tmp_pdf.name

    # Extract figure captions using PyMuPDF
    doc = fitz.open(pdf_path)
    captions = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if re.match(r'^Figure\s*\d+', text, re.IGNORECASE):
                captions.append((page_num, text))
    tablecaptions = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if re.match(r'^Table\s*\d+', text, re.IGNORECASE):
                tablecaptions.append((page_num, text))

    grouped = defaultdict(list)
    for _, text in tablecaptions:
        match = re.match(r"^(Table \d+):", text)
        if match:
            key = match.group(1)
            grouped[key].append(text)

    combined_captions = {
        table: " ".join(lines) for table, lines in grouped.items()
    }
    combined_table_captions = list(combined_captions.values())

    chunks = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        image_output_dir_path=output_path,
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts = [chunk.text for chunk in chunks if "CompositeElement" in str(type(chunk)) and chunk.text]
    images = get_images(chunks)

    raw_tables = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res"
    )
    tables = [tab for tab in raw_tables if tab.category == "Table" and hasattr(tab.metadata, "text_as_html")]
    tables_html = [tab.metadata.text_as_html for tab in tables]

    table_summaries = [
        summarize_table_html(tab.metadata.text_as_html, caption)
        for tab, caption in zip(tables, combined_table_captions)
    ]

    text_summaries = [summarize_text_element(text) for text in texts]
    image_summaries = []

    for i, img in enumerate(images):
        # Generate the summary using the image base64 and the default context
        summary = summarize_image_base64(img["base64"])
        image_summaries.append(summary)

    def store_data(items, summaries):
        ids = [str(uuid.uuid4()) for _ in items]
        retriever.vectorstore.add_documents([
            Document(page_content=summaries[i], metadata={id_key: ids[i]})
            for i in range(len(items))
        ])
        retriever.docstore.mset(list(zip(ids, items)))

    if texts:
        store_data(texts, text_summaries)
    if tables:
        store_data(tables_html, table_summaries)
    if images:
        store_data(images, image_summaries)

    return texts, text_summaries, images, captions, image_summaries, tables, combined_table_captions, table_summaries
    
def parse_docs(docs):
    b64, text = [], []
    for doc in docs:
        try:
            base64.b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    # Assemble the context text from parsed documents
    context_text = ""
    for text_element in docs_by_type.get("texts", []):
        context_text += text_element.text if hasattr(text_element, "text") else str(text_element)

    # Use the refined prompt template for research papers
    prompt_text = f"""
You are an expert AI assistant helping a user understand and summarize a scientific research paper. Use both text and image content to generate a clear, informative, and accurate answer.

Your response must be based only on the following context, which may include:
- Paragraphs of text
- Tables (data, results, or evaluations)
- Figures or diagrams (e.g. model architecture, workflows, experimental setups)

Context: {context_text}
Question: {user_question}

INSTRUCTIONS:
- Treat any image references (e.g., `[IMAGE REFERENCE: Figure 3 - ...]`) as key visual context and refer to them explicitly when answering.
- Summarize key insights from diagrams, workflows, or model architectures where relevant.
- Use table content to support results, comparisons, or experiments if mentioned.
- Refer to the Abstract and Conclusion (if available) to provide a concise summary of the overall contribution.
- Do not include external knowledge or citations unless specifically asked.
- Keep your response clear and technically sound.
"""

    # Compose prompt parts (text + images)
    prompt_parts = [{"text": prompt_text}]
    image_refs = []
    for idx, image_b64 in enumerate(docs_by_type.get("images", [])):
        try:
            image_bytes = base64.b64decode(image_b64)
            Image.open(io.BytesIO(image_bytes))  # validate image
            prompt_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                }
            })
            image_refs.append({"id": f"image_{idx+1}", "base64": image_b64})
        except Exception as e:
            print(f"Skipping invalid image {idx+1}: {e}")
            continue
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
    response = model.generate_content(contents=[{"role": "user", "parts": prompt_parts}])
    return {
        "text": response.text.strip(),
        "images": image_refs
    }

if __name__ == "__main__":
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    texts, text_summaries, images, captions, image_summaries, tables, combined_table_captions, table_summaries = process_pdf_url(url)
    chain_with_sources = {"context": retriever | RunnableLambda(parse_docs), "question": RunnablePassthrough(),} | RunnablePassthrough().assign(response=RunnableLambda(build_prompt))
    result = chain_with_sources.invoke("Explain about Embeddings and softmax function. What are the Maximum Path Length of each layer type?")
    print("\n\nResponse:")
    print(result["response"]["text"])
    print("\n\nImages:")
    for i, item in enumerate(result["context"]["texts"]):
        if isinstance(item, dict) and "base64" in item:
            print(f"Image {i+1}:")
            display_base64_image(item["base64"])