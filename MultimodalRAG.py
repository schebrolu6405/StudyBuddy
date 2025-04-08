import os
import re
import io
import base64
import requests
import arxiv
import dotenv
from tqdm import tqdm
from PIL import Image
from fuzzywuzzy import fuzz
from IPython.display import display, Image as IPImage
from unstructured.partition.pdf import partition_pdf
import google.generativeai as genai

from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import uuid

# Load API Key from .env
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model_text = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
model_vision = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(collection_name="multimodal_rag_chunks", embedding_function=embedding_function, persist_directory="./chroma_db")
docstore = InMemoryStore()
id_key = "doc_id"
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=docstore, id_key=id_key)

def sanitize_filename(title):
    return re.sub(r'[<>:"/\\|?*]', '_', title)

def is_relevant_paper(title, abstract, keyword, threshold=50):
    keyword_words = keyword.lower().split()
    title_words = title.lower().split()
    abstract_words = abstract.lower().split()
    match_count = sum(1 for word in keyword_words if word in title_words or word in abstract_words)
    fuzzy_score = max(
        fuzz.partial_ratio(keyword.lower(), title.lower()),
        fuzz.partial_ratio(keyword.lower(), abstract.lower())
    )
    return match_count / len(keyword_words) > 0.5 or fuzzy_score > threshold

def extract_tagged_sections(text_blocks, tag_prefix):
    pattern = re.compile(rf"{tag_prefix}\s*\d+", re.IGNORECASE)
    return [block.strip() for block in text_blocks if pattern.search(block)]

def get_images_grouped_by_caption(chunks):
    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    caption = el.metadata.text_as_html or ""
                    images.append({
                        "caption": caption.strip(),
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
    prompt = f"""You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary.

Text or Table:
{text}
"""
    response = model_text.generate_content(prompt)
    return response.text.strip()

def summarize_image_base64(base64_code, context_text=""):
    image_data = base64.b64decode(base64_code)
    img = Image.open(io.BytesIO(image_data))
    prompt = f"""Describe the image in detail. For context:
The image is part of a research paper. Be specific about graphs, charts, or visual content.

{context_text}
"""
    response = model_vision.generate_content([prompt, img])
    return response.text.strip()

def fetch_and_summarize_arxiv_papers(query, num_papers=2, output_path="./content/"):
    os.makedirs(output_path, exist_ok=True)
    search = arxiv.Search(query=f"all:{query}", max_results=num_papers * 2, sort_by=arxiv.SortCriterion.Relevance)
    results = list(arxiv.Client().results(search))
    relevant_papers = [r for r in results if is_relevant_paper(r.title, r.summary, query)]
    if not relevant_papers:
        return False

    all_texts = []
    all_tables = []
    all_images = []
    tagged_fig_blocks = []
    tagged_table_blocks = []
    text_summaries = []
    image_summaries = []
    paper_summaries = []

    for result in tqdm(relevant_papers[:num_papers], desc="Processing"):
        try:
            pdf_url = result.pdf_url
            response = requests.get(pdf_url, stream=True)
            if response.status_code != 200:
                continue

            title = result.title.strip()
            abstract = result.summary.strip()
            paper_summaries.append((title, abstract))
            filename = os.path.join(output_path, sanitize_filename(title) + ".pdf")

            with open(filename, "wb") as f:
                f.write(response.content)

            chunks = partition_pdf(
                filename=filename,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000
            )

            texts = [chunk.text for chunk in chunks if "CompositeElement" in str(type(chunk)) and chunk.text]
            tables = [chunk.text for chunk in chunks if "Table" in str(type(chunk))]
            images = get_images_grouped_by_caption(chunks)

            all_texts.extend(texts)
            all_tables.extend(tables)
            all_images.extend(images)

            tagged_fig_blocks.extend(extract_tagged_sections(texts, "Fig"))
            tagged_table_blocks.extend(extract_tagged_sections(texts, "Table"))

        except Exception as e:
            continue

    # === Summarize all text/table chunks
    summary_inputs = all_texts + all_tables + tagged_table_blocks
    text_summaries = [summarize_text_element(text) for text in summary_inputs]

    # === Summarize image content
    for img in all_images:
        caption = img.get("caption", "").strip()
        context = caption
        for block in tagged_fig_blocks:
            if "figure" in block.lower() or "fig" in block.lower():
                context = block
                break
        summary = summarize_image_base64(img["base64"], context)
        image_summaries.append(summary)

    text_ids = [str(uuid.uuid4()) for _ in all_texts]
    text_docs = [Document(page_content=summary, metadata={id_key: text_ids[i]}) for i, summary in enumerate(text_summaries[:len(all_texts)])]
    retriever.vectorstore.add_documents(text_docs)
    retriever.docstore.mset(list(zip(text_ids, all_texts)))

    table_data = all_tables + tagged_table_blocks
    table_ids = [str(uuid.uuid4()) for _ in table_data]
    table_docs = [Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(text_summaries[len(all_texts):len(all_texts)+len(table_data)])]
    retriever.vectorstore.add_documents(table_docs)
    retriever.docstore.mset(list(zip(table_ids, table_data)))

    image_ids = [str(uuid.uuid4()) for _ in all_images]
    image_docs = [Document(page_content=summary, metadata={id_key: image_ids[i]}) for i, summary in enumerate(image_summaries)]
    retriever.vectorstore.add_documents(image_docs)
    retriever.docstore.mset(list(zip(image_ids, all_images)))

    return {
    "titles": [res.title for res in relevant_papers],
    "abstracts": [res.summary for res in relevant_papers]}
