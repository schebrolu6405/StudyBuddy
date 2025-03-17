import os
import io
import re
import requests
import arxiv
from PyPDF2 import PdfReader
from tqdm import tqdm
from fuzzywuzzy import fuzz
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
import google.generativeai as gai
gai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

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

def fetch_and_store_arxiv_papers(query, num_papers, save_path="faiss_index"):
    print(f"Searching ArXiv for: {query}")
    search_query = f"all:{query}"
    search = arxiv.Search(
        query=search_query,
        max_results=num_papers * 2,
        sort_by=arxiv.SortCriterion.Relevance
    )
    client = arxiv.Client()
    results = list(client.results(search))

    relevant_papers = [res for res in results if is_relevant_paper(res.title, res.summary, query)]
    if not relevant_papers:
        print(f"No relevant papers found for: {query}")
        return False

    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1500, chunk_overlap=200, length_function=len)
    all_chunks = []

    print("Downloading and processing papers...")
    for result in tqdm(relevant_papers[:num_papers], desc="Processing"):
        try:
            response = requests.get(result.pdf_url, stream=True)
            if response.status_code == 200:
                pdf_stream = io.BytesIO(response.content)
                reader = PdfReader(pdf_stream)
                raw_text = ''
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text
                chunks = text_splitter.split_text(raw_text)
                all_chunks.extend(chunks)
            else:
                print(f"Failed to download PDF for: {result.title}")
        except Exception as e:
            print(f"Skipping {result.title} due to error: {e}")

    if not all_chunks:
        print("No text extracted.")
        return False

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)
    vector_store.save_local(save_path)
    print(f"Stored {len(all_chunks)} chunks to FAISS at '{save_path}'")
    return True
