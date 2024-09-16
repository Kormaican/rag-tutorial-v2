import argparse
import requests
import os
import shutil
from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from bs4 import BeautifulSoup
import nest_asyncio
import logging
# import requests
import gzip
import xml.etree.ElementTree as ET
from io import BytesIO
# from langchain.document_loaders import SitemapLoader

nest_asyncio.apply()


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    add_to_chroma(documents)

    # chunks = split_documents(documents)
    # add_to_chroma(chunks)
    # add_to_chroma(documents)


# def load_documents():
#     sitemap_url = 'https://help.autodesk.com/sitemaps-index.xml'
#
#     # Retrieve the sitemap index
#     response = requests.get(sitemap_url)
#     sitemap_index = ET.fromstring(response.content)
#
#     # Extract URLs of the .xml.gz files
#     sitemap_urls = [elem.text for elem in
#                     sitemap_index.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap/{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
#
#     # Download and extract each .xml.gz file
#     for sitemap_url in sitemap_urls:
#         response = requests.get(sitemap_url)
#         with gzip.open(BytesIO(response.content), 'rb') as f:
#             xml_content = f.read()
#
#         # Parse the XML content
#         sitemap = ET.fromstring(xml_content)
#
#         # Run the SitemapLoader for each sitemap
#         loader = SitemapLoader(
#             web_path=sitemap_url,
#             filter_urls=["https://api.python.langchain.com/en/latest"]
#         )
#
#         documents = loader.load()
#         # Process the documents as needed
#
#     return documents

def load_documents():
    loader = SitemapLoader(
            web_path='temp_sitemap.xml',
            # filter_urls=["https://help.autodesk.com/view/RVT/2025/ENU/?guid=GUID-"],
            # blocknum=10,
            parsing_function=remove_nav_and_header_elements,
            is_local=True
        )

    return loader.load()

def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")

    # Remove each 'nav' and 'header' element from the BeautifulSoup object
    for element in nav_elements + header_elements:
        element.decompose()

    return str(content.get_text())

# def load_documents():
#     sitemap_url = 'https://help.autodesk.com/sitemaps-index.xml'
#
#     # Retrieve the sitemap index
#     response = requests.get(sitemap_url)
#     sitemap_index = response.content
#
#     # Extract URLs of the .xml.gz files
#     sitemap_urls = [elem.text for elem in
#                     ET.fromstring(sitemap_index).findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap/{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
#
#     documents = []
#
#     # Download and extract each .xml.gz file
#     for sitemap_url in sitemap_urls:
#         response = requests.get(sitemap_url)
#         with gzip.open(BytesIO(response.content), 'rb') as f:
#             xml_content = f.read()
#
#         # Save the extracted XML content to a temporary file
#         with open('temp_sitemap.xml', 'wb') as temp_file:
#             temp_file.write(xml_content)
#
#         # Run the SitemapLoader for the temporary XML file
#
#         loader = SitemapLoader(
#             web_path='temp_sitemap.xml',
#             filter_urls=["https://help.autodesk.com/view/RVT/2025/ENU/?guid=GUID-"],
#             blocknum=10,
#             is_local=True
#         )
#
#         documents.extend(loader.load())

    # return documents

# def load_documents():
#     sitemap_url = 'https://help.autodesk.com/sitemaps-index.xml'
#
#     # TODO - retreive the .xml.gz sitemap files from the sitemap-url.
#     #  - extract the contents so it is in .xml format
#     #  - run the SitemapLoader for all sitemap .xml files
#
#     # loader = SitemapLoader(
#     #     web_path="https://api.python.langchain.com/sitemap.xml",
#     #     filter_urls=["https://api.python.langchain.com/en/latest"]
#     # )
#
#     documents = loader.load()


# def fetch_sitemap(url):
#     headers = {'User-Agent': os.getenv('USER_AGENT', 'Mozilla/5.0')}
#     response = requests.get(url, headers=headers)
#     if response.headers.get('Content-Encoding') == 'gzip':
#         return gzip.decompress(response.content)
#     return response.content

# def load_documents():
#     sitemap_url = "https://help.autodesk.com/sitemaps-index.xml"
#     response = requests.get(sitemap_url)
#     soup = BeautifulSoup(response.content, "xml")
#
#     urls = [loc.text for loc in soup.find_all("loc")]
#
#     documents = []
#     for url in urls:
#         try:
#             page_response = requests.get(url)
#             page_response.encoding = 'utf-8'  # Set encoding to UTF-8
#             page_soup = BeautifulSoup(page_response.text, "html.parser")
#             page_text = page_soup.get_text()
#             documents.append(Document(page_content=page_text, metadata={"source": url}))
#         except Exception as e:
#             print(f"Error processing {url}: {e}")
#
#     return documents


# def load_documents():
#     # document_loader = CSVLoader(DATA_PATH)
#
#     # load a single CSV, for my specific use-case
#     document_loader = CSVLoader(file_path='data/In-house script directory-wip 09.13.24.csv')
#
#     return document_loader.load()

# def load_documents():
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":


    main()
