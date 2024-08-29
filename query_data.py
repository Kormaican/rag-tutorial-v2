import argparse
from pprint import pprint

from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
from csv import writer

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    #query_rag(query_text)

    query_search_only(query_text)


def select_text_between(text, start, end):
    """
    This function selects text between two strings.

    :param text: The input text
    :param start: The starting string
    :param end: The ending string
    :return: The text between the starting and ending strings
    """
    try:
        # Find the start and end indices
        start_idx = text.index(start) + len(start)
        end_idx = text.index(end, start_idx)

        # Return the text between the start and end indices
        return text[start_idx:end_idx]
    except ValueError:
        # If start or end string is not found, return an empty string
        return ""

def query_search_only(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB. Results is a list of Tuples, (document, score). Lower score means more similar.
    # document is an object with page_content and metadata
    results = db.similarity_search_with_score(query_text, k=5)

    count = 1
    for result in results:
        filepath_str = select_text_between(result[0].page_content, "REVIT FILE PATH: ", "\n")
        print(f"{count}. Filepath: \n{filepath_str}\n   Score:{result[1]}")

    # with open("results.csv", 'w', newline='') as file:
    #     writer(file).writerows(results)

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)
    # # print(prompt)
    #
    # model = Ollama(model="mistral")
    # response_text = model.invoke(prompt)
    #
    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    # return response_text

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
