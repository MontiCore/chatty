from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json
import uuid
from tqdm import tqdm


def get_embedding_text(chunk):
    text = f"{chunk['document_group'].capitalize()}: {chunk['document_title']}"
    if chunk["authors"] and len(chunk["authors"]) > 0:
        text += f", Authors: {', '.join(chunk['authors'])}"
    if chunk["publish_date"]:
        text += f", Published: {chunk['publish_date']}"
    if chunk["chapter"] and chunk["chapter"] is not None:
        text += f", Chapter: {chunk['chapter']}"
    # if chunk["subchapter"] and chunk["subchapter"] is not None:
    #     text += f", Subchapter: {chunk['subchapter']}"
    # if chunk["title"] and chunk["title"] is not None:
    #     text += f", Title: {chunk['title']}"
    if chunk["section"] and chunk["section"] is not None:
        text += f", Section: {chunk['section']}"
    if chunk["subsection"]:
        text += f", Subsection: {chunk['subsection']}"
    if chunk["subsubsection"]:
        text += f", Subsubsection: {chunk['subsubsection']}"
    text += f"\n\nContent:\n{chunk['body']}"
    return text


def embed_chunks(chunks):
    load_dotenv("../src/.env")
    embedding_model = "text-embedding-3-large"
    embedding_client = AzureOpenAI(azure_endpoint=os.getenv("SWEDEN_OPENAI_ENDPOINT"),
                                   azure_deployment=embedding_model,
                                   api_version='2023-05-15',
                                   api_key=os.getenv("SWEDEN_OPENAI_KEY"))
    text_list = [get_embedding_text(c) for c in chunks]
    # TODO: Use batch API (half cost)
    # Embeds all entries at once
    res = embedding_client.embeddings.create(input=text_list, model=embedding_model)
    for i, r in enumerate(res.data):
        chunks[i]["_additional"] = {}
        chunks[i]["_additional"]["vector"] = r.embedding
        chunks[i]["_additional"]["id"] = str(uuid.uuid4())
    return chunks


if __name__ == "__main__":
    chunk_path = "../data/latex/processed"
    embedding_path = "../data/latex/embedded"
    with open("latex_meta.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata = [metadata[1]]

    for pub in tqdm(metadata, desc="Embedding publications"):
        file_name = pub['source'].split('/')[0]
        with open(f"{chunk_path}/{file_name}.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        chunks = embed_chunks(chunks)
        # TODO: Automatically upload to sciebo?
        with open(f"{embedding_path}/{file_name}.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)


