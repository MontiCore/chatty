import json

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import argparse
import weaviate.classes as wvc
import weaviate
from pathlib import Path
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from laser_encoders import LaserEncoderPipeline
from laser_encoders.laser_tokenizer import LaserTokenizer
from laser_encoders.models import SentenceEncoder
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import time
from sentence_transformers import SentenceTransformer, util
import csv


def retrieve_data(data="documents"):
    path = f"../data/{data}.csv"
    if os.path.exists(path):
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            writer = csv.reader(f)
            for row in writer:
                documents.append(row)
    else:
        try:
            client = weaviate.connect_to_local(port=8080, grpc_port=50051, host="localhost")
        except:
            client = weaviate.connect_to_local(port=8080, grpc_port=50051, host="weaviate")
        collection = client.collections.get("Documents" if data=="documents" else "Cd4a")
        documents = []
        for item in tqdm(collection.iterator(), desc="Retrieving"):
            if data == "documents":
                documents.append(item.properties["body"])
            else:
                documents.append((item.properties["body"], item.properties["comment"]))
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for d in documents:
                if data == "cd4a":
                    writer.writerow([d[0], d[1]])
                else:
                    writer.writerow([d])
    return documents

def get_sentence_embeddings(german_prompts, english_prompts, sentence_model="e5"):
    file_path = f"../data/sentence_embeddings_{sentence_model}.pkl"
    # !!! Always add "query: ", or "passage: " as prefix for Retrieval !!!
    if sentence_model == "e5":
        model = SentenceTransformer("intfloat/multilingual-e5-large")
    elif sentence_model == "e5-instruct":
        model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    elif sentence_model == "e5-mistral":
        model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    else:
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            db_embeddings = pickle.load(f)
    else:
        documents = retrieve_data(data="documents")
        if sentence_model in ["e5"]:
            documents = [f"passage: {d}" for d in documents]
        else:
            documents = [d[0] for d in documents]

        print(f"Encoding...")
        print(len(documents))
        db_embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True, device="cuda", batch_size=128)

        db_embeddings = np.array(db_embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(db_embeddings, f)

    if sentence_model in ["e5"]:
        german_prompts = [f"query: {p}" for p in german_prompts]
        english_prompts = [f"query: {p}" for p in english_prompts]

    print(f"Creating prompt embeddings")
    if sentence_model in ["e5_instruct", "e5-mistral"]:
        german_embeddings = np.array(model.encode(german_prompts, normalize_embeddings=True, prompt_name="sts_query"))
        english_embeddings = np.array(model.encode(english_prompts, normalize_embeddings=True, prompt_name="sts_query"))
    else:
        german_embeddings = np.array(model.encode(german_prompts, normalize_embeddings=True))
        english_embeddings = np.array(model.encode(english_prompts, normalize_embeddings=True))
    print(f"Dist embedding 0: {np.linalg.norm(english_embeddings[0]-german_embeddings[0])}\n"
          f"Dist embedding 0, with DB: {np.linalg.norm(english_embeddings[0]-db_embeddings[0])}")
    return db_embeddings, german_embeddings, english_embeddings

def get_ada_embeddings(german_prompts, english_prompts):
    file_path = "../data/adav2_embeddings.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            db_embeddings = pickle.load(f)
    else:
        try:
            client = weaviate.connect_to_local(port=8080, grpc_port=50051, host="localhost")
        except:
            client = weaviate.connect_to_local(port=8080, grpc_port=50051, host="weaviate")
        collection = client.collections.get("Documents")
        db_embeddings = []
        for item in tqdm(collection.iterator(include_vector=True)):
            vector = item.vector
            db_embeddings.append(vector)
        db_embeddings = np.array(db_embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(db_embeddings, f)

    embedding_model = "text-embedding-ada-002"
    openai_client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                azure_deployment=embedding_model,
                                api_version='2023-05-15',
                                api_key=os.getenv("AZURE_OPENAI_KEY"))
    german_embeddings = []
    english_embeddings = []
    for g_p in german_prompts:
        print(f"Embedding prompt pair...")
        response = openai_client.embeddings.create(input=g_p, model=embedding_model)
        time.sleep(1)
        german_embeddings.append(response.data[0].embedding)
    german_embeddings = np.array(german_embeddings)
    for e_p in english_prompts:
        response = openai_client.embeddings.create(input=e_p, model=embedding_model)
        time.sleep(1)
        english_embeddings.append(response.data[0].embedding)
    english_embeddings = np.array(english_embeddings)
    return db_embeddings, german_embeddings, english_embeddings

def get_laser_embeddings(german_prompts, english_prompts):
    file_path = "../data/laser_embeddings.pkl"
    model_path = "../data/laser"
    encoder = SentenceEncoder(model_path=f"{model_path}/laser2.pt", spm_vocab=f"{model_path}/laser2.cvocab", verbose=True, spm_model=f"{model_path}/laser2.spm")

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            db_embeddings = pickle.load(f)
    else:
        documents = retrieve_data("documents")
        print(f"Encoding...")
        db_embeddings = encoder.encode_sentences(documents, normalize_embeddings=True)

        # db_embeddings = encoder.encode_sentences(documents, normalize_embeddings=True)
        db_embeddings = np.array(db_embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(db_embeddings, f)
    print(f"Creating german embeddings")
    german_embeddings = np.array(encoder.encode_sentences(german_prompts, normalize_embeddings=True))
    print(f"Creating english embeddings")
    english_embeddings = np.array(encoder.encode_sentences(english_prompts, normalize_embeddings=True))
    print(f"Dist embedding 0: {np.linalg.norm(english_embeddings[0]-german_embeddings[0])}\n"
          f"Dist embedding 0, with DB: {np.linalg.norm(english_embeddings[0]-db_embeddings[0])}")
    return db_embeddings, german_embeddings, english_embeddings


def get_avg_cosine(german_embeddings, english_embeddings):
    total_cosine = 0
    for g_e, e_e in zip(german_embeddings, english_embeddings):
        dot_product = np.dot(g_e, e_e)
        # Compute the norms of the vectors
        norm_vector1 = np.linalg.norm(e_e)
        norm_vector2 = np.linalg.norm(g_e)
        # Compute the cosine similarity
        cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
        total_cosine += cosine_similarity
    avg_cosine = total_cosine/german_embeddings.shape[0]
    print(f"Average cosine similarity: {avg_cosine:.4f}")
    return avg_cosine

def get_avg_euclidean(german_embeddings, english_embeddings):
    total_euclidean = 0
    for g_e, e_e in zip(german_embeddings, english_embeddings):
        euclidean = np.linalg.norm(g_e-e_e)
        total_euclidean += euclidean
    avg_euclidean = total_euclidean / german_embeddings.shape[0]
    print(f"Average cosine similarity: {avg_euclidean:.4f}")
    return avg_euclidean

def eval(args):
    embedding_path = f"{args.path}/{args.embedding}_{args.sentence_model}" if args.embedding == "sentence" \
        else f"{args.path}/{args.embedding}"
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)

    load_dotenv("../src/.env")

    with open(f"{args.path}/english_prompts.json", "r", encoding="utf-8") as f:
        english_prompts = json.load(f)
    with open(f"{args.path}/german_prompts.json", "r", encoding="utf-8") as f:
        german_prompts = json.load(f)


    if args.embedding == "adav2":
        db_embeddings, german_embeddings, english_embeddings = get_ada_embeddings(german_prompts, english_prompts)
    elif args.embedding == "laser":
        db_embeddings, german_embeddings, english_embeddings = get_laser_embeddings(german_prompts, english_prompts)
    elif args.embedding == "sentence":
        db_embeddings, german_embeddings, english_embeddings = get_sentence_embeddings(german_prompts, english_prompts, sentence_model=args.sentence_model)
    avg_cosine = get_avg_cosine(german_embeddings, english_embeddings)
    avg_euclidean = get_avg_euclidean(german_embeddings, english_embeddings)
    with open(f"{embedding_path}/metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Avg Cosine: {avg_cosine:.4f}\n"
                f"Avg Euclidean: {avg_euclidean:.4f}")
    db_embeddings = db_embeddings[:5000]
    combined_embeddings = np.concatenate((db_embeddings, german_embeddings, english_embeddings), axis=0)

    print(f"Reducing with PCA")
    pca_embeddings = PCA(n_components=50).fit_transform(combined_embeddings)
    print(f"Applying {args.reduction} on array of shape {pca_embeddings.shape}")

    if args.reduction == "tsne":
        reduced_embeddings = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(pca_embeddings)
    elif args.reduction == "umap":
        reduced_embeddings = UMAP().fit_transform(pca_embeddings)
    print(f"Reduced shape: {reduced_embeddings.shape}")
    print(reduced_embeddings[-2:])
    # Extract x and y coordinates
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]

    # Plot the points
    # Plot Documents
    plt.scatter(x[:db_embeddings.shape[0]], y[:db_embeddings.shape[0]],
                c="Gray", marker=".")
    # Plot German Documents
    plt.scatter(x[db_embeddings.shape[0]:-english_embeddings.shape[0]], y[db_embeddings.shape[0]:-english_embeddings.shape[0]],
                c="Red", alpha=0.7)
    # Plot English Documents
    plt.scatter(x[-english_embeddings.shape[0]:], y[-english_embeddings.shape[0]:],
                c="Blue", alpha=0.7)

    #plt.title('2D Embedding Visualization\n'
    #          'Red: German\n'
    #          'Blue: English')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{embedding_path}/vis_{args.reduction}.svg")
    print(f"Finished! Results available at: {embedding_path}")


def get_min_euclidean(prompt_embeddings, data_embeddings):
    total_euclidean = 0
    for p in prompt_embeddings:
        min_euclidean = 1
        for e in data_embeddings:
            euclidean = np.linalg.norm(p-e)
            min_euclidean = min(min_euclidean, euclidean)
        total_euclidean += min_euclidean

    avg_min_euclidean = total_euclidean / len(prompt_embeddings)
    print(f"Average min euclidean: {avg_min_euclidean:.4f}")
    return avg_min_euclidean

def get_max_cosine(prompt_embeddings, data_embeddings):
    total_cosine = 0
    for p in prompt_embeddings:
        max_cosine = -1
        for e in data_embeddings:
            dot_product = np.dot(p, e)
        # Compute the norms of the vectors
        #mnorm_vector1 = np.linalg.norm(e_e)
        # norm_vector2 = np.linalg.norm(g_e)
        # Compute the cosine similarity
            cosine_similarity = dot_product #/ (norm_vector1 * norm_vector2)
            max_cosine = max(max_cosine, cosine_similarity)
        total_cosine += max_cosine
    avg_max_cosine = total_cosine/len(prompt_embeddings)
    print(f"Average max cosine similarity: {avg_max_cosine:.4f}")
    return avg_max_cosine


def get_openai_embeddings(text_list):
    load_dotenv("../src/.env")

    embeddings = []
    embedding_model = "text-embedding-ada-002"

    openai_client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                azure_deployment=embedding_model,
                                api_version='2023-05-15',
                                api_key=os.getenv("AZURE_OPENAI_KEY"))
    for t in tqdm(text_list):
        print(f"Embedding prompt...")
        response = openai_client.embeddings.create(input=t, model=embedding_model)
        time.sleep(1)
        embeddings.append(response.data[0].embedding)
    return embeddings

def eval_cd4a(args):
    cd4a_dir = f"{args.path}/cd4a"

    if not os.path.exists(cd4a_dir):
        os.makedirs(cd4a_dir)
    if os.path.exists(f"{cd4a_dir}/{args.embedding}.pkl"):
        with open(f"{cd4a_dir}/{args.embedding}.pkl", "rb") as f:
            body_embeddings, comment_embeddings, prompt_embeddings = pickle.load(f)
    else:
        with open(f"./evaluation_prompts.json", "r", encoding="utf-8") as f:
            cd4a_prompts = json.load(f)
        cd4a_db = retrieve_data("cd4a")
        print(len(cd4a_db[0]))
        bodies = [d[0] for d in cd4a_db]
        comments = [d[1] for d in cd4a_db]
        print(len(bodies))
        body_embeddings = np.array(get_openai_embeddings(bodies))
        comment_embeddings = np.array(get_openai_embeddings(comments))
        prompt_embeddings = np.array(get_openai_embeddings(cd4a_prompts))
        with open(f"{cd4a_dir}/{args.embedding}.pkl", "wb") as f:
            pickle.dump((body_embeddings, comment_embeddings, prompt_embeddings), f)

    cosine_body = get_max_cosine(prompt_embeddings, body_embeddings)
    cosine_comment = get_max_cosine(prompt_embeddings, comment_embeddings)
    euclidean_body = get_min_euclidean(prompt_embeddings, body_embeddings)
    euclidean_comment = get_min_euclidean(prompt_embeddings, comment_embeddings)
    with open(f"{cd4a_dir}/metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Body:\n"
                f"Max cosine: {cosine_body}, Min euclidean: {euclidean_body}\n"
                f"Comment:\n"
                f"Max cosine: {cosine_comment}, Min euclidean: {euclidean_comment}")

    combined_embeddings = np.concatenate((body_embeddings, comment_embeddings, prompt_embeddings), axis=0)
    print(f"Reducing with PCA")
    pca_embeddings = PCA(n_components=50).fit_transform(combined_embeddings)
    print(f"Applying {args.reduction} on array of shape {pca_embeddings.shape}")

    if args.reduction == "tsne":
        reduced_embeddings = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(pca_embeddings)
    elif args.reduction == "umap":
        reduced_embeddings = UMAP().fit_transform(pca_embeddings)
    print(f"Reduced shape: {reduced_embeddings.shape}")
    print(reduced_embeddings[-2:])
    # Extract x and y coordinates
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]

    # Plot the points
    # Plot Body
    plt.scatter(x[:body_embeddings.shape[0]], y[:body_embeddings.shape[0]],
                c="Blue", alpha=0.7, label="Models")
    # Plot Comment
    plt.scatter(x[body_embeddings.shape[0]:-prompt_embeddings.shape[0]], y[body_embeddings.shape[0]:-prompt_embeddings.shape[0]],
                c="Green", alpha=0.7, label="Comments")
    # Plot Prompts
    plt.scatter(x[-prompt_embeddings.shape[0]:], y[-prompt_embeddings.shape[0]:],
                c="Red", alpha=0.7, label="Prompts")
    plt.legend(title='Legend', loc='upper right', fontsize='medium', title_fontsize='large', frameon=True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{cd4a_dir}/vis_{args.reduction}.svg")
    print(f"Finished! Results available at: {cd4a_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", "-e", choices=["laser", "adav2", "sentence"], type=str, help="Embedding model to evaluate")
    parser.add_argument("--sentence_model", "-se", type=str, default="e5", choices=["e5", "e5-instruct", "e5-mistral"])
    parser.add_argument("--path", "-p", type=str, default="../data/embeddings/", help="Path to store visualization")
    parser.add_argument("--reduction", "-r", type=str, default="umap", choices=["tsne", "umap"])
    args = parser.parse_args()
    # eval(args)
    eval_cd4a(args)


