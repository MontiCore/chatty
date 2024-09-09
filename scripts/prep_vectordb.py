import json
import time
# from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
import weaviate
import ijson
import sys

try:
    client = weaviate.connect_to_local(port=8080, grpc_port=50051, host="weaviate")
except:
    client = weaviate.connect_to_local(port=8080, grpc_port=50051, host="localhost")

def retrieve_json(save_file,
                  sciebo_link='https://rwth-aachen.sciebo.de/public.php/webdav/text-embedding-ada-002-scieneers.json',):
    load_dotenv("../src/.env")
    print("Downloading raw json...")
    basic = HTTPBasicAuth(os.getenv("SCIEBO_UNAME"), os.getenv("SCIEBO_PSWD"))
    response = requests.get(sciebo_link,
                            auth=basic,
                            stream=True)
    if response.status_code == 200:
        content_size = int(response.headers.get('content-length', 0))
        with open(save_file, "wb") as f, \
                tqdm(desc="Progress", total=content_size, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))


def preprocess_store(input_file, c_name, fix_format=True):
    print("Preprocessing data...")
    new_data = []
    # for item in data:
    chunk_size = 5000
    with open(input_file, "rb") as f:
        for i, item in tqdm(enumerate(ijson.items(f, "item")), desc="Loading data..."):
            updated_dict = {}
            for k, v in item.items():
                # Remove wordsplitting over linebreaks
                if fix_format and isinstance(v, str):
                    updated_dict[k] = v.replace("- ", "")
                elif k == "_additional":
                    v["vector"] = [float(d_v) for d_v in v["vector"]]
                    updated_dict[k] = v
                elif k == "citations":
                    if len(v) == 0:
                        updated_dict[k] = None
                    else:
                        updated_dict[k] = v
                else:
                    updated_dict[k] = v
            if "security_level" not in item:
                updated_dict["security_level"] = 1
            new_data.append(updated_dict)
            if (i+1) % chunk_size == 0:
                print("Storing chunk...")
                weaviate_db(new_data, c_name)
                new_data = []
        print("Storing chunk...")
        weaviate_db(new_data, c_name)


def create_weaviate_collections(collections, force=False):
    existing_collections = client.collections.list_all()
    created_list = []

    for c in collections:
        if force or not c in existing_collections:
            if client.collections.exists(name=c):
                client.collections.delete(name=c)

            if c == "Documents":
                property_config = [Property(name="document", data_type=DataType.TEXT),
                                   Property(name="chapter", data_type=DataType.TEXT),
                                   Property(name="document_group", data_type=DataType.TEXT),
                                   Property(name="document_title", data_type=DataType.TEXT),
                                   Property(name="file_type", data_type=DataType.TEXT),
                                   Property(name="page_nr", data_type=DataType.INT),
                                   Property(name="paragraph", data_type=DataType.INT),
                                   Property(name="section", data_type=DataType.TEXT),
                                   Property(name="subchapter", data_type=DataType.TEXT),
                                   Property(name="title", data_type=DataType.TEXT),
                                   Property(name="version", data_type=DataType.TEXT),
                                   Property(name="security_level", data_type=DataType.INT)]
            elif c == "Books":
                property_config = [Property(name="document", data_type=DataType.TEXT),
                                   Property(name="chapter", data_type=DataType.TEXT),
                                   Property(name="document_group", data_type=DataType.TEXT),
                                   Property(name="document_title", data_type=DataType.TEXT),
                                   Property(name="embedding_text", data_type=DataType.TEXT),
                                   Property(name="section", data_type=DataType.TEXT),
                                   Property(name="subsection", data_type=DataType.TEXT),
                                   Property(name="subsubsection", data_type=DataType.TEXT),
                                   Property(name="publish_date", data_type=DataType.TEXT),
                                   Property(name="authors", data_type=DataType.TEXT_ARRAY),
                                   Property(name="citations", data_type=DataType.TEXT_ARRAY),
                                   Property(name="tokens", data_type=DataType.INT),
                                   Property(name="index", data_type=DataType.INT),
                                   Property(name="security_level", data_type=DataType.INT)]
            else:
                property_config = [Property(name="security_level", data_type=DataType.INT)]


            client.collections.create(name=c,
                                       vectorizer_config=None,
                                       generative_config=None,
                                       properties=property_config,
                                       vector_index_config=Configure.VectorIndex.hnsw(
                                           distance_metric=VectorDistances.COSINE))
            created_list.append(c)
    return created_list

def weaviate_db(preprocess_dict, c_name):
    collection = client.collections.get(c_name)

    data_objects = []
    print("Creating data objects...")
    for i, d in tqdm(enumerate(preprocess_dict)):
        properties = {k: v for k, v in d.items() if k not in ["_additional"] and v is not None}
        data_objects.append(wvc.data.DataObject(properties=properties,
                                           vector=d["_additional"]["vector"],
                                            uuid=d["_additional"]["id"]))
    chunk_size = 1000
    print("Storing in database...")
    for data_chunk in tqdm([data_objects[j*chunk_size:min((j+1)*chunk_size, len(data_objects))] for j in range((len(data_objects)//chunk_size)+1)]):
        if len(data_chunk) == 0:
            break
        collection.data.insert_many(data_chunk)
        time.sleep(0.2)


def create_dsl_collection(name):
    dir_path = "../data"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    download_file = f"{dir_path}/{name}.json"
    retrieve_json(save_file=download_file,
        sciebo_link=f'https://rwth-aachen.sciebo.de/public.php/webdav/{name.lower()}.json')
    with open(download_file, "r") as f:
        data = json.load(f)
        weaviate_db(data, name)


def run_db_setup():
    force = '--force' in sys.argv
    collections = ["Documents", "Cd4a", "Userdocs", "Books"]
    books = ["MCHandbuch"]
    print("Checking if collections already exists...")

    created_collections = create_weaviate_collections(collections, force)
    if len(created_collections) == 0:
        print("Vector DB collections already exists")
        return

    for c in created_collections:
        print(f"Creating collection {c}")
        if c == "Userdocs":
            continue
        elif c == "Documents":
            download = True
            preprocessing = True
            input_file = "../data/text-embedding-ada-002-scieneers.json"

            if not os.path.isdir("../data"):
                os.makedirs("../data")
            if download:
                retrieve_json(input_file)
            if preprocessing:
                preprocess_store(input_file, c_name=c)
        elif c == "Books":
            book_dir = f"../data/latex/embedded"
            if not os.path.isdir(book_dir):
                os.makedirs(book_dir)
            for b in books:
                retrieve_json(f"{book_dir}/{b}.json",
                              sciebo_link=f'https://rwth-aachen.sciebo.de/public.php/webdav/{b}.json')
                preprocess_store(f"{book_dir}/{b}.json", c_name=c, fix_format=False)

        elif c in ["Cd4a"]:
            create_dsl_collection(c)

    print("Finished!")



if __name__ == "__main__":
    run_db_setup()
