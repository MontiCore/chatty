import weaviate
import os
import weaviate.classes as wvc
from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time
import json

load_dotenv("../src/.env")

embedding_model = "text-embedding-ada-002"
embedding_client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                         azure_deployment=embedding_model,
                         api_version='2023-05-15',
                         api_key=os.getenv("AZURE_OPENAI_KEY"))

chat_client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                  azure_deployment='Gpt35Turbo',
                                  api_version='2023-05-15',
                                  api_key=os.getenv("AZURE_OPENAI_KEY"))
try:
    vdb_client = weaviate.connect_to_local(port=8080, grpc_port=50051, host="weaviate")
except:
    vdb_client = weaviate.connect_to_local(port=8080, grpc_port=50051, host="localhost")


SYSTEM = """You are an expert for Domain Specific Languages (DSLs).
You will receive a code snippet of a DSL and your task is to create a natural language documentation of the code.
This should allow users, which are not familiar with code to understand the topic and the code just by reading the documentation.
Keep your documentation concise and include relevant key words to find it easily.
Also summarize the purpose of the code snippet in the first sentence.
*DO NOT* mention that it is a domain-specific language in your comment.
Begin your comment after the delimiter ```.
"""
USER = """
Title: {title}
Code: {code}

Comment: ```
"""

def get_comment(code, title):
    message = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(title=title, code=code)}]

    response = chat_client.chat.completions.create(model='gpt-3.5-turbo',
                                                   messages=message,
                                                   temperature=0.8,
                                                   n=1,
                                                   stream=False)
    comment = response.choices[0].message.content
    return comment


def get_embeddings(dsl):
    dir_path = f"../data/{dsl}"
    properties = []
    embeddings = []
    count = 0
    for f in tqdm(os.listdir(dir_path), desc="Creating embeddings..."):
        #if count > 100:
        #    break
        f_path = os.path.join(dir_path, f)
        size = os.path.getsize(f_path)
        if size > 2_500:  # Skip large files (>2.5kb)
            continue

        with open(f_path, "r") as file:
            doc = "".join(file.readlines()[1:])  # Skip first line
        title = f.split(".")[0]
        comment = get_comment(doc, title)
        properties.append({"document_title": title, "security_level": 1, "body": doc, "comment": comment})

        embedding = embedding_client.embeddings.create(input=[comment], model=embedding_model).data[0].embedding
        embeddings.append(embedding)
        count += 1
        time.sleep(0.1)
    return properties, embeddings



def main(dsls):
    for d in dsls:
        try:
            vdb_client.collections.delete(name=d)
        except:
            pass
        properties, embeddings = get_embeddings(d)
        collection = vdb_client.collections.create(name=d,
                                  vectorizer_config=None,
                                  generative_config=None,
                                  vector_index_config=wvc.Configure.VectorIndex.hnsw(
                                      distance_metric=wvc.VectorDistance.COSINE
                                  ))
        data_objects = []
        for p, e in zip(properties, embeddings):
            data_objects.append(wvc.DataObject(properties=p,
                                               vector=e))

        collection.data.insert_many(data_objects)


def test(prompt):
    collection = vdb_client.collections.get("cd4a")
    # print(prompt)
    prompt_embedding =embedding_client.embeddings.create(input=prompt,
                                                  model=embedding_model)
    similar_docs = collection.query.near_vector(prompt_embedding.data[0].embedding,
                                                limit=3,
                                                certainty=0.0,
                                                filters=wvc.Filter("security_level").less_or_equal(3))
    for i, doc in enumerate(similar_docs.objects):
        properties = doc.properties
        print(f"Document {i}\n"
              f"Title: {properties['document_title']}\n"
              f"Body:\n{properties['body']}")

def test_d():
    f = "Lecture1.cd"
    f_path = f"../data/cd4a/{f}"
    with open(f_path, "r") as file:
        doc = "".join(file.readlines()[1:])  # Skip first line
    title = f.split(".")[0]
    comment = get_comment(doc, title)
    print(f"{comment=}")


def weaviate_to_json(c_name, data_path):
    collection = vdb_client.collections.get(c_name)
    response = collection.aggregate.over_all(total_count=True)
    print(response.total_count)

    # response = collection.query.fetch_objects(include_vector=True)
    object_list = []
    for r in tqdm(collection.iterator(include_vector=True)):
        properties = r.properties
        properties["_additional"] ={"id": str(r.uuid), "vector": r.vector}
        object_list.append(properties)


    # Store the objects in a JSON file
    output_file = f"{data_path}/{c_name}.json"
    with open(output_file, "w") as json_file:
        json.dump(object_list, json_file, indent=2)

if __name__ == "__main__":
    # test("Create a class diagram")
    data_path = "../data/dsl"
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    dsls = ["cd4a"]
    weaviate_to_json("cd4a", data_path)
    # test_d()
    # main(dsls)

