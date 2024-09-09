import time
import uuid
from dotenv import load_dotenv
import os
import weaviate
from openai import AzureOpenAI
import weaviate.classes as wvc
import logging
from abc import ABC, abstractmethod


class DatabaseInterface(ABC):
    def __init__(self, embedding_model: str):
        load_dotenv()
        self.embedding_model = embedding_model
        self.openai_client = AzureOpenAI(azure_endpoint=os.getenv("SWEDEN_OPENAI_ENDPOINT"),
                                         azure_deployment=embedding_model,
                                         api_version='2023-05-15',
                                         api_key=os.getenv("SWEDEN_OPENAI_KEY"))

    def get_embedding(self, prompt: str) -> list[float]:
        res = self.openai_client.embeddings.create(input=prompt, model=self.embedding_model)
        return res.data[0].embedding

    @abstractmethod
    def insert_db(self, title: str, content: str, username: str, role: str):
        raise NotImplementedError

    @abstractmethod
    def semantic_search(self, prompt: str, num_docs: int, min_similarity: float, security: int, collections: list) -> dict:
        raise NotImplementedError


class WeaviateInterface(DatabaseInterface):
    def __init__(self,
                 embedding_model="text-embedding-ada-002",
                 host="weaviate"):
        super().__init__(embedding_model)
        self.client = weaviate.connect_to_local(port=8080, grpc_port=50051, host=host)

    def insert_db(self, title, content, username="default", role="admin", target_collection="Documents"):
        collection = self.client.collections.get("Userdocs")
        properties = {"body": content, "user": username, "role": role, "status": "pending_approval",
                      "document_title": title, "submission_time": time.time(), "collection": target_collection,
                      "security_level": 1}
        collection.data.insert(properties=properties,
                               vector=self.get_embedding(content),
                               uuid=str(uuid.uuid4()))

    def semantic_search(self, prompt, num_docs, min_similarity=0.0, security=1, collections=None):
        # !Weaviate collections always start with uppercase letter!
        search_collections = []
        if collections is None or len(collections) == 0:
            search_collections.append(self.client.collections.get("Documents"))
        else:
            for c in collections:
                search_collections.append(self.client.collections.get(c.capitalize()))

        prompt_embedding = self.get_embedding(prompt)
        return_docs = {k: [[]] for k in ["metadatas", "documents", "cosine"]}
        for c in search_collections:
            similar_docs = c.query.near_vector(prompt_embedding,
                                               limit=num_docs,
                                               certainty=min_similarity,
                                               return_metadata=wvc.query.MetadataQuery(certainty=True),
                                               filters=wvc.query.Filter.by_property("security_level").less_or_equal(security))

            for doc in similar_docs.objects:
                logging.info(f"Cosine similarity: {doc.metadata.certainty}")
                properties = doc.properties
                return_docs["documents"][0].append(properties["body"])
                return_docs["cosine"][0].append(doc.metadata.certainty)
                return_docs["metadatas"][0].append({k: v for k, v in properties.items() if k not in ["body"]})
        # Limit to num_docs
        indexed_list = list(enumerate(return_docs["cosine"][0]))
        sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
        top_k_indices = [index for index, value in sorted_indexed_list[:num_docs]]
        for k, v in return_docs.items():
            return_docs[k][0] = [v[0][i] for i in top_k_indices]
        return return_docs
