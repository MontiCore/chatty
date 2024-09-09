import socket

import pymongo.errors
from pymongo.mongo_client import MongoClient
import weaviate
import time
import sys


def wait_for_db_startup(databases):
    nosql_started = False
    vectordb_started = False
    retry_time = 30
    max_retries = 10
    startup_time = 30
    time.sleep(startup_time)
    retries = 0
    loop_time = 0
    while not (nosql_started and vectordb_started):
        if loop_time < retry_time and retries > 0:
            time.sleep(retry_time-loop_time)
        print(f"Starting try #{retries}")
        if retries >= max_retries:
            raise TimeoutError
        start_time = time.time()
        if '--local' in sys.argv:
            nosql_client = MongoClient("mongodb://localhost:27017/")["local"]
        else:
            nosql_client = MongoClient("mongodb://mongodb:27017/")["local"]
        if not nosql_started:
            try:
                res = nosql_client.command("ping")
                print(res)
                if res["ok"] == 1.0:
                    nosql_started = True
                    print("MongoDB connection successful!")
            except pymongo.errors.ServerSelectionTimeoutError:
                pass
        if not vectordb_started:
            try:
                if '--local' in sys.argv:
                    _ = weaviate.connect_to_local(port=8080, grpc_port=50051, host="localhost")
                else:
                    _ = weaviate.connect_to_local(port=8080, grpc_port=50051, host="weaviate")
                vectordb_started = True
                print("WeaviateDB connection successful!")
            except socket.gaierror:
                pass
        retries += 1
        loop_time = time.time() - start_time


if __name__ == "__main__":
    databases = ["weaviate", "mongo"]
    wait_for_db_startup(databases)
