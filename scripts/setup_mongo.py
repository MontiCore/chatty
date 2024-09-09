import os.path

from pymongo.mongo_client import MongoClient
import json
import time
import sys

def create_new_collections():
    """
    Fills the db with collections and initial values, if the collections do not exist in the db
    """
    if '--local' in sys.argv:
        mongo_db = MongoClient("mongodb://localhost:27017/")["local"]
    else:
        mongo_db = MongoClient("mongodb://mongodb:27017/")["local"]
    collections = ["user_info", "user_data", "roles", "feedback", "email_codes"]
    existing_collections = mongo_db.list_collection_names()
    if "email_codes" in existing_collections:
        mongo_db.drop_collection("email_codes")
    if "roles" in existing_collections:
        mongo_db.drop_collection("roles")
    existing_collections = mongo_db.list_collection_names()
    print(f"Creating collections: {collections}")
    for c in collections:

        if c not in existing_collections:
            mongo_db.create_collection(c)
            if c == "user_info":
                with open("../scripts/hashed_initial_users.json", "r") as f:
                    initial_users = json.load(f)
                for k, v in initial_users.items():
                    mongo_db["user_info"].insert_one({"username": v["user_hash"],
                                                      "password": v["pswd_hash"],
                                                      "role": k,
                                                      "registration_date": time.time()})

            elif c == "roles":
                mongo_db["roles"].insert_many([{"role": "admin", "security_level": 5, "daily_limit": 2_000_000},
                                               {"role": "wimi", "security_level": 3, "daily_limit": 1_000_000},
                                               {"role": "student", "security_level": 1, "daily_limit": 250_000},
                                               {"role": "global", "security_level": 1, "daily_limit": 25_000_000}])
            elif c == "email_codes":
                with open("../scripts/email_codes.json", "r") as f:
                    email_codes = json.load(f)
                for k, v in email_codes.items():
                    mongo_db["email_codes"].insert_one({"code": v["onetime_code"], "email_hash": v["email_hash"], "role": "student"})
                    if int(k) % 50 == 0:
                        time.sleep(0.5)  # Do not spam the db with requests
                with open("../scripts/wimi_email_codes_mapping.json", "r") as f:
                    email_codes = json.load(f)
                    for k, v in email_codes.items():
                        mongo_db["email_codes"].insert_one(
                            {"code": v["onetime_code"], "email_hash": v["email_hash"], "role": "wimi"})
                        if int(k) % 50 == 0:
                            time.sleep(0.5)  # Do not spam the db with requests

                mongo_db["onetime_codes"]
            elif c == "user_data":
                mongo_db["user_data"].insert_one({"username": "global", "daily_tokens": {}})
            print(f"Collection '{c}' created")
        else:
            if c == "user_info":
                print("Deleting users")
                with open("../scripts/delete.json", "r") as f:
                    delete_users = json.load(f)
                new_delete_users = []
                for v in delete_users:
                    if not v["deleted"]:
                        mongo_db["user_info"].delete_one({"username": v["hash"]})
                        if "feedback" in existing_collections:
                            mongo_db["feedback"].delete_many({"username": v["hash"]})
                        if "user_data" in existing_collections:
                            mongo_db["user_data"].delete_one({"username": v["hash"]})
                        print(f"User with hash {v['hash']} deleted")
                        v["deleted"] = True
                    new_delete_users.append(v)
                with open("delete.json", "w") as f:
                    json.dump(new_delete_users, f)
            print(f"Collection '{c}' already exists")


if __name__ == "__main__":
    create_new_collections()
