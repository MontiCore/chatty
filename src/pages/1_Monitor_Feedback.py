import pymongo
import streamlit as st
import pandas as pd
import datetime
import sys
from PIL import Image

sys.path.append("../src")
from menu import menu

score_map = {"😀": 5, "🙂": 4, "😐": 3, "🙁": 2, "😞": 1}
ICON_PATH = "./app_data/favicon.ico"
icon = Image.open(ICON_PATH)

def collection_to_table(collection):
    df_dict = {"_id": [], "rating": [], "comment": [], "user_message": [], "ai_message": [], "time": [], "username": [], "annotation": []}

    for doc in collection.find().sort("time", pymongo.DESCENDING):
        # print(doc)
        df_dict["_id"].append(doc["_id"])
        df_dict["rating"].append(score_map[doc["feedback"]["score"]])
        df_dict["comment"].append(doc["feedback"]["text"])
        df_dict["user_message"].append(doc["user_message"])
        df_dict["ai_message"].append(doc["ai_message"])
        feedback_time = datetime.datetime.fromtimestamp(doc["time"]).strftime("%d/%m/%y")
        df_dict["time"].append(feedback_time)
        if "username" in doc:
            df_dict["username"].append(doc["username"])
        else:
            df_dict["username"].append("default")
        if "annotation" not in doc.keys():
            doc["annotation"] = ""
        df_dict["annotation"].append(doc["annotation"])
    #for k, v in df_dict.items():
    #    print(f"Key {k}, num_items: {len(v)}")
    return df_dict

def print_session_state():
    history = st.session_state.session_history
    del st.session_state.session_history
    st.write(st.session_state)
    st.session_state.session_history = history


def edit_feedback(**kwargs):
    collection = kwargs["collection"]
    df = kwargs["df"]
    for row_idx, value_dict in st.session_state.edited_df["edited_rows"].items():
        fb_id = df["_id"][int(row_idx)]
        for k, v in value_dict.items():
            collection.update_one(filter={"_id": fb_id}, update={"$set": {k: v}})

def monitor():

    #with st.sidebar:
    #    st.button(label="Print session state", on_click=print_session_state)
    collection = st.session_state.mongo_db["feedback"]
    df = pd.DataFrame(data=collection_to_table(collection))
    container = st.empty()
    container.data_editor(
        df,
        disabled=("id", "rating", "comment", "user_message", "ai_message", "time", "username"),
        use_container_width=True,
        on_change=edit_feedback,
        column_config={"_id": None},
        kwargs={"collection": collection, "df": df},
        key="edited_df"
    )
    # st.dataframe(data=df)

st.set_page_config(page_title="Feedback", page_icon=icon, layout="wide", initial_sidebar_state="auto")
st.session_state.switched_page = True
if not ("authentication_status" in st.session_state) or st.session_state.authentication_status != "completed" or \
        (st.session_state.role not in ["admin", "wimi"]):
    st.error("Access denied! Login to an account with admin or wimi permissions.")
else:
    st.header("View Feedback")
    monitor()
menu()