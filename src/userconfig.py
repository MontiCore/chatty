import json
import streamlit as st
from datetime import datetime
import time


def save_userdata():
    collection = st.session_state.mongo_db["user_data"]
    user_settings = {k: st.session_state[k] for k in st.session_state.default_settings.keys()}

    user_data = collection.find_one(filter={"username": st.session_state.username})
    if user_data is None:
        user_data = {"username": st.session_state.username, "settings": user_settings, "history": {},
                     "daily_tokens": {}}
        collection.insert_one(user_data)
    else:
        user_data["settings"] = user_settings
    collection.replace_one(filter={"username": st.session_state.username}, replacement=user_data)

    #if "chatty" not in st.session_state:
    #    collection.replace_one(filter={"username": st.session_state.username}, replacement=user_data)
    #    return

def save_message():
    date = str(datetime.now().date())
    collection = st.session_state.mongo_db["user_data"]
    user_data = collection.find_one(filter={"username": st.session_state.username})
    messages = st.session_state.chatty.messages
    if len(messages) > 1:  # Only store if user wrote something
        if st.session_state.session_id not in user_data["history"]:  # Stores session creation time
            user_data["history"][st.session_state.session_id] = {"time": time.time()}
        user_data["history"][st.session_state.session_id]["messages"] = messages

    # Update user rate limit
    if date in user_data["daily_tokens"]:
        user_data["daily_tokens"][date] += st.session_state.chatty.token_delta
    else:
        user_data["daily_tokens"][date] = st.session_state.chatty.token_delta
    collection.replace_one(filter={"username": st.session_state.username}, replacement=user_data)

    # Update global rate limit
    global_data = collection.find_one(filter={"username": "global"})
    if date in global_data["daily_tokens"]:
        global_data["daily_tokens"][date] += st.session_state.chatty.token_delta
    else:
        global_data["daily_tokens"][date] = st.session_state.chatty.token_delta
    collection.replace_one(filter={"username": "global"}, replacement=global_data)


def load_userdata():
    collection = st.session_state.mongo_db["user_data"]
    user_data = collection.find_one(filter={"username": st.session_state.username})
    with open("./app_data/default_settings.json", "r") as f:
        default_settings = json.load(f)
    st.session_state.default_settings = default_settings
    if user_data is not None:
        user_settings = user_data["settings"]
        for k, v in default_settings.items():
            if k in user_settings:
                st.session_state[k] = user_settings[k]
            else:
                st.session_state[k] = v
        st.session_state.session_history = user_data["history"]
    else:
        for k, v in default_settings.items():
            st.session_state[k] = v
        st.session_state.session_history = {}
