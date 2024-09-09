import pymongo
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.append("../src")
from menu import menu
from PIL import Image

ICON_PATH = "./app_data/favicon.ico"
icon = Image.open(ICON_PATH)

def edit_user(**kwargs):
    collection_info = st.session_state.mongo_db["user_info"]
    collection_data = st.session_state.mongo_db["user_info"]

    for row_idx, value_dict in st.session_state.edited_user_df["edited_rows"].items():
        user = kwargs["df"]["username"][int(row_idx)]
        for k, v in value_dict.items():
            if k == "role":
                collection_info.update_one(filter={"username": user}, update={"$set": {"role": v}})

def edit_roles(**kwargs):
    collection = st.session_state.mongo_db["roles"]
    for row_idx, value_dict in st.session_state.edited_roles_df["edited_rows"].items():
        role = kwargs["df"]["role"][int(row_idx)]
        for k, v in value_dict.items():
            collection.update_one(filter={"role": role}, update={"$set": {k: v}})


def manage_roles():
    """
    st.session_state.mongo_db.create_collection("roles")
    collection = st.session_state.mongo_db["roles"]
    collection.insert_many([{"role": "admin", "security_level": 5, "daily_limit": 100_000_000},
                     {"role": "wimi", "security_level": 3, "daily_limit": 5_000_000},
                     {"role": "student", "security_level": 1, "daily_limit": 100_000}])
    """
    collection = st.session_state.mongo_db["roles"]
    roles = collection.find()
    empty_df = {k: [] for k in roles[0].keys()}
    for r in roles:
        for k, v in r.items():
            empty_df[k].append(v)

    df = pd.DataFrame(empty_df)
    container = st.empty()
    container.data_editor(
        df,
        disabled=("_id", "role"),
        use_container_width=True,
        column_config={"_id": None,
                       "security_level": st.column_config.NumberColumn(required=True, min_value=0, max_value=5, step=1),
                       "daily_limit": st.column_config.NumberColumn(required=True, step=1)},
        on_change=edit_roles,
        kwargs={"df": df},
        key="edited_roles_df"
    )





def manage_users():
    collection = st.session_state.mongo_db["user_info"]
    user_data = st.session_state.mongo_db["user_data"]
    roles = [res["role"] for res in st.session_state.mongo_db["roles"].find()]

    users = collection.find()
    keys = users[0].keys()
    empty_df = {k: [] for k in keys}

    empty_df["daily_tokens"] = []
    dates = []
    for i in range(10):
        dates.append(str(datetime.now().date() - timedelta(days=i)))
    for u in users:
        u_data = user_data.find_one(filter={"username": u["username"]})
        if u_data is not None and "daily_tokens" in u_data:
            usage = []
            for d in reversed(dates):
                if d in u_data["daily_tokens"]:
                    usage.append(u_data["daily_tokens"][d])
                else:
                    usage.append(0)
            empty_df["daily_tokens"].append(usage)
        else:
            # print("No daily tokens found")
            empty_df["daily_tokens"].append([0 for _ in range(10)])
        for k in keys:
            if k == "registration_date":
                empty_df[k].append(datetime.fromtimestamp(u[k]).strftime("%Y-%m-%d/%H:%M:%S"))
            else:
                empty_df[k].append(u[k])
    df = pd.DataFrame(empty_df)
    container = st.empty()

    container.data_editor(
        df,
        disabled=("_id", "username", "password", "daily_tokens"),
        use_container_width=True,
        column_config={"_id": None, "password": None,
                       "role": st.column_config.SelectboxColumn(required=True, default="student", options=[r for r in roles]),
                       "daily_tokens": st.column_config.BarChartColumn(help="Token usage of the last 10 days",
                                                                       y_min=0, y_max=st.session_state.mongo_db["roles"].find_one({"role": "student"})["daily_limit"])},
        on_change=edit_user,
        kwargs={"df": df},
        key="edited_user_df"
    )

st.set_page_config(page_title="User management", page_icon=icon, layout="wide", initial_sidebar_state="auto")
st.session_state.switched_page = True
if not ("authentication_status" in st.session_state) or st.session_state.authentication_status != "completed" or \
        st.session_state.role != "admin":
    st.error("Access denied! Login to an account with admin permissions.")
else:
    st.header("Manage role permissions")
    manage_roles()
    st.header("Manage users")
    manage_users()
menu()