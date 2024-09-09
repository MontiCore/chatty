import logging

import streamlit as st
import pandas as pd
import datetime
import time
import sys
from PIL import Image

sys.path.append("../src")
from menu import menu

# TODO cache?
ICON_PATH = "./app_data/favicon.ico"
icon = Image.open(ICON_PATH)

def change_entry(**kwargs):
    db_client = st.session_state.chatty.database.client
    df = kwargs["df"]
    print(st.session_state.db_changes["edited_rows"])
    for row_idx, value_dict in st.session_state.db_changes["edited_rows"].items():
        d_id = df["id"][int(row_idx)]
        prev_collection = db_client.collections.get(df["collection"][int(row_idx)])
        user_coll = db_client.collections.get("Userdocs")
        vector = user_coll.query.fetch_object_by_id(d_id, include_vector=True).vector
        new_values = {}
        for k, v in df.items():
            if k in value_dict:
                new_values[k] = value_dict[k]
            elif k != "id":
                new_values[k] = df[k][int(row_idx)]
            if k == "security_level":
                new_values[k] = int(new_values[k])
        logging.info(f"New values dict: {new_values},")

        collection = db_client.collections.get(new_values["collection"])
        was_approved = df["status"][int(row_idx)]
        if "status" in value_dict:
            if value_dict["status"]:
                if not was_approved or "collection" in value_dict:
                    try:
                        collection.data.insert(
                            properties=new_values,
                            uuid=d_id,
                            vector=vector)
                    except:
                        collection.data.update(
                            properties=new_values,
                            uuid=d_id,
                            vector=vector)
                elif "collection" not in value_dict:
                    collection.data.update(
                        properties=new_values,
                        uuid=d_id
                    )
                if "collection" in value_dict:
                    prev_collection.data.delete_by_id(d_id)
            elif not value_dict["status"] and was_approved:
                collection.data.delete_by_id(d_id)
        new_values["status"] = "approved" if new_values["status"] else "pending_approval"
        user_coll.data.update(properties=new_values, uuid=d_id)

    for row in st.session_state.db_changes["deleted_rows"]:
        d_id = df["id"][int(row)]
        coll = db_client.collections.get("Userdocs")
        coll.data.delete_by_id(d_id)

PROCESSED_KEY = 'processed'
def approve_db():
    db_client = st.session_state.chatty.database.client
    collection = db_client.collections.get("Userdocs")
    object_list = []
    for r in collection.iterator(include_vector=False):
        # properties = r.properties
        # properties["_additional"] ={"id": str(r.uuid), "vector": r.vector}
        object_list.append(r)
    if len(object_list) == 0:
        st.write("No entries to approve")
        return

    # Add collection to df entries
    keys = object_list[0].properties.keys()
    df = pd.DataFrame(columns=keys)
    # Append values of each dictionary as rows to the DataFrame
    for d in object_list:
        properties = d.properties
        properties["id"] = str(d.uuid)

        dt_object = datetime.datetime.fromtimestamp(properties["submission_time"])
        properties["submission_time"] =  time.mktime(datetime.datetime.strptime(dt_object.strftime('%Y-%m-%d/%H:%M:%S'),'%Y-%m-%d/%H:%M:%S').timetuple())
        properties["status"] = True if properties["status"] == "approved" else False
        if PROCESSED_KEY not in properties.keys():
            properties[PROCESSED_KEY] = False
        if properties["status"]:
            properties[PROCESSED_KEY] = True
        df = pd.concat([df, pd.DataFrame([properties])], ignore_index=True)

    st.data_editor(df,
                   disabled=(k for k in keys if
                             k not in ["body", "document_title", "collection", "status", "security_level",
                                       PROCESSED_KEY]),
                   num_rows="dynamic",
                   on_change=change_entry,
                   kwargs={"df": df},
                   column_config={
                       "id": None,
                       "status": st.column_config.CheckboxColumn(label="approved", default=properties['status']),
                       #PROCESSED_KEY: st.column_config.CheckboxColumn(label=PROCESSED_KEY, default=properties[PROCESSED_KEY]),
                       "collection": st.column_config.SelectboxColumn(options=["Documents", "Cd4a"], required=True,
                                                                      help="Target collection")},

                   use_container_width=True,
                   column_order=['status','processed','body','document_title','role','user','submission_time','security_level'],
                   key="db_changes")


st.set_page_config(page_title="Approve DB", page_icon=icon, layout="wide", initial_sidebar_state="auto")
st.session_state.switched_page = True
if not ("authentication_status" in st.session_state) or st.session_state.authentication_status != "completed" or \
        (st.session_state.role not in ["admin", "wimi"]):
    st.error("Access denied! Login to an account with admin or wimi permissions.")
else:
    st.header("Approve DB entries")
    approve_db()
menu()