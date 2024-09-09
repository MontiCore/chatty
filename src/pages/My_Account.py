import streamlit as st
import bcrypt
import sys
import time

sys.path.append("../src")
from menu import menu
from authentication import validate_password, logout

from PIL import Image

ICON_PATH = "./app_data/favicon.ico"
icon = Image.open(ICON_PATH)


def change_password():
    with st.form("password_form", clear_on_submit = True, border=False):
        current_pswd = st.text_input("Current Password", placeholder="Enter current password",
                                     type="password", max_chars=256)
        new_pswd = st.text_input("New Password", placeholder="Enter new password", type="password",
                                 max_chars=256)
        confirm_new_pswd = st.text_input("Confirm New Password", placeholder="Confirm new password", type="password",
                                         max_chars=256)
        submit = st.form_submit_button('Submit', type="primary")
        if submit:
            validate_result = validate_password(st.session_state.useremail, current_pswd)
            if validate_result is not None:
                if len(new_pswd) < 5:
                    st.error(f"Password is required to have at least 5 characters")
                elif new_pswd != confirm_new_pswd:
                    st.error(f"Passwords do not match. Please make sure new passwords are identical.")
                else:
                    user_hash, user = validate_result
                    collection = st.session_state.mongo_db["user_info"]
                    collection.update_one(
                        {"username": user_hash},
                        {"$set": {"password": bcrypt.hashpw(new_pswd.encode('utf8'), bcrypt.gensalt()).decode('utf8'), }}
                    )
                    st.success(f"Successfully change your password! Please log in with your new password.")
                    time.sleep(2)
                    logout()
                    st.switch_page("app.py")


def delete_account():
    st.warning("Are you sure you want to delete your account? This action cannot be undone. All your data will be permanently removed.")
    if st.button("Delete", type="primary"):
        collection = st.session_state.mongo_db["user_info"]
        user_hash = st.session_state.username
        result = collection.delete_one({"username": user_hash})

        if result.deleted_count > 0:
            st.success("User deleted successfully.")
            logout()
            st.switch_page("app.py")
        else:
            st.error("No user found with that username.")



st.set_page_config(page_title="My Account", page_icon=icon, layout="wide", initial_sidebar_state="auto")
st.session_state.switched_page = True

st.header(f"Hi, {st.session_state.useremail}")

change_pwd, delete = st.tabs(["Change Password", "Delete Account"])
with change_pwd:
    change_password()
with delete:
    delete_account()

menu()
