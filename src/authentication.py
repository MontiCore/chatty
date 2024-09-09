import streamlit as st
import json
import time
import bcrypt
import hashlib
import streamlit.components.v1 as components
from captcha.image import ImageCaptcha
import secrets
import string
from userconfig import *


def authenticate(username, password):
    user_hash, user = validate_password(username, password)
    st.session_state.username = user_hash
    st.session_state.role = user["role"]
    st.session_state.global_limit = st.session_state.mongo_db["roles"].find_one({"role": "global"})["daily_limit"]
    role_permissions = st.session_state.mongo_db["roles"].find_one({"role": st.session_state.role})
    st.session_state.daily_limit = role_permissions["daily_limit"]
    st.session_state.security_level = role_permissions["security_level"]
    st.session_state.first_run = True
    st.session_state.authentication_status = "completed"
    st.session_state.useremail = username
    st.success(f"Logged in as {username}!")
    time.sleep(0.5)
    # st.rerun()
    st.switch_page("app.py")


def validate_password(username, password):
    collection = st.session_state.mongo_db["user_info"]
    user_hash = hashlib.sha256(username.encode('utf8')).hexdigest()
    user = collection.find_one(filter={"username": user_hash})
    if user is None:
        st.error(f"User with email {username} does not exist")
        return

    correct = bcrypt.checkpw(password.encode('utf8'), user["password"].encode('utf8'))
    if not correct:
        st.error(f"Incorrect password")
        return
    return user_hash, user


def register():
    collection = st.session_state.mongo_db["user_info"]
    with st.form("register"):
        st.text_input("Email address", placeholder="Enter email address", autocomplete="email", max_chars=256,
                      key="new_user")
        st.text_input("Password", placeholder="Enter password", autocomplete="new-password", type="password",
                      max_chars=256, key="new_pswd")
        st.text_input("Registration code", placeholder="Enter registration code", max_chars=256, key="code")
        reg1, reg2 = st.columns(2)
        register_user = reg1.form_submit_button("Register", type="primary", use_container_width=True)
    if register_user:
        #st.error("Registration is unavailable at this moment")
        #return
        user_hash = hashlib.sha256(st.session_state.new_user.encode('utf8')).hexdigest()
        if collection.find_one(filter={"username": user_hash}) is not None:
            st.error(f"User with email {st.session_state.new_user} already exists")
        elif len(st.session_state.new_pswd) < 5:
            st.error(f"Password is required to have at least 5 characters")
        else:
            user_code = st.session_state.mongo_db["email_codes"].find_one(filter={"email_hash": user_hash})
            if user_code is None:
                st.error("E-mail not found in whitelist, check the spelling or contact your course administrator")
                return
            elif user_code["code"] != st.session_state.code:
                st.error("One-time code does not match your e-mail!")
                return
            new_user = {"username": user_hash,
                        "password": bcrypt.hashpw(st.session_state.new_pswd.encode('utf8'), bcrypt.gensalt()).decode(
                            'utf8'),
                        "role": user_code["role"],
                        "registration_date": time.time()}
            collection.insert_one(new_user)

            user_data = {"username": user_hash, "settings": {}, "history": {},
                         "daily_tokens": {}}
            st.session_state.mongo_db["user_data"].insert_one(user_data)

            del st.session_state.new_pswd
            del st.session_state.new_user
            st.success(f"Successfully created new user! You can login now.")
            st.session_state.authentication_status = "login"


def generate_captcha(captcha_len):
    characters = "02345689"
    random_string = ''.join(secrets.choice(characters) for _ in range(captcha_len))
    image = ImageCaptcha(width=280, height=90)
    data = image.generate(random_string)
    st.image(data, caption="")
    return random_string


def verify_captcha(captcha_text, login_name, login_pswd):
    user_text = st.session_state.user_captcha
    if captcha_text == user_text:
        authenticate(login_name, login_pswd)
    else:
        st.error("Wrong CAPTCHA text")


def captcha_component(login_name, login_pswd):
    captcha_len = 7
    captcha_text = generate_captcha(captcha_len)
    with st.form("captcha"):
        st.text_input(label="Enter CAPTCHA", max_chars=captcha_len, help="Enter CAPTCHA text from image",
                      key="user_captcha")
        col1, col2 = st.columns(2)
        col1.form_submit_button(label="Submit", on_click=verify_captcha,
                                args=(captcha_text, login_name, login_pswd,), type="primary", use_container_width=True)
        # col2.form_submit_button(label="Reload", on_click=captcha_component, args=(login_name, login_pswd,), use_container_width=True)


def login(captcha):
    # login_container = st.empty()
    with st.form("login"):
        login_name = st.text_input("Email address", placeholder="Enter email address", autocomplete="email",
                                   max_chars=256)
        login_pswd = st.text_input("Password", placeholder="Enter password", autocomplete="current-password",
                                   type="password", max_chars=256)
        log1, log2 = st.columns(2)
        login_user = log1.form_submit_button("Login", type="primary", use_container_width=True)
    if login_user:
        if captcha:
            captcha_component(login_name, login_pswd)
        else:
            authenticate(login_name, login_pswd)


def logout():
    for key in st.session_state.keys():
        del st.session_state[key]
