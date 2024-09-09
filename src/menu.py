import streamlit as st

def menu():
    if st.session_state.role in ["wimi", "admin"]:
        st.sidebar.page_link("app.py", label="Home")
        st.sidebar.page_link("pages/1_Monitor_Feedback.py", label="Monitor Feedback")
        st.sidebar.page_link("pages/3_Approve_DB.py", label="Approve_DB")
    if st.session_state.role == "admin":
        st.sidebar.page_link("pages/2_User_Management.py", label="User Management")
    # print("my role is: ", st.session_state.role)
    st.sidebar.page_link("pages/My_Account.py", label="My Account")
    st.sidebar.page_link("pages/4_Impressum.py", label="Impressum")