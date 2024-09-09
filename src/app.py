import logging
import uuid
import streamlit as st
from chatty import Chatty
from utils import *
import os
import time
from datetime import datetime
from PIL import Image
from authentication import *
from userconfig import *
from dotenv import load_dotenv
from streamlit_feedback import streamlit_feedback
from pymongo.mongo_client import MongoClient
from menu import menu
import sys

load_dotenv()
ICON_PATH = "./app_data/favicon.ico"
icon = Image.open(ICON_PATH)

def disable_browser_error():
    def exception_handler(e):
        st.error("Error occured in backend! Try reloading the page.")
        print(e)
    script_runner = sys.modules["streamlit.runtime.scriptrunner.script_runner"]
    script_runner.handle_uncaught_app_exception = exception_handler


def save_feedback(feedback, **kwargs):
    collection = st.session_state.mongo_db["feedback"]
    # feedback = st.session_state[f"feedback_{kwargs['index']}"]
    user_message, ai_message = st.session_state.chatty.messages[kwargs['index']:kwargs['index']+2]
    feedback_doc = {"time": time.time(), "feedback": feedback, "user_message": user_message["content"], "ai_message": ai_message["content"], "username": st.session_state.username}
    collection.insert_one(feedback_doc)
    st.session_state.chatty.set_widget({"name": "success", "content": "Feedback saved!"}, kwargs["index"]+1)


def restart_chatty():
    save_userdata()
    start_chatty()


def start_operation(**kwargs):
    st.session_state.ongoing_operation = True
    st.session_state.operation = kwargs["operation"]


def create_description():
    st.subheader("SE-Chatty", help="SE-Chatty works best with english requests.")


def print_session_state():
    st.write(st.session_state)


def switch_session():
    save_userdata()
    st.session_state.session_id = st.session_state.new_session
    st.session_state.chatty.messages = st.session_state.session_history[st.session_state.new_session]["messages"]


def format_session(session_id):
    if len(st.session_state.session_history[session_id]["messages"]) > 1:
        question = st.session_state.session_history[session_id]["messages"][1]["content"]
        words = question.split()
        if len(words) > 5:
            session_spellout = ' '.join(words[:5]) + '...'
        else:
            session_spellout = question

    else:
        session_spellout = "New chat"
    return session_spellout


def clear_history():
    st.session_state.session_history = {st.session_state.session_id: st.session_state.session_history[st.session_state.session_id]}
    collection = st.session_state.mongo_db["user_data"]
    collection.update_one(filter={"username": st.session_state.username}, update={"$set": {"history": {}}})


def create_sidebar():
    # Sidebar settings
    # st.session_state.key to access values
    disabled = st.session_state.ongoing_operation
    with st.sidebar:
        st.button('New session', key="restart", on_click=restart_chatty, use_container_width=True)
        st.button("Generate model", key="dsl",
                  help="Will open a form to generate a model",
                  on_click=start_operation, kwargs={"operation": "generate_dsl"},
                  disabled=st.session_state.ongoing_operation,
                  use_container_width=True)

        st.header("Session management")
        if "session_history" in st.session_state:
            sorted_ids = sorted(st.session_state.session_history, key=lambda x: st.session_state.session_history[x]["time"], reverse=True)
            st.selectbox("Session", options=sorted_ids, disabled=disabled, index=0, on_change=switch_session,
                         key="new_session", format_func=format_session, help="Switch to previous conversations")

        st.button(label="Clear session history", on_click=clear_history)

        with st.expander("Settings", expanded=False):
            # Add option once other LLMs are supported
            # st.session_state.model = \
            if st.session_state.role in ["wimi", "admin"]:
                format_map = {"gpt-3.5-turbo": "GPT-3.5", "gpt-4o-mini": "GPT-4o-mini", "gpt-4o": "GPT-4o"}
                st.selectbox("LLM", options=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"], disabled=disabled, index=0, on_change=restart_chatty,
                             key="model", help="Change the LLM used internally", format_func=lambda x: format_map[x])

            st.select_slider("RAG", options=["off", "auto", "on"], key="rag", disabled=st.session_state.ongoing_operation, help="Use RAG to retrieve documents from a database?")

            st.slider(label="Temperature", min_value=0., max_value=2.,
                      step=0.05, key="temp", disabled=disabled, help=format_markdown("*Change LLM temperature*\n"
                                                                                     "**Higher**: More creative answers\n"
                                                                                     "**Lower**: More deterministic answers"))
            st.checkbox(label="Stream output", key="stream", disabled=disabled, help=format_markdown("Streams the output e.g., as in ChatGPT\n"
                                                                                                     "Only works after semantic search is completed"))
            st.toggle("Re-rank", key="rerank", help="**Re-ranks** the retrieved documents to improve quality",
                      disabled=st.session_state.ongoing_operation)
            st.toggle("Post-process Documents", key="post_process",
                      help="**Post-processes** the retrieved **documents** to only contain information relevant to the user query",
                      disabled=st.session_state.ongoing_operation)

            st.header("Semantic search")
            st.slider(label="Max number of documents to include",
                      min_value=1, max_value=10, step=1, key="num_docs", disabled=disabled, help=format_markdown("Sets the number of retrieved documents using semantic search.\n"
                                                                                                 "Can be lower due to filtering"))
            st.slider(label="Minimum cosine similarity",
                      min_value=-1.0, max_value=1.0, step=0.01, key="min_similarity", disabled=disabled, help=format_markdown("Specifies the cosine similarity threshold for semantic search.\n"
                                                                                                       "More relevant documents have a higher cosine similarity"))

            if st.session_state.role != "student":
                st.subheader("Re-ranking")
                st.slider(label="Number of documents to rank",
                          min_value=1, max_value=100, step=1, key="num_rerank", disabled=disabled or st.session_state.role == "student", help="Number of documents retrieved by semantic search when using the re-rank feature")

            # Semantic search settings
            st.subheader("LLM Settings")
            st.checkbox(label="Limit output tokens", key="limit_tokens", disabled=disabled,
                        help="Limits the output length of the Assistant")
            st.slider(label="Max tokens", min_value=10, max_value=1000, step=1, key="tokens",
                      disabled=disabled or not st.session_state.limit_tokens,
                      help="Each word can consist of multiple tokens")
            # theme change
            st.subheader("Theme")
            if "theme_refreshed" not in st.session_state:
                st.session_state.theme_refreshed = True
            st.toggle(label="dark", key="theme_dark", on_change=change_theme)
            if st.session_state.theme_refreshed == False:
                st.session_state.theme_refreshed = True
                st.rerun()
            # reset to default
            st.button(label="Reset to Default", on_click=reset_settings)

        menu()
        st.button(label="Logout", on_click=logout, use_container_width=True)

def change_theme():
    st.session_state.theme_refreshed = False
    theme = "dark" if st.session_state.theme_dark else "light"
    st._config.set_option('theme.base', theme)

def reset_settings():
    default_settings = st.session_state.default_settings
    for k, v in default_settings.items():
        st.session_state[k] = v
    change_theme()

def update_settings():
    st.session_state.chatty.change_settings("limit_tokens", st.session_state.limit_tokens)
    if st.session_state.limit_tokens:
        st.session_state.chatty.change_settings("max_tokens", st.session_state.tokens)
    st.session_state.chatty.change_settings("temperature", st.session_state.temp)
    if "num_rerank" in st.session_state.keys():
        st.session_state.chatty.change_settings("num_rerank", st.session_state.num_rerank)
    if "num_docs" in st.session_state.keys():
        st.session_state.chatty.change_settings("num_docs", st.session_state.num_docs)
    if "min_similarity" in st.session_state.keys():
        st.session_state.chatty.change_settings("min_similarity", st.session_state.min_similarity)
    if "min_rating" in st.session_state.keys():
        st.session_state.chatty.change_settings("min_rating", st.session_state.min_rating)


def stream_message(reply):
    if isinstance(reply, str):
        st.session_state.chatty.set_widget({"name": "error", "content": reply})
    else:
        response = st.chat_message("assistant").write_stream(reply)
        st.session_state.chatty.messages.append({"role": "assistant", "content": response})
    st.session_state.chatty.calculate_tokens()


def has_permission():
    date = str(datetime.now().date())
    daily_tokens = st.session_state.mongo_db["user_data"].find_one({"username": st.session_state.username})["daily_tokens"]
    global_tokens = st.session_state.mongo_db["user_data"].find_one({"username": "global"})["daily_tokens"]
    if date in daily_tokens and daily_tokens[date] >= st.session_state.daily_limit:
        st.error(f"Daily token limit of {st.session_state.daily_limit} reached, reach out to administrator to increase your limit")
        return False
    elif date in global_tokens and global_tokens[date] >= st.session_state.global_limit:
        st.error(f"Application wide token limit reached, try again later")
        return False
    elif st.session_state.rerank and st.session_state.role == "student":
        st.error(f"No permission to use this feature")
        return False
    else:
        return True


def display_additional_info(question):
    if "additional" in question or "info" in question:
        expander = st.popover(":mag:")
        if "info" in question:
            expander.write(question["info"])
        if "additional" in question:
            expander.json(question["additional"])


def create_old_messages():
    messages = st.session_state.chatty.messages
    for msg_idx, msg in enumerate(messages[1:]):  # Does not print system message
        st.chat_message(msg["role"]).write(format_markdown(msg["content"]))
        if msg["role"] == "assistant":
            question = messages[msg_idx]
            c1, c2, c3 = st.columns([0.5, 0.2, 0.3])
            with c2:
                if st.session_state.role not in ["student"] and msg_idx == len(messages) - 2:
                    b1, b2 = st.columns(2)
                    with b1:
                        display_additional_info(question)
                    with b2:
                        st.button(':minidisc:', key="store_db", on_click=start_operation,
                                  kwargs={"operation": "store_db"},
                                  disabled=len(st.session_state.chatty.messages) <= 1 or st.session_state.ongoing_operation,
                                  use_container_width=True)
                else:
                    display_additional_info(question)

            with c3:
                if (f"feedback_submitted_feedback_{msg_idx}" not in st.session_state) or (
                not st.session_state[f"feedback_submitted_feedback_{msg_idx}"]):
                    streamlit_feedback(feedback_type="faces", optional_text_label="[Optional] Feedback",
                                       key=f"feedback_{msg_idx}", on_submit=save_feedback, kwargs={"index": msg_idx})

        if "widgets" in msg:
            for widget in msg["widgets"]:
                if widget["name"] == "success":
                    st.success(widget["content"])
                elif widget["name"] == "warning":
                    st.warning(widget["content"])
                elif widget["name"] == "error":
                    st.error(widget["content"])
                else:
                    st.warning(f"Widget {widget['name']} currently not supported!")


def create_input():
    with st.container():
        db_options = ["Documents", "Books"]
        st.multiselect(label_visibility="collapsed", label="", placeholder="Database Collections", options=db_options,
                       key="collections",
                       disabled=st.session_state.ongoing_operation,
                       help=format_markdown("*Select database collections to search*\n"
                                            "**Documents**: Contains various document chunks\n"
                                            "**Books**: Contains software engineering books\n"))
        user_message = st.chat_input(placeholder="What would you like to know?",
                                     disabled=st.session_state.ongoing_operation, max_chars=2048,
                                     on_submit=start_operation, kwargs={"operation": "generate_message"}, key="input")
        show_used_token()

    return user_message

def show_used_token():
    context_size, used_token = st.columns([1,2])
    context = st.session_state.chatty.context
    with context_size:
        context_map = {"gpt-4o": 128_000, "gpt-4o-mini": 128_000, "gpt-3.5-turbo": 16_384}  # https://platform.openai.com/docs/models
        st.caption(f"Context size: {context}/{context_map[st.session_state.model]}")
    with used_token:
        rate_limit = st.session_state.mongo_db["roles"].find_one({"role": st.session_state.role})["daily_limit"]
        tokens = st.session_state.chatty.used_tokens
        st.caption(f"Used tokens: {tokens}/{rate_limit}")

def generate_dsl():
    if "stage" not in st.session_state:
        st.session_state.stage = "query"
    if st.session_state.stage == "query":
        with st.form(key="modify_entry"):
            st.subheader("Generate a model")
            st.selectbox(label="MontiCore DSL", options=["cd4a"], index=0, key="output_dsl", format_func=lambda x: x.upper())
            st.text_area(label="Query", placeholder="Enter query for model generation...", key="dsl_query")
            store_col = st.columns(2)
            submit = store_col[0].form_submit_button("Generate", help="Generate model", type="primary", use_container_width=True)
            cancel = store_col[1].form_submit_button("Cancel", use_container_width=True)

        if submit or cancel:
            if submit:
                with st.spinner("Generating questions..."):
                    questions, context_prompt = st.session_state.chatty.dsl_questions(user_message=st.session_state.dsl_query,
                                                                                      dsl=st.session_state.output_dsl)
                    st.session_state.desired_dsl = st.session_state.output_dsl
                    st.session_state.user_message = st.session_state.dsl_query
                    st.session_state.questions = questions
                    st.session_state.context_prompt = context_prompt
                st.session_state.stage = "questions"
                st.rerun()
            elif cancel:
                st.session_state.chatty.set_widget({"name": "warning", "content": "Canceled DSL generation!"})
                st.session_state.ongoing_operation = False
                st.rerun()

    if st.session_state.stage == "questions":
        if st.session_state.questions is not None:
            with st.form("question_form"):
                for k, v in st.session_state.questions.items():
                    st.text_input(label=v, placeholder="[Optional] Provide additional information", key=k)
                generate = st.form_submit_button("Generate", type="primary",
                                                 help="Generate response to original question", )
        else:
            generate = True
        if generate:
            status = st.status("Generating model...", expanded=True)

            question_answers = ""
            if st.session_state.questions is not None:
                for k, v in st.session_state.questions.items():
                    if k in st.session_state and st.session_state[k] != "":
                        question_answers += f"Question: {v}\n Answer: {st.session_state[k]}.\n"
            dsl_model = st.session_state.chatty.dsl_pipeline(user_message=st.session_state.user_message,
                                                             context_prompt=st.session_state.context_prompt,
                                                             questions=question_answers,
                                                             dsl=st.session_state.desired_dsl,
                                                             stream=st.session_state.stream,
                                                             status=status)
            if st.session_state.stream:
                status.write(f"Generating final model...")
                stream_message(dsl_model)
            status.write(f"Verifying Syntax...")
            cor = st.session_state.chatty.check_syntax(dsl=st.session_state.desired_dsl, index=-1)
            if cor:
                st.session_state.chatty.set_widget({"name": "success", "content": "Syntax check passed!"})
            else:
                st.session_state.chatty.set_widget({"name": "warning", "content": "Syntax check failed."})
            save_message()
            status.update(label="Finished stages", state="complete", expanded=False)
            st.session_state.ongoing_operation = False
            del st.session_state.stage
            st.rerun()


def create_new_message(user_message):
    st.chat_message("user").write(format_markdown(user_message))
    status = st.status("Generating response...", expanded=True)
    reply, status = st.session_state.chatty.qa_pipeline(user_message, status=status,
                                                        rag=st.session_state.rag,
                                                        post_process=st.session_state.post_process,
                                                        stream=st.session_state.stream,
                                                        security=st.session_state.security_level,
                                                        re_rank=st.session_state.rerank,
                                                        collections=st.session_state.collections)

    if st.session_state.stream:
        stream_message(reply)

    status.update(label="Finished stages", state="complete", expanded=False)
    save_message()
    st.session_state.ongoing_operation = False
    st.rerun()


def store_db():
    if st.session_state.role in ["student"]:
        st.error("You do not have permissions for this action")
        st.session_state.ongoing_operation = False
        return
    messages = st.session_state.chatty.messages
    collection_options = ["Documents", "Cd4a"]
    additional_list = [m["additional"] for m in messages if "additional" in m]
    c_index = 0
    if len(additional_list) > 0 and "fewshot" in additional_list[-1] and not "documents" in additional_list[-1]:
        c_index = 1

    with st.form(key="modify_entry"):
        st.subheader("New database entry")
        st.selectbox(label="Collection", options=collection_options, index=c_index, key="db_collection")
        st.text_input(label="Document title", placeholder="Enter title...", key="db_title")
        st.text_area(label="Document body", height=250, value=messages[-1]['content'], key="db_content")
        store_col = st.columns(2)
        submit = store_col[0].form_submit_button("Store", help="Store document in database", type="primary", use_container_width=True)
        cancel = store_col[1].form_submit_button("Cancel", use_container_width=True)
    if submit or cancel:
        if submit:
            try:
                with st.spinner("Storing data..."):
                    st.session_state.chatty.store_db(st.session_state.db_title, st.session_state.db_content,
                                                     st.session_state.username, st.session_state.role,
                                                     st.session_state.db_collection)
                    st.session_state.chatty.set_widget({"name": "success", "content": "Stored output in database!"})
            except:
                st.session_state.chatty.set_widget({"name": "error", "content": "Output could not be stored in database!"})
        elif cancel:
            st.session_state.chatty.set_widget({"name": "warning", "content": "Canceled store operation!"})
        st.session_state.ongoing_operation = False
        st.rerun()


def execute_user_actions():
    if st.session_state.ongoing_operation:
        if st.session_state.operation == "store_db":
            store_db()
        elif st.session_state.operation == "generate_dsl":
            generate_dsl()


def login_page():
    st.set_page_config(page_title="Login", page_icon=icon, layout="centered", initial_sidebar_state="collapsed")
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with login_tab:
        login(captcha='--local' not in sys.argv)
    with register_tab:
        register()


def initialize_app():
    if '--local' in sys.argv:
        st.session_state.mongo_db = MongoClient("mongodb://localhost:27017")["local"]
    else:
        st.session_state.mongo_db = MongoClient("mongodb://mongodb:27017/")["local"]
    st.session_state.authentication_status = "login"
    st.session_state.initialized = True


def start_chatty():
    load_userdata()
    save_userdata()
    st.session_state.session_id = str(uuid.uuid4())
    if st.session_state.role != "admin":
        disable_browser_error()
    user_data = st.session_state.mongo_db["user_data"].find_one(filter={"username": st.session_state.username})
    # Initialize Chatbot
    with st.spinner("Initializing Chatty..."):
        if str(datetime.now().date()) in user_data["daily_tokens"]:
            prev_tokens = user_data["daily_tokens"][str(datetime.now().date())]
        else:
            prev_tokens = 0
        if st.session_state.role not in ["wimi", "admin"]:
            assert st.session_state.model != "gpt-4o"
        chatty = Chatty(model=st.session_state.model, total_tokens=prev_tokens, db_host="localhost" if '--local' in sys.argv else "weaviate")
        st.session_state.chatty = chatty
        st.session_state.session_history[st.session_state.session_id] = {"messages": chatty.messages, "time": time.time()}
        save_userdata()
        st.session_state.ongoing_operation = False
        st.session_state.initialized = True
        st.session_state.switched_page = False
        st.session_state.check_syntax = False


def run_app():
    logging.basicConfig(level=logging.INFO)
    if "initialized" not in st.session_state:
        initialize_app()
    if st.session_state.authentication_status != "completed":
        login_page()
        return
    st.set_page_config(page_title="SE-Chatty", page_icon=icon, layout="centered", initial_sidebar_state="auto")
    create_description()
    if st.session_state.first_run:
        start_chatty()
        st.session_state.first_run = False
    if st.session_state.switched_page:
        load_userdata()
        st.session_state.switched_page = False
    create_sidebar()
    msg_ct = st.container()
    with msg_ct:
        create_old_messages()
    if not has_permission():
        return
    user_message = create_input()
    if user_message:
        with msg_ct:
            create_new_message(user_message)

    update_settings()
    execute_user_actions()
    save_userdata()


run_app()
