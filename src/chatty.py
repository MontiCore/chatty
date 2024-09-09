import json
from openai_interface import OpenAIChat
#from huggingface_interface import HFChat
from database_interface import WeaviateInterface
from prompts import *
from utils import *
import time
import logging
import numpy as np
from sentence_transformers import CrossEncoder
import asyncio
import nest_asyncio
from dsl_generators import CD4AGenerator
import os
import copy
nest_asyncio.apply()


class Chatty:
    def __init__(self,
                 model="gpt-3.5-turbo",
                 total_tokens=0,
                 db_host="weaviate"):
        self.model = model
        self.global_settings = {"max_tokens": 1000, "temperature": 0.7, "limit_tokens": False}
        self.num_docs = 3
        self.num_rerank = 50
        self.min_similarity = 0.0
        self.min_rating = 5

        self.min_dsl_similarity = 0.92
        self.num_dsl_dynamic = 3

        self.used_tokens = total_tokens
        self.token_delta = 0
        self.context = 0
        self.context_delta = 0
        self.stored_prompt = None

        if model in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]:
            self.llm = OpenAIChat(self.global_settings, model=model)
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        else:
            raise KeyError(f"Unknown model {model}")

        self.dsl_generators = {"cd4a": CD4AGenerator(llm=self.llm)}


        self.database = WeaviateInterface(embedding_model="text-embedding-ada-002", host=db_host)
        # For reranking, small model for efficient computation. Choose larger model for increase in quality
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512)

    def change_settings(self, key, value):
        if key == "num_docs":
            self.num_docs = value
        elif key == "min_similarity":
            self.min_similarity = value
        elif key == "min_rating":
            self.min_rating = value
        elif key == "num_rerank":
            self.num_rerank = value
        elif key == "min_dsl_similarity":
            self.min_dsl_similarity = value
        elif key == "num_dsl_dynamic":
            self.num_dsl_dynamic = value
        else:
            self.llm.settings[key] = value

    def store_db(self, title, content, user_name, role, collection):
        self.database.insert_db(title, content, user_name, role, collection)

    def set_widget(self, widget, index=-1):
        if "widgets" in self.messages[index].keys():
            self.messages[index]["widgets"].append(widget)
        else:
            self.messages[index]["widgets"] = [widget]

    def calculate_tokens(self):
        # Need to format last user prompt to get correct token usage
        tokens = self.llm.count_tokens([*self.format_messages(self.messages[:-1]), self.messages[-1]])
        logging.info(f"{tokens=}")
        self.token_delta += tokens
        self.used_tokens += self.token_delta
        self.context_delta = tokens - self.context
        self.context = tokens

    def re_rank_docs(self, similar_docs, user_message, num_docs):
        start = time.time()
        content = similar_docs["documents"][0]
        scores = self.cross_encoder.predict([(user_message, doc) for doc in content])
        top_n_indices = np.argpartition(scores, -num_docs)[-num_docs:]
        similar_docs["documents"][0] = [doc for i, doc in enumerate(content) if i in top_n_indices]
        similar_docs["metadatas"][0] = [md for i, md in enumerate(similar_docs["metadatas"][0]) if i in top_n_indices]

        logging.info(f"Reranking time: {time.time()-start:.2f}s")
        return similar_docs

    def reformulate(self, similar_docs, user_message):
        content = similar_docs["documents"][0]
        reformulated_docs = {k: [[]] for k in similar_docs.keys()}
        start = time.time()
        message_list = []
        template = """> Question: {question}
                      > Context:
                      >>>
                      {context}
                      >>>
                  """
        for i in range(len(content)):
            task = template.format(question=user_message, context=content[i])
            message_list.append([{"role": "system", "content": SYSTEM_POSTPROCESS}, {"role": "user", "content": task},
                                 {"role": "assistant", "content": "Extracted relevant parts:"}])
        reply_list = asyncio.run(self.llm.chat_parallel(message_list))
        for i, reply in enumerate(reply_list):
            content, tokens = reply
            self.token_delta += tokens
            for k in reformulated_docs.keys():
                if content == "{no_output_str}":
                    continue
                if k == "documents":
                    reformulated_docs[k][0].append(content)
                elif k == "metadatas":
                    reformulated_docs[k][0].append(similar_docs[k][0][i])
                    if "citations" in similar_docs[k][0][i]:
                        citations = similar_docs[k][0][i]["citations"]
                        citations = [r for r in citations if r.split(":")[0] in content]
                        reformulated_docs[k][0][-1]["citations"] = citations
        logging.info(f"Time used for reparaphrizing: {time.time()-start:.2f} seconds")
        return reformulated_docs

    def format_docs(self, documents):
        metadata = documents["metadatas"][0]
        content = documents["documents"][0]
        formatted_docs = [{'document content': content[i]} for i in range(len(content))]
        meta_dict = {"document_title": "document title",
                     "page_nr": "page number",
                     "paragraph": "paragraph",
                     "chapter": "chapter",
                     "section": "section",
                     "subsection": "subsection",
                     "subsubsection": "subsubsection",
                     "citations": "references",
                     "publish_date": "published",
                     "authors": "authors",
                     "user": "author",
                     "status": 'approval status'}

        for i in range(len(metadata)):
            for k, v in meta_dict.items():
                if k in metadata[i].keys() and metadata[i][k]:
                    # formatted_docs[i] += f"{v}'{metadata[i][k]}'"
                    formatted_docs[i][v] = metadata[i][k]
        doc_dict = {f'document {i + 1}': doc for i, doc in enumerate(reversed(formatted_docs))}
        return doc_dict

    def format_fewshot(self, fewshot):
        metadata = fewshot["metadatas"][0]
        content = fewshot["documents"][0]
        formatted_docs = [{'code': content[i]} for i in range(len(content))]
        meta_dict = {"document_title": "file name",
                     "comment": "comment"}
        for i in range(len(metadata)):
            for k, v in meta_dict.items():
                if k in metadata[i].keys():
                    formatted_docs[i][v] = metadata[i][k]
        fewshot_dict = {f'fewshot example {i + 1}': doc for i, doc in enumerate(reversed(formatted_docs))}
        return fewshot_dict

    def format_messages(self, messages):
        """
        The last previous documents are always provided
        Order matters for most LLMs
        Question supposed to be at the end
        Most relevant documents also at the end
        """
        messages = copy.deepcopy(messages)
        previous_indices = [i for i in range(len(messages)-1) if ("info" in messages[i] or "additional" in messages[i])]
        indices = [-1]
        if len(previous_indices) > 0:
            indices.append(previous_indices[-1])

        for idx in indices:
            if "info" in messages[idx] or "additional" in messages[idx]:
                content = f"Question: {messages[idx]['content']}"
                if "additional" in messages[idx]:
                    additional = messages[idx]['additional']
                    if "documents" in additional:
                        content = f"Documents: {json.dumps(additional['documents'])}\n" \
                                  f"{content}"
                if "info" in messages[idx]:
                    content = f"Information: {messages[idx]['info']}\n" \
                              f"{content}"
                messages[idx] = {"role": messages[idx]["role"], "content": content}
        return messages

    def contextualize(self, user_message, messages):
        """
        Creates a 'context prompt', i.e., a prompt that contextualizes the user prompt with the chat history.
        This allows more accurate retrieval via RAG or better selection of stages
        """
        context_prompt = [{"role": "system", "content": SYSTEM_CONTEXT},
                          *[m for m in messages[1:] if m["role"] == "user"],
                          {"role": "user", "content": user_message}]
        content, tokens = self.llm.chat(context_prompt)
        logging.info(f"Context: {content}")
        self.token_delta += tokens
        return content

    def predict_rag(self, user_message):
        rag_prompt = [{"role": "system", "content": SYSTEM_RAG},
                      {"role": "user", "content": f"Input: {user_message}"},
                      {"role": "assistant", "content": f"Output:\n"}]
        content, tokens = self.llm.chat(rag_prompt, force_json=True)
        self.token_delta += tokens
        logging.info(f"RAG Prediction: {content}")
        try:
            res = json.loads(content)
        except json.decoder.JSONDecodeError:
            logging.warning("Predict RAG returned invalid response")
            return False
        if "answer" not in res:
            res = False
        else:
            res = res["answer"] == "true"
        return res

    def write(self, status, message):
        if status is None:
            print(message)
        else:
            status.write(message)

    def check_syntax(self, dsl, index=-1):
        cor, _, _ = self.dsl_generators[dsl].verify_syntax(self.messages[index]["content"])
        return cor

    def rag_pipeline(self, user_message, contextualized_query, rag="auto", security=1, collections=None, status=None, re_rank=True, post_process=True):
        """
        Retrieval-Augemented Generation Pipeline
        """
        if rag == "auto":
            use_rag = self.predict_rag(user_message)
        else:
            use_rag = True if rag == "on" else False

        doc_dict = {}
        if use_rag:
            self.write(status, "Searching similar documents...")
            similar_docs = self.database.semantic_search(contextualized_query,
                                                         num_docs=self.num_docs if not re_rank else self.num_rerank,
                                                         min_similarity=self.min_similarity, security=security,
                                                         collections=collections)
            if len(similar_docs["documents"][0]) > 0:
                if re_rank:
                    self.write(status, "Reranking documents...")
                    similar_docs = self.re_rank_docs(similar_docs, contextualized_query, self.num_docs)
                if post_process:
                    self.write(status, "Reformulating documents...")
                    similar_docs = self.reformulate(similar_docs, contextualized_query)
                if len(similar_docs["documents"][0]) > 0:
                    doc_dict["documents"] = self.format_docs(similar_docs)
        return doc_dict

    def dsl_questions(self, user_message, dsl, status=None):
        if len(self.messages) > 1:
            self.write(status, "Contextualizing prompt...")
            context_prompt = self.contextualize(user_message, self.messages)
        else:
            context_prompt = user_message
        generator = self.dsl_generators[dsl]
        self.write(status, "Generating questions...")
        questions, tokens = generator.generate_questions(context_prompt)
        self.token_delta += tokens
        return questions, context_prompt


    def dsl_pipeline(self, user_message, context_prompt, questions=None, security=1, stream=False, status=None, collections=None, rag="auto", dsl="cd4a"):
        """
        DSL Generation QA Pipeline
        """
        generator = self.dsl_generators[dsl]
        self.token_delta = 0
        docs = self.rag_pipeline(user_message, context_prompt, rag=rag, security=security, collections=collections, status=status)

        prompt = {"role": "user", "content": user_message}
        if len(docs) > 0:
            prompt["additional"] = docs

        if questions is not None and questions != "":
            prompt["info"] = (f"The user answered following questions for the CD4A creation. "
                              f"Be creative and only use them as additions to your class diagram:\n {questions}")

        self.write(status, "Retrieving dynamic few-shot examples...")
        dynamic_fewshot = self.database.semantic_search(context_prompt, num_docs=self.num_dsl_dynamic, min_similarity=self.min_dsl_similarity,
                                                        collections=[dsl])
        if len(dynamic_fewshot) > 0:
            if "additional" not in prompt:
                prompt["additional"] = {}
            prompt["additional"]["fewshot"] = self.format_fewshot(dynamic_fewshot)
        self.write(status, f"Generating {dsl} model...")
        self.messages.append(prompt)
        response, tokens = generator.generate_dsl(self.format_messages(self.messages), context_prompt, dynamic_fewshot, status=status, stream=stream)
        self.token_delta += tokens
        if not stream:
            self.messages.append({"role": "assistant", "content": response})
            self.calculate_tokens()
        return response


    def qa_pipeline(self, user_message, stream=False, status=None, rag="auto", security=1, collections=None, post_process=True, re_rank=True):
        """
        Default Software Engineering QA Pipeline
        """
        if len(self.messages) > 1:
            self.write(status, "Contextualizing...")
            context_prompt = self.contextualize(user_message, self.messages)
        else:
            context_prompt = user_message
        docs = self.rag_pipeline(user_message, context_prompt, status=status, rag=rag, security=security, collections=collections, post_process=post_process, re_rank=re_rank)
        prompt = {"role": "user", "content": user_message}
        if len(docs) > 0:
            prompt["additional"] = docs
        self.write(status, "Generating final response...")
        self.messages.append(prompt)
        print(f"Message len: {len(self.messages)}")
        response, _ = self.llm.chat(self.format_messages(self.messages), stream=stream)

        if not stream:
            self.messages.append({"role": "assistant", "content": response})
            self.calculate_tokens()
        return response, status
