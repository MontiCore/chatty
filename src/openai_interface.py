from dotenv import load_dotenv
import os
import json
import openai
from openai import AzureOpenAI, AsyncAzureOpenAI
import tiktoken
import asyncio
from chat_interface import ChatInterface
import logging
import nest_asyncio
nest_asyncio.apply()


class OpenAIChat(ChatInterface):
    def __init__(self,
                 settings,
                 model='gpt-3.5-turbo'):
        super().__init__()
        model_cfg = {"gpt-3.5-turbo": {"deployment": "gpt-35-turbo", "endpoint": "sweden"},
                     "gpt-4o-mini": {"deployment": "gpt-4o-mini", "endpoint": "sweden"},
                     "gpt-4o": {"deployment": "gpt-4o", "endpoint": "luknet"}}
        cfg = model_cfg[model]
        load_dotenv()
        self.client = AzureOpenAI(azure_endpoint=os.getenv(f"{cfg['endpoint'].upper()}_OPENAI_ENDPOINT"),
                                  azure_deployment=cfg["deployment"],
                                  api_version='2023-05-15',
                                  api_key=os.getenv(f"{cfg['endpoint'].upper()}_OPENAI_KEY"))
        self.async_client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv(f"{cfg['endpoint'].upper()}_OPENAI_ENDPOINT"),
            azure_deployment=cfg["deployment"],
            api_version='2023-05-15',
            api_key=os.getenv(f"{cfg['endpoint'].upper()}_OPENAI_KEY")
        )
        self.settings = settings
        self.encoding = tiktoken.encoding_for_model(model)
        self.model = model

    def count_tokens(self, messages):
        # Only an estimate
        # Here input and output tokens are calculated the same, however openai has a different billing for input/output
        tokens_per_message = 3
        tokens = 0
        for idx, msg in enumerate(messages):
            tokens += tokens_per_message
            tokens += len(self.encoding.encode(msg["content"]))
        tokens += 3
        return tokens

    async def chat_async(self, message):
        response = await self.async_client.chat.completions.create(model=self.model,
                                                                   messages=message,
                                                                   temperature=self.settings["temperature"],
                                                                   n=1,
                                                                   stream=False)
        return response.choices[0].message.content, response.usage.total_tokens

    async def chat_parallel(self, message_list):
        responses = [self.chat_async(m) for m in message_list]
        return await asyncio.gather(*responses)

    def chat(self, messages, stream=False, force_json=False):
        chat_message = [{"role": i["role"], "content": i["content"]} for i in messages]

        try:
            response = self.client.chat.completions.create(model=self.model,
                                                           messages=chat_message,
                                                           max_tokens=int(self.settings["max_tokens"]) if self.settings["limit_tokens"] else None,
                                                           temperature=self.settings["temperature"],
                                                           response_format={"type": "json_object" if force_json else "text"},
                                                           n=1,
                                                           stream=stream)
        except openai.APIError as e:
            response = "Error from the OpenAI API, try again later..."
            logging.error(f"OpenAI API Error: {e}")
            return response, 0
        if not stream:
            tokens = response.usage.total_tokens
            response = response.choices[0].message
            response = response.content
        else:
            tokens = 0
        return response, tokens
