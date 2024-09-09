from dotenv import load_dotenv
load_dotenv()
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, Conversation, LlamaTokenizer


# Currently only LLama-2-chat models supported
# TODO: support any hf LLM, maybe with superclass as build_prompt is depending on model?
class HFChat:
    def __init__(self,
                 settings=None,
                 messages=None,
                 model='meta-llama/Llama-2-7b-chat-hf'):
        load_dotenv()
        self.model = LlamaForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        if messages is None:
            self.messages = []
        else:
            self.messages = messages
        self.settings = {}
        self.tokens = {"total": 0, "delta": 0}
        # self.encoding = tiktoken.encoding_for_model(model)
        # self.token_cutoff = 1100  # TODO
        # self.__count_tokens(self.messages)

    def __count_tokens(self, input_message):
        raise NotImplementedError

    def build_prompt(self, messages):
        """
        According to:
        https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        Contradicts:
        https://huggingface.co/papers/2307.09288#651d184b312b6fd1c6f49bdb
        """
        prompt = "<s>[INST] "
        if messages[0]['role'] == "system":
            prompt += f"<<SYS>> {messages[0]['content']} <</SYS>> "
            messages = messages[1:]
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if i == 0:
                    prompt += f"{msg['content']} [/INST] "
                else:
                    prompt += f"<s>[INST] {msg['content']} [/INST] "
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']} </s>"
            else:
                raise KeyError(
                    f"Message role should be either 'user' or 'assistant' (or 'system' for first message), not {msg['role']}.")
        return prompt


    def chat(self, input_message):
        self.messages.append(input_message)
        input_message = [{"role": i["role"], "content": i["content"]
                         if (i != self.messages[-1] or "additional" not in i.keys())
                         else i["content"] + i["additional"]} for i in self.messages]
        prompt = self.build_prompt(input_message)
        inputs = self.tokenizer(prompt)
        input_len = len(self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False))
        output = self.model.generate(torch.tensor([inputs['input_ids']]), output_attentions=True, return_dict_in_generate=True)
        output_ids = output['sequences'][0]
        output_attentions = output['attentions']
        output_tokens = self.tokenizer.convert_ids_to_tokens(output_ids)
        output_string = ''.join(output_tokens[input_len:-1])
        output_message = output_string.replace("▁", " ")

        self.messages.append({"role": "assistant", "content": output_message})
        # self.__count_tokens(input_message)
        return output_message
