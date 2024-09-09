from abc import ABC, abstractmethod
import os
import logging
import subprocess
import json
from prompts import CD4A_SYSTEM, CD4A_QUESTION, CD4A_CLASSES, CD4A_ATTRIBUTES, CD4A_RELATIONSHIP, CD4A_GENERATE


class DSLGenerator(ABC):
    def __init__(self, llm):
        self.llm = llm

    @abstractmethod
    def generate_questions(self, context_prompt: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def generate_dsl(self, messages: list, context_prompt: str, dynamic_few_shot: dict, stream: bool, status: object) -> (str, int):
        raise NotImplementedError

    @abstractmethod
    def verify_syntax(self, model: str) -> bool:
        raise NotImplementedError


class CD4AGenerator(DSLGenerator):
    def __init__(self, llm):
        super().__init__(llm)
        # Static examples for CD4A generation
        self.static_list = []
        cd4a_static_dir = "../src/app_data/dsl/cd4a"
        for file in os.listdir(cd4a_static_dir):
            with open(f"{cd4a_static_dir}/{file}", "r") as f:
                cd4a = f.read()
                self.static_list.append(cd4a)


    def generate_questions(self, context_prompt):
        messages = [{"role": "system", "content": CD4A_QUESTION},
                    {"role": "user", "content": f"User: {context_prompt}"},
                    {"role": "assistant", "content": "Answer:"}]

        content, tokens = self.llm.chat(messages, force_json=True)
        logging.info(f"Returned content: {content}")
        try:
            res = json.loads(content.strip("`"))
        except json.decoder.JSONDecodeError:
            logging.warning("Get questions returned invalid response")
            res = None
        return res, tokens


    def generate_dsl(self, messages, context_prompt, dynamic_few_shot, stream=False, status=None):
        # This way few-shot examples are only included in the last request
        # This is intended behavior but might not be beneficial
        used_tokens = 0

        user_message = f"Question: {messages[-1]}"
        messages = [{"role": "system", "content": CD4A_SYSTEM},
                    *messages[1:-1],
                    {"role": "user", "content": user_message}]
        # self.write(status, "Generating classes, attributes and relations...")
        for task in [CD4A_CLASSES, CD4A_ATTRIBUTES, CD4A_RELATIONSHIP]:
            messages.append({"role": "user", "content": task})
            answer, tokens = self.llm.chat(messages)
            used_tokens += tokens
            messages.append({"role": "assistant", "content": answer})

        cd4a_message = CD4A_GENERATE
        for i, s in enumerate([*self.static_list, *dynamic_few_shot["documents"][0]]):
            cd4a_message += (f"Example {i}:\n"
                             f"CD4A Code:```\n{s}\n```")
        messages.append({"role": "user", "content": cd4a_message})
        res, _ = self.llm.chat(messages, stream=stream)
        return res, used_tokens

    def verify_syntax(self, model: str) -> bool:
        if "import" in model:
            index = model.find("import")
            diagram = model[index:]
        elif "classdiagram" in model:
            index = model.find("classdiagram")
            diagram = model[index:]
        else:
            logging.warning("No valid response format")
            logging.warning(f"Invalid response:\n{model}")
            return False, None, None
        if len(diagram.split("```")) > 1:
            diagram = diagram.split("```")[0]
        command = ["java", "-jar", "../MCCD.jar", "--stdin"]
        stripped = diagram.replace("```", "").replace("(extends)", "(isExtending)").replace("(implements)",
                                                                                            "(isImplementing)").strip()
        diagram = "import java.lang.*;\n" + "import java.util.*;\n" + stripped  # Import dependencies
        encoded = diagram.encode('utf-8')
        try:
            subprocess.run(command, input=encoded, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True, None, encoded.decode("utf-8")
        except subprocess.CalledProcessError as e:
            return False, e.stderr.decode("latin-1", "replace"), encoded.decode("utf-8")