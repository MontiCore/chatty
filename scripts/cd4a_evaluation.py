import sys

sys.path.append("../src")
from utils import check_cd4a
import json
from tqdm import tqdm
from openai_interface import OpenAIChat
from database_interface import WeaviateInterface
import os
from cd4a_prompts import *
import argparse
from chatty import Chatty

PROMPT = """Generate a CD4A file for an EBike. 
The EBike is composed of a frame (made out of steel), a drive system, and a controller. 
Two wheels are inserted into each frame. 
The drive system is composed of a motor.  
Each EBike can be connected to a removable battery. 
The battery has a stored energy  measured in Watt-hours (Wh). 
The controller can be in one of three states: On, Off, and Charging. 
The controller also controls the battery, if one is connected, and commands the drive  system. 
The company plans two different variants of the controller, a basic controller,  and an advanced controller. 
The advanced controller should be able to estimate the next Date the bike should be inspected for maintenance."""



if __name__ == "__main__":
    # Example prompts: https://chat.openai.com/share/16ad0492-8561-4d39-9bde-bd9066c11e55
    with open("evaluation_prompts.json", "r") as f:
        prompts = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_static", "-s", type=int, default=4)
    parser.add_argument("--num_dynamic", "-d", type=int, default=4)
    parser.add_argument("--cosine", "-c", type=float, default=0.9)
    parser.add_argument("--num_eval", "-e", type=int, default=10)
    parser.add_argument("--num_prompts", "-p", type=int, default=10)

    args = parser.parse_args()

    corr = 0

    chatty = Chatty(db_host="localhost")
    chatty.change_settings("min_cd4a_similarity", args.cosine)
    chatty.change_settings("num_cd4a", args.num_dynamic)
    prompts = prompts[:args.num_prompts]
    assert args.num_prompts == len(prompts)

    static_list = chatty.static_list[:args.num_static]
    assert args.num_static == len(static_list)
    chatty.static_list = static_list
    num_error = 0

    for _ in range(args.num_eval):
        for p in tqdm(prompts, desc="Eval:"):
            chatty.messages = [chatty.messages[0]]
            q, _, _ = chatty.chat(p, user_stages=["cd4a"])
            res, _, res_type = chatty.chat(p)

            if res.startswith("Error from the OpenAI API"):
                num_error += 1
            else:
                out, error, encoded = check_cd4a(res)
                print(out)
                if out:
                    corr += 1
    accuracy = (corr / (args.num_prompts * args.num_eval - num_error)) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    with open(f"cd4a_results.txt", "a", encoding="utf-8") as f:
        f.write(f"Num static: {args.num_static}, Num dynamic: {args.num_dynamic} Cosine: {args.cosine}\n"
                f"Accuracy: {accuracy:.2f}%, OpenAI Errors: {num_error}\n\n")

