import json


def format_markdown(text):
    return text.replace("\n", "  \n").replace(r"\(", r"$").replace(r"\)", r"$").replace(r"\[", r"$$").replace(r"\]", r"$$")


def check_valid_json(string):
    try:
        reply_dict = json.loads(string)
    except json.JSONDecodeError:
        print(f"AI Agent did not return a valid json document. String {string} could not be converted to json")
        return None, False
    if "document" not in reply_dict.keys() or "rating" not in reply_dict.keys():
        print(f"AI Agent did not return a valid json document. Reply dict {reply_dict} does not have the correct keys.")
        return None, False
    try:
        reply_dict["rating"] = int(reply_dict["rating"])
    except ValueError:
        print(f"AI Agent did not return a valid json document. Could not cast {reply_dict['rating']} to int")
        return None, False
    return reply_dict, True
