from flask import Flask, jsonify, request
from chatty import Chatty
from prompts import SYSTEM_PROMPT

app = Flask(__name__)
chatty = Chatty()


@app.route('/api/init')
def get_initial():
    return jsonify({"session": [{"role": "system", "content": SYSTEM_PROMPT}]})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()  # Get JSON data from the request body
    session = data.get('session')
    message = data.get('message')  # Get the 'message' parameter from the JSON data
    rerank = data.get('rerank')
    reformulate = data.get('reformulate')

    chatty.messages = session
    if message:
        response_message, _, _ = chatty.chat(message, rerank=rerank, reformulate=reformulate, security=1, stream=False)
        return jsonify({'response': response_message, 'session': chatty.messages})
    else:
        return jsonify({'error': 'No message provided'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0')
