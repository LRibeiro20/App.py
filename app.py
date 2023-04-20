from flask import Flask, request, jsonify
import bot
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/chatbot', methods=['POST'])
def chatbot():
    text = request.json['message']
    response = bot.get_response(text)
    return jsonify({'message': str(response)})

if __name__ == '__main__':
    app.run(debug=True)