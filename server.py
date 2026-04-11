from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", "sk-c66f556dac644aa7b40e52f1bda10eee"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        model = data.get('model', 'qwen-turbo')
        enable_search = data.get('search', True)
        system_prompt = data.get('tone', '你是锐锐，一个友好、专业的AI助手。')
        
        if not user_message:
            return jsonify({"success": False, "error": "消息不能为空"}), 400
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": 0.7
        }
        
        if enable_search:
            params["extra_body"] = {"enable_search": True}
        
        response = client.chat.completions.create(**params)
        reply = response.choices[0].message.content
        
        return jsonify({"success": True, "reply": reply})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
