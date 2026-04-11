from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import os
import json
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

# 会话存储（生产环境建议用 Redis）
sessions = {}

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", "sk-c66f556dac644aa7b40e52f1bda10eee"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ==================== 流式输出接口 ====================
@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', '')
    model = data.get('model', 'qwen-turbo')
    enable_search = data.get('search', True)
    system_prompt = data.get('tone', '你是锐锐，一个友好、专业的AI助手。')
    
    # 创建或获取会话
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "created_at": datetime.now(),
            "messages": [{"role": "system", "content": system_prompt}]
        }
    
    # 添加用户消息
    sessions[session_id]["messages"].append({"role": "user", "content": user_message})
    
    def generate():
        params = {
            "model": model,
            "messages": sessions[session_id]["messages"],
            "temperature": 0.7,
            "stream": True
        }
        
        if enable_search:
            params["extra_body"] = {"enable_search": True}
        
        try:
            stream = client.chat.completions.create(**params)
            full_reply = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_reply += content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            # 保存助手回复到会话
            sessions[session_id]["messages"].append({"role": "assistant", "content": full_reply})
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# ==================== 普通接口（兼容旧版）====================
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', '')
        model = data.get('model', 'qwen-turbo')
        enable_search = data.get('search', True)
        system_prompt = data.get('tone', '你是锐锐，一个友好、专业的AI助手。')
        
        # 创建或获取会话
        if not session_id or session_id not in sessions:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                "created_at": datetime.now(),
                "messages": [{"role": "system", "content": system_prompt}]
            }
        
        # 添加用户消息
        sessions[session_id]["messages"].append({"role": "user", "content": user_message})
        
        params = {
            "model": model,
            "messages": sessions[session_id]["messages"],
            "temperature": 0.7
        }
        
        if enable_search:
            params["extra_body"] = {"enable_search": True}
        
        response = client.chat.completions.create(**params)
        reply = response.choices[0].message.content
        
        # 保存助手回复
        sessions[session_id]["messages"].append({"role": "assistant", "content": reply})
        
        return jsonify({
            "success": True,
            "reply": reply,
            "session_id": session_id
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==================== 健康检查 ====================
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "sessions": len(sessions)})

# ==================== 清空会话 ====================
@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    try:
        session_id = request.get_json().get('session_id', '')
        if session_id in sessions:
            del sessions[session_id]
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
