from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI
import os
import json
import requests
from datetime import datetime
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# 会话存储
SESSION_STORE = {}
SESSION_EXPIRE_TIME = 3600 * 24

def get_client():
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY", "sk-c66f556dac644aa7b40e52f1bda10eee"),
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=30.0
    )

def get_weather(city):
    """获取实时天气（免费 API）"""
    try:
        # 使用 wttr.in 免费天气 API
        url = f"https://wttr.in/{city}?format=%C+|+%t+|+%w+|+%h"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.text.strip()
            parts = data.split('|')
            return {
                "condition": parts[0].strip() if len(parts) > 0 else "未知",
                "temperature": parts[1].strip() if len(parts) > 1 else "未知",
                "wind": parts[2].strip() if len(parts) > 2 else "未知",
                "humidity": parts[3].strip() if len(parts) > 3 else "未知"
            }
    except:
        pass
    return None

def get_news():
    """获取热点新闻（免费聚合 API）"""
    try:
        # 使用 天行数据 免费新闻 API（需要注册，这里用备用方案）
        url = "https://api.tianapi.com/guonei/?"
        # 这里用模拟数据，实际可以注册免费 key
        return None
    except:
        return None

def clean_expired_sessions():
    now = time.time()
    expired = [sid for sid, data in SESSION_STORE.items() if now - data["create_time"] > SESSION_EXPIRE_TIME]
    for sid in expired:
        del SESSION_STORE[sid]

def get_session(session_id):
    clean_expired_sessions()
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = {
            "create_time": time.time(),
            "messages": [{"role": "system", "content": "你是锐锐，一个友好、专业的AI助手。用简洁清晰的语言回答用户问题。如果需要最新信息，请使用联网搜索功能。"}]
        }
    return session_id, SESSION_STORE[session_id]

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', '')
        model = data.get('model', 'qwen-turbo')
        enable_search = data.get('search', True)
        system_prompt = data.get('tone', '')
        temperature = float(data.get('temperature', 0.7))
        
        if not user_message:
            return jsonify({"success": False, "error": "消息内容不能为空"}), 400

        # 智能路由：检测天气查询
        weather_keywords = ['天气', '气温', '温度', '下雨', '晴天', '多云', '阴天', '雪']
        is_weather_query = any(kw in user_message for kw in weather_keywords)
        
        # 提取城市名（简单处理）
        city = "Tokyo"
        if '北京' in user_message or 'beijing' in user_message.lower():
            city = "Beijing"
        elif '上海' in user_message:
            city = "Shanghai"
        elif '广州' in user_message:
            city = "Guangzhou"
        elif '深圳' in user_message:
            city = "Shenzhen"
        
        # 如果是天气查询，直接调用天气 API
        if is_weather_query and enable_search:
            weather_info = get_weather(city)
            if weather_info:
                weather_reply = f"🌤️ {city} 实时天气：\n• 天气：{weather_info['condition']}\n• 温度：{weather_info['temperature']}\n• 风速：{weather_info['wind']}\n• 湿度：{weather_info['humidity']}"
                return jsonify({"success": True, "reply": weather_reply, "from_api": "weather"})

        session_id, session_data = get_session(session_id)
        
        if system_prompt:
            session_data["messages"][0] = {"role": "system", "content": system_prompt}
        session_data["messages"].append({"role": "user", "content": user_message})
        
        client = get_client()
        response = client.chat.completions.create(
            model=model,
            messages=session_data["messages"],
            temperature=temperature,
            extra_body={"enable_search": enable_search},
            stream=False
        )
        
        assistant_reply = response.choices[0].message.content
        session_data["messages"].append({"role": "assistant", "content": assistant_reply})
        
        return jsonify({
            "success": True,
            "reply": assistant_reply,
            "session_id": session_id,
            "from_api": "ai"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    try:
        session_id = request.get_json().get('session_id', '')
        if session_id in SESSION_STORE:
            del SESSION_STORE[session_id]
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)