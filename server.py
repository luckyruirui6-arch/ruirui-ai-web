import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import os
import json
import uuid
from datetime import datetime, timedelta
import PyPDF2
import docx

app = Flask(__name__)
CORS(app)

# 会话存储（生产环境建议用 Redis）
sessions = {}

# ⚠️ 安全修复：API Key 必须通过环境变量传入，不要在代码里硬编码
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置，请先配置后再启动服务")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ==================== 会话清理（TTL 2小时）====================
def cleanup_sessions():
    """清理超过2小时未活动的会话，防止内存泄漏"""
    now = datetime.now()
    expired = [sid for sid, s in sessions.items()
               if now - s["created_at"] > timedelta(hours=2)]
    for sid in expired:
        del sessions[sid]
    return len(expired)

# ==================== 文件上传接口 ====================
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "没有上传文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "文件名为空"}), 400
        
        filename = file.filename
        file_ext = filename.split('.')[-1].lower()
        
        content = ""
        
        # 处理 PDF
        if file_ext == 'pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(file.stream)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            except Exception as e:
                return jsonify({"success": False, "error": f"PDF解析失败: {str(e)}"}), 400
        
        # 处理 Word
        elif file_ext == 'docx':
            try:
                doc = docx.Document(file.stream)
                for para in doc.paragraphs:
                    if para.text:
                        content += para.text + "\n"
            except Exception as e:
                return jsonify({"success": False, "error": f"Word解析失败: {str(e)}"}), 400
        
        # 处理 TXT / MD
        elif file_ext in ('txt', 'md'):
            try:
                content = file.read().decode('utf-8')
            except UnicodeDecodeError:
                try:
                    file.stream.seek(0)
                    content = file.read().decode('gbk')
                except Exception as e:
                    return jsonify({"success": False, "error": f"TXT解码失败: {str(e)}"}), 400
        
        else:
            return jsonify({"success": False, "error": f"不支持的文件类型: {file_ext}，仅支持 .txt .md .pdf .docx"}), 400
        
        if not content or not content.strip():
            return jsonify({"success": False, "error": "文件内容为空或无法解析"}), 400
        
        # 限制内容长度（防止 token 过多）
        if len(content) > 8000:
            content = content[:8000] + "\n\n... 内容已截断，文件太大仅显示前8000字符"
        
        return jsonify({
            "success": True,
            "filename": filename,
            "content": content,
            "length": len(content)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==================== 流式输出接口 ====================
@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', '')
    model = data.get('model', 'qwen-turbo')
    enable_search = data.get('search', True)
    system_prompt = data.get('tone', '你是锐锐，一个友好、专业的AI助手。')
    
    if not user_message:
        return jsonify({"success": False, "error": "消息不能为空"}), 400

    # 创建或获取会话
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "created_at": datetime.now(),
            "messages": [{"role": "system", "content": system_prompt}]
        }
    else:
        # 刷新活跃时间
        sessions[session_id]["created_at"] = datetime.now()
    
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
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
            
            # 保存助手回复到会话
            sessions[session_id]["messages"].append({"role": "assistant", "content": full_reply})
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
    
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

        if not user_message:
            return jsonify({"success": False, "error": "消息不能为空"}), 400
        
        # 创建或获取会话
        if not session_id or session_id not in sessions:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                "created_at": datetime.now(),
                "messages": [{"role": "system", "content": system_prompt}]
            }
        else:
            sessions[session_id]["created_at"] = datetime.now()
        
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

# ==================== 健康检查（顺带清理过期会话）====================
@app.route('/api/health', methods=['GET'])
def health():
    cleaned = cleanup_sessions()
    return jsonify({
        "status": "ok",
        "sessions": len(sessions),
        "cleaned": cleaned
    })

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
