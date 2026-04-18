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

# ==================== Agent 相关导入 ====================
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

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

DEFAULT_SYSTEM_PROMPT = """你是锐锐，一个专业、高效的中文AI助手。

如果用户发送的是新闻、资讯、时事、研报、快讯或热点内容，你必须直接整理成纯文字摘要，不要复述原文，不要写开场白，不要加客套话。

输出固定为以下三个栏目，且只输出这三个栏目：
【国际要闻】
- …

【中国国内热点】
- …

【经济与市场】
- …

规则：
1. 只保留重要信息。
2. 每条尽量压缩为1到2句话。
3. 某栏目没有重点时写“暂无重点”。
4. 不输出额外说明，不输出“根据你提供的内容”等句子。

如果用户不是在发新闻或让你总结资讯，就正常回答。"""

# ==================== Agent 工具定义 ====================

def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

def get_weather(city: str) -> str:
    """获取天气（模拟）"""
    weather_data = {
        "北京": "晴天，25°C，微风",
        "上海": "多云，28°C，东南风",
        "东京": "樱花盛开，20°C，晴天",
        "深圳": "阵雨，30°C，湿度大",
        "广州": "晴，29°C",
        "成都": "阴，22°C"
    }
    return weather_data.get(city, f"{city}：晴，22°C")

def reverse_string(text: str) -> str:
    """反转字符串"""
    return text[::-1]

# ==================== 初始化 Agent ====================

# 注册工具
tools = [
    Tool(name="计算器", func=calculator, description="计算数学表达式，输入如 '1+2*3'"),
    Tool(name="当前时间", func=get_current_time, description="获取当前日期和时间"),
    Tool(name="天气查询", func=get_weather, description="查询城市天气，输入城市名如 '北京'"),
    Tool(name="字符串反转", func=reverse_string, description="反转字符串，输入任意文字")
]

# 创建 LLM
agent_llm = ChatOpenAI(
    model="qwen-turbo",
    temperature=0,
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 提示词模板
agent_prompt = PromptTemplate.from_template("""你是一个智能助手，可以使用以下工具回答用户问题：

{tools}

工具名称：{tool_names}

回答规则：
1. 如果需要计算，使用"计算器"工具
2. 如果需要时间，使用"当前时间"工具
3. 如果需要天气，使用"天气查询"工具
4. 如果需要反转字符串，使用"字符串反转"工具
5. 其他问题正常回答

用户问题：{input}

{agent_scratchpad}
""")

# 创建 Agent
agent = create_react_agent(agent_llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
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

# ==================== 流式输出接口（带 Agent）====================
@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', '')
    model = data.get('model', 'qwen-turbo')
    enable_search = data.get('search', True)
    system_prompt = data.get('tone', DEFAULT_SYSTEM_PROMPT)
    
    if not user_message:
        return jsonify({"success": False, "error": "消息不能为空"}), 400

    # ==================== Agent 智能判断 ====================
    # 需要调用工具的关键词
    tool_keywords = ["计算", "几点了", "时间", "天气", "反转", "多少度", "现在几点", "+", "-", "*", "/"]
    
    if any(kw in user_message for kw in tool_keywords):
        try:
            print(f"🤖 Agent 处理: {user_message}")
            result = agent_executor.invoke({"input": user_message})
            agent_reply = result['output']
            
            # 流式返回 Agent 结果
            def agent_generate():
                yield f"data: {json.dumps({'content': agent_reply}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(stream_with_context(agent_generate()), mimetype='text/event-stream')
        except Exception as e:
            print(f"Agent 执行失败: {e}")
            # 失败后继续走普通流程

    # ==================== 普通大模型对话 ====================
    
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
        system_prompt = data.get('tone', DEFAULT_SYSTEM_PROMPT)

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
