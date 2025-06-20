

import json
import threading
import torch

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import datetime
import os
from transformers import TextIteratorStreamer
from modelscope import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# ============ 1. 全局加载模型 & 分词器 ============

# 默认模型路径，前端不暴露此项
DEFAULT_MODEL_NAME = "/home/wang/model_cache/Qwen3-14B-FP8"

# 全局变量：分词器 & 模型
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    DEFAULT_MODEL_NAME,
    torch_dtype="auto",
    device_map="cuda"
)

# ============ 2. 辅助函数：把 messages 转换成模型输入的 prompt ============

def build_prompt_from_messages(messages: list, enable_thinking: bool) -> str:
    """
    将 OpenAI 格式的 messages 列表（role/content）转换成模型期望的输入文本。
    enable_thinking 控制是否在 apply_chat_template 时开启思考模式。
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    return text

# ============ 3. Web 接口：/v1/chat/completions ============

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    模仿 OpenAI Chat Completions 接口格式，支持 stream=True 的流式返回，
    并根据前端传入的 enable_thinking 决定是否输出思考过程。
    """
    payload = await request.json()

    # 校验 messages 字段
    if "messages" not in payload or not isinstance(payload["messages"], list):
        raise HTTPException(status_code=400, detail="请求体必须包含 'messages' 字段，且为数组。")

    # 取出模型名（可选），如果不传则使用默认
    model_name = payload.get("model", DEFAULT_MODEL_NAME)

    # 如果要支持动态加载不同模型，这里可以加加载逻辑；此处直接复用全局 model
    # messages 列表
    messages = payload["messages"]

    # 是否流式
    stream = payload.get("stream", False)
    # 取出前端传来的 enable_thinking，默认为 False
    enable_thinking = payload.get("enable_thinking", False)

    # 1) 先把 messages 转成 prompt 文本，传入 enable_thinking
    prompt_text = build_prompt_from_messages(messages, enable_thinking)
    # 2) 分词 & 准备输入张量
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    # ===== 非流式一次性返回 =====
    if not stream:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=payload.get("max_new_tokens", 2048),
        )
        # 跳过 prompt 部分，只保留新生成的 token
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        # 组合成 OpenAI 风格 JSON
        response = {
            "id": "chatcmpl-xxxxxxxx",
            "object": "chat.completion",
            "created": int(torch.time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        return response

    # ===== 流式返回部分 =====

    # 3) 构造一个 TextIteratorStreamer，让模型一路推 token 进来
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # 4) 在子线程里调用 model.generate，往 streamer 里塞数据
    def generate_in_thread():
        model.generate(
            **{
                "input_ids": model_inputs.input_ids,
                "max_new_tokens": payload.get("max_new_tokens", 2048),
                "streamer": streamer
            }
        )

    thread = threading.Thread(target=generate_in_thread)
    thread.start()

    # 5) 定义 generator，把每个新 token 封装成 SSE 格式推送给客户端
    def event_generator():
        for new_str in streamer:
            chunk_dict = {
                "choices": [
                    {
                        "delta": {"content": new_str},
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"

        # 等模型线程跑完，最后输出一个终止事件
        yield "data: [DONE]\n\n"
        thread.join()

    # 6) 返回 StreamingResponse，media_type="text/event-stream"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============ Page Assist 访问时的 /api/tags 接口 ============
@app.get("/api/tags")
async def get_tags():
    """
    当 Page Assist 发起 GET /api/tags 时，我们直接返回一个与 Ollama 原生 /api/tags 相同格式的 JSON，
    这里只假设本地只有一个模型：MODEL_TAG（例如 "Qwen3-14B-FP8"）。
    """
    # 1. “modified_at”：使用当前 UTC 时间（示例）；生产环境可根据模型文件最后修改时间来填写。
    modified_at = datetime.datetime.utcnow().isoformat() + "Z"
    # 2. “size”：如果 DEFAULT_MODEL_NAME 是模型文件路径，可以用 os.path.getsize；这里仅示范 0。
    try:
        size = os.path.getsize(DEFAULT_MODEL_NAME)
    except Exception:
        size = 0
    # 3. “digest”：如不方便计算，可以留空或用占位字符串。
    digest = ""

    # 4. “details”：可以根据实际情况填写；此处示范为简易占位。
    details = {
        "format": "",
        "family": "",
        "families": None,
        "parameter_size": "",
        "quantization_level": ""
    }

    # 最终返回与 Ollama /api/tags 一致的结构：
    return JSONResponse(
        content={
            "models": [
                {
                    "name": 'Qwen3-14B-FP8',
                    "modified_at": modified_at,
                    "size": size,
                    "digest": digest,
                    "details": details
                }
            ]
        }
    )




# ============ 4. 直接用 python 运行时的入口 ============
if __name__ == "__main__":
    import uvicorn
    # 这里 host、port 可以按需修改
    uvicorn.run(app, host="0.0.0.0", port=8000)