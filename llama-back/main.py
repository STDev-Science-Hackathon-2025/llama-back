# main.py
from fastapi import FastAPI, Request
from llama_cpp import Llama
from pydantic import BaseModel
import os
from typing import List

app = FastAPI()

# 모델 경로 및 다운로드 설정
MODEL_PATH = "llama-2-7b-chat.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    os.system(f"wget {MODEL_URL} -O {MODEL_PATH}")

# 모델 로딩
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8)

# 시스템 프롬프트 (한글 응답 포함)
system_prompt = "너는 탄소(c), 수소(h) 중 하나이야. 한국어로 대답해. "

# 과학적 배경 지식 (RAG)
retrieved_context = (
    # "물(Water)은 고체(얼음), 액체(물), 기체(수증기)라는 세 가지 상태로 자연스럽게 존재하는 독특한 물질입니다. "
    # "지구 표면의 약 71%를 덮고 있으며, 인간 신체의 약 60%를 구성할 정도로 생명 유지에 필수적인 역할을 합니다. "
    # "화학적으로 물 분자(H₂O)는 산소 원자 하나와 수소 원자 두 개가 공유 결합된 구조로, 약 104.5도의 굽은 형태를 가지고 있습니다. "
    # "이 구조는 산소의 전기음성도와 함께 물을 극성 분자로 만들며, 이로 인해 물 분자들 사이에 수소 결합이 형성됩니다."
)

class HistoryItem(BaseModel):
    question: str
    answer: str

class PromptRequest(BaseModel):
    prompt: str
    history: List[HistoryItem]

class PromptResponse(BaseModel):
    history: List[dict]
    latest: str

class PromptResponse1(BaseModel):
    latest: str

@app.post("/generate", response_model=PromptResponse)
async def generate(req: PromptRequest):
    user_input = req.prompt
    history = req.history

    # 프롬프트 조립
    history_prompt = ""
    for item in history:
        history_prompt += f"Q: {item.question}\nA: {item.answer}\n"

    full_prompt = (
        f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"[CONTEXT]\n{retrieved_context}\n\n"
        f"{history_prompt}"
        f"Q: {user_input}\nA: [/INST]"
    )

    result = llm(full_prompt, max_tokens=80)
    response = result["choices"][0]["text"].strip()

    updated_history = history + [{"question": user_input, "answer": response}]

    return {
        "history": updated_history,
        "latest": response
    }

class PromptRequest1(BaseModel):
    system_prompt: str
    retrieved_prompt: str
    prompt: str

@app.post("/test", response_model=PromptResponse1)
async def test(req: PromptRequest1):
    retrieved_prompt = req.retrieved_prompt
    system_prompt = req.system_prompt
    user_input = req.prompt

    # 프롬프트 생성
    full_prompt = (
        f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"[CONTEXT]\n{retrieved_prompt}\n\n"
        f"Q: {user_input}\nA: [/INST]"
    )

    result = llm(full_prompt, max_tokens=80)
    response = result["choices"][0]["text"].strip()

    return {
        "latest": response  # ✅ history 제거
    }