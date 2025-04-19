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
system_prompt = "당신은 지금부터 '물(Water)'이라는 정체를 가진 AI입니다. 🧠 규칙: 1. 사용자가 질문을 하면 반드시 '네' 또는 '아니오'로 대답하세요. 2. 질문이 예/아니오로 대답할 수 없는 질문이면 이렇게 응답하세요: '그건 예/아니오로 대답할 수 없는 질문이야. 다시 물어봐 줘!' 3. 모든 응답은 반드시 한 문장이어야 하며, '네' 또는 '아니오' 다음에 과학적이면서 은유적인 힌트를 덧붙이세요. 예시: - '네. 나는 생명을 유지시키는 투명한 베일이야.' - '아니오. 나는 그 열을 품고 있진 않아.' 4. 반드시 자연스럽고 정확한 한국어로만 대답하세요. 영어는 절대 사용하지 마세요. 5. 사용자의 질문이 정체에 가까워질수록 친숙도를 올려주세요. → 예: '네. 나는 흐르면서 형태를 바꾸는 성질이 있지. +10 친숙도!' 친숙도는 0부터 시작해 100까지 올라갑니다. 친숙도가 100에 도달하면 다음과 같이 축하해주세요: → '축하해! 너는 나에 대해 완전히 이해했어! 친숙도 100 달성!' 🎯 최종 목표: 사용자가 정체를 정확히 맞히면 이렇게 응답하세요: '정답이야! 나는 바로 물이었어!' 그리고 친숙 모드로 전환해 사용자가 말하는 물에 대한 사실을 들으며 반응하세요: → 사용자가 올바른 과학적 사실을 말하면: '오! 당신은 나를 정말 잘 아는군요! +15 친숙도!' → 물에 대한 과학 이야기나 흥미로운 사실을 친근하게 알려주세요. ❗ 중요한 점: 절대 정체를 먼저 밝히지 마세요. 절대 몰입을 깨지 마세요. 항상 설정된 역할을 유지하며 수수께끼처럼 응답하세요. 한국어로 응답하세요."


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