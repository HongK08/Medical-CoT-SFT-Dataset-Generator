#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import itertools
import requests
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ==========================================
# [설정]
# ==========================================
PORTS = [22134]
MODEL = "gpt-oss:120b"

BASE_DIR = "/home/HongKi-Arch/Desktop/LLM_DATASET_Project/Data"
INPUT_FILE = os.path.join(BASE_DIR, "scenarios.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "medical_chat_data.jsonl")

# 생성 옵션
MAX_CASES: Optional[int] = None 
RANDOM_SEED: Optional[int] = 42
OVERWRITE_OUTPUT = False
AUTOSAVE_EVERY = 10

# LLM 파라미터
PROFILE_TEMP = 0.70       # 약간 낮춤 (안정성)
DIALOGUE_TEMP = 0.55      # 약간 낮춤 (포맷 준수)
REPAIR_TEMP = 0.30        # 수선은 확실하게
TIMEOUT = 600
RETRIES = 3

cycle = itertools.cycle(PORTS)

# ==========================================
# [다양성(Persona) 설정]
# ==========================================
USER_STYLES = [
    "Standard: 묻는 말에 적절히 대답함.",
    "Passive: 단답형으로 짧게 대답함 (정보를 조금씩 줌).",
    "Talkative: 질문 하나에 여러 정보를 섞어서 길게 대답함.",
    "Anxious: 증상을 걱정하며 되묻거나 불안해함.",
    "Vague: 표현이 모호하고 정확하지 않음 ('그냥 좀 이상해요')."
]

DOCTOR_STYLES = [
    "Standard: 표준적인 문진 (OPQRST 순서).",
    "Risk-Focused: 위험 징후(Red Flag)부터 먼저 확인.",
    "Empathetic: 공감하며 대화, 환자의 생활 습관도 물어봄.",
    "Efficient: 핵심만 빠르게 질문하여 감별 진단."
]

# HPI intent 목록
HPI_INTENTS = {"onset", "location", "severity", "quality", "aggravating", "relieving", "associated"}

# ==========================================
# [완벽을 위한 프롬프트 튜닝]
# ==========================================
PROFILE_PROMPT_TPL = """
당신은 의학 시나리오 작가입니다.
아래 정보를 바탕으로 가상의 환자 프로필(JSON)을 작성하십시오.

[입력 정보]
- 진료과: {category}
- 주호소: {complaint}
- 위험도: {risk} ({diagnosis_guess})

[작성 규칙]
1. 나이, 성별, 과거력(history), 복용약(meds)을 구체적이고 현실적으로 설정.
2. 증상의 OPQRST(Onset, Provocation, Quality, Region, Severity, Time)를 확정.
3. 위험도 '{risk}'에 맞는 동반 증상 및 위험 징후(Red Flag) 포함 여부 결정.
4. red_flag_symptoms는 절대 빈 문자열로 두지 말 것. (없으면 "없음")
5. 출력은 JSON 포맷만.

[출력 예시]
{{
  "profile": {{ "age": 45, "gender": "M", "history": "고혈압", "meds": "아몰디핀" }},
  "symptoms": {{
    "chief_complaint": "...",
    "onset": "...",
    "location": "...",
    "severity": 5,
    "quality": "...",
    "associated_symptoms": "...",
    "aggravating_factors": "...",
    "relieving_factors": "...",
    "red_flag_symptoms": "없음"
  }}
}}
""".strip()

# ★★★ 여기가 핵심 변경 구간 (Anti-Telepathy Rules) ★★★
DIALOGUE_PROMPT_TPL = """
당신은 '의료 문진 시뮬레이터'입니다.
환자 프로필을 바탕으로 의사(Assistant)와 환자(User)의 대화를 생성하십시오.

[환자 프로필]
{profile_json}

[설정]
- 의사 스타일: {doctor_style}
- 환자 스타일: {user_style}

[절대 규칙 (Data Leakage 방지)]
1. Assistant는 대화 시작 시점에 환자의 구체적인 정보(과거력, 복용약, 세부 증상)를 **전혀 모른다고 가정**해야 한다.
2. 따라서 Assistant는 **프로필에 있는 병명이나 약물명을 먼저 언급해서는 안 된다.**
   - (X) "당뇨약은 드시고 계신가요?" (프로필을 훔쳐본 질문)
   - (O) "평소 앓고 있는 지병이나 드시는 약이 있나요?" (올바른 질문)
3. User는 Assistant가 '포괄적인 질문(Open-ended question)'을 했을 때, 비로소 프로필의 정보를 구체적으로 답변한다.

[대화 생성 규칙]
1. Assistant는 매 턴 `thought`, `intent`, `content` 필수.
2. Assistant는 한 턴에 질문 1개만. (물음표 '?'는 1개만 사용)
3. Assistant와 User는 반드시 번갈아가며 등장.
4. HPI(onset, location, severity, quality 등)를 충분히 수집 후, 마지막에 intent="summary"로 종료.
5. Summary에서는 **대화 중에 User가 직접 말한 내용**만 요약해야 한다. (묻지 않은 정보 포함 금지)
6. 출력은 JSON 포맷만.
7. HPI 수집 후 Summary로 넘어가기 전에, 반드시 '과거력(History)'과 '약물(Meds)'을 확인하는 질문을 해야 한다.

[출력 예시]
{{
  "dialogue": [
    {{
      "role": "assistant",
      "thought": "주호소 확인",
      "intent": "onset",
      "content": "어디가 불편하신가요?"
    }},
    {{
      "role": "user",
      "content": "배가 아파요."
    }},
    {{
      "role": "assistant",
      "thought": "과거력 확인 (구체적 병명 언급 금지)",
      "intent": "history",
      "content": "혹시 예전부터 앓고 계신 다른 질환이 있나요?"
    }},
    {{
      "role": "user",
      "content": "네, 고혈압이랑 당뇨가 있어요."
    }}
  ]
}}
""".strip()

# Repair용 프롬프트들도 동일하게 유지하되, 규칙 강화
REPAIR_DIALOGUE_PROMPT_TPL = """
너는 의료 문진 데이터 편집기다.
아래 대화(JSON)를 규칙에 맞게 "다시 작성"해라.

[환자 프로필]
{profile_json}

[원본 대화]
{dialogue_json}

[수정 규칙]
1. Assistant가 프로필의 병명/약물을 먼저 말하는 '텔레파시 오류'가 있다면, "앓고 있는 질환이 있나요?" 같은 포괄적 질문으로 고쳐라.
2. Assistant와 User는 번갈아 등장.
3. Assistant 질문은 한 턴에 1개만.
4. 마지막은 summary로 종료.
5. 출력은 JSON만.

[출력 포맷(JSON)]
{{ "dialogue": [ ... ] }}
""".strip()

# ... (나머지 APPEND_SUMMARY_PROMPT, ALLOWED_INTENTS 등은 그대로 유지) ...
APPEND_SUMMARY_PROMPT = """
아래 대화의 마지막에 의사(Assistant)의 '요약 및 권고(summary)' 턴을 추가하여 JSON을 완성하라.

[환자 프로필]
{profile_json}

[현재 대화]
{dialogue_json}

[규칙]
1. 요약 내용은 대화에서 환자가 실제로 언급한 내용에 기반해야 한다.
2. 출력은 JSON 객체 하나만.

[출력 포맷(JSON)]
{{
  "role": "assistant",
  "thought": "종합 소견 및 향후 계획 안내",
  "intent": "summary",
  "content": "..."
}}
""".strip()
# ==========================================
# [유틸리티]
# ==========================================
def call_llm(prompt: str, temperature: float) -> str:
    for attempt in range(1, RETRIES + 1):
        port = next(cycle)
        url = f"http://127.0.0.1:{port}/api/generate"
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "top_p": 0.9, "num_ctx": 4096},
        }
        try:
            r = requests.post(url, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            res = r.json().get("response", "")
            if res: return res
        except Exception as e:
            sleep_s = 1.0 * attempt # 대기 시간 조금 늘림
            print(f"[LLM Error] Port {port}: {e} (Sleep {sleep_s}s)")
            time.sleep(sleep_s)
    return ""

def extract_json(text: str) -> Dict[str, Any]:
    if not text: return {}
    # 1. ```json ... ``` 패턴
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    
    # 2. 리스트만 덜렁 있는 경우 ([...]) -> 감싸주기
    m_list = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    if m_list:
        try: return {"dialogue": json.loads(m_list.group(1))}
        except: pass

    # 3. Brace Scanning
    start = text.find("{")
    if start != -1:
        # 간단한 스택 방식 (네스팅 고려)
        count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{': count += 1
            elif char == '}': count -= 1
            if count == 0:
                try: return json.loads(text[start:i+1])
                except: break
    
    # 4. 리스트 스캐닝 ([...])
    start = text.find("[")
    if start != -1:
        count = 0
        for i, char in enumerate(text[start:], start):
            if char == '[': count += 1
            elif char == ']': count -= 1
            if count == 0:
                try: return {"dialogue": json.loads(text[start:i+1])}
                except: break
                
    return {}

def normalize_key(text: str) -> str:
    if not text: return ""
    return re.sub(r"[^\w가-힣]", "", text.lower())[:120]

# ★ 변경: 검증 로직 완화 (Multi-question 완화)
def is_multi_question(content: str) -> bool:
    if not content: return False
    # 금지어(그리고, 혹은 등) 체크 제거 -> 한국어에서 너무 흔함
    # 오직 물음표 개수로만 판단
    return content.count("?") >= 2

def sanitize_single_question(content: str) -> str:
    """물음표가 2개 이상이면 첫 번째 물음표 뒤를 잘라버림"""
    if not content: return content
    if content.count("?") >= 2:
        first_q = content.find("?")
        return content[:first_q+1]
    return content

# ==========================================
# [검증 및 Repair]
# ==========================================
def validate_dialogue(data: Dict[str, Any]) -> Tuple[bool, str]:
    if not data or "dialogue" not in data: return False, "no_dialogue_key"
    dlg = data["dialogue"]
    if not isinstance(dlg, list): return False, "dialogue_not_list"
    if len(dlg) < 2: return False, "too_short"

    seen_assistant = False
    
    # Turn Mismatch 검사 (Assistant -> User -> Assistant)
    for i, turn in enumerate(dlg):
        role = turn.get("role")
        
        # 순서 강제
        expected_role = "assistant" if i % 2 == 0 else "user"
        if role != expected_role:
            return False, f"turn_mismatch_expected_{expected_role}"

        if role == "assistant":
            seen_assistant = True
            # 필드 누락 검사
            if not all(k in turn for k in ("thought", "intent", "content")):
                return False, "assist_missing_fields"
            
            # 멀티 질문 검사
            if is_multi_question(turn.get("content", "")):
                return False, "multi_question"

    if not seen_assistant: return False, "no_assistant"

    # Summary 검사
    last_turn = dlg[-1]
    # 마지막이 User면 Summary가 없는 것 (보통 Assistant가 마무리해야 함)
    if last_turn["role"] == "user":
        return False, "ends_with_user"
    
    if "summary" not in str(last_turn.get("intent", "")).lower():
        return False, "no_summary"

    return True, "ok"

def append_summary(profile_str: str, dlg_data: Dict) -> Dict:
    """대화가 User로 끝났거나 Summary가 없을 때 강제로 Summary 턴 생성 후 부착"""
    dlg_json = json.dumps(dlg_data["dialogue"], ensure_ascii=False)
    prompt = APPEND_SUMMARY_PROMPT.format(profile_json=profile_str, dialogue_json=dlg_json)
    
    # LLM이 단일 턴 JSON을 줄 것을 기대?
    res_str = call_llm(prompt, temperature=REPAIR_TEMP)
    
    # 파싱 시도 (객체 하나)
    try:
        # 간단히 중괄호 찾기
        start = res_str.find("{")
        end = res_str.rfind("}")
        if start != -1 and end != -1:
            summary_turn = json.loads(res_str[start:end+1])
            if "role" in summary_turn and "content" in summary_turn:
                dlg_data["dialogue"].append(summary_turn)
                return dlg_data
    except:
        pass
    return dlg_data # 실패하면 원본 반환

# ==========================================
# [Main Logic]
# ==========================================
def process_case(case_id: int, seed: Dict) -> Tuple[Optional[Dict], str]:
    # 1. Profile
    p_prompt = PROFILE_PROMPT_TPL.format(
        category=seed.get("category",""),
        complaint=seed.get("complaint",""),
        risk=seed.get("risk",""),
        diagnosis_guess=seed.get("diagnosis_guess","")
    )
    p_raw = call_llm(p_prompt, temperature=PROFILE_TEMP)
    profile = extract_json(p_raw)
    
    # Profile Validation (간소화)
    if "profile" not in profile or "symptoms" not in profile:
        return None, "profile_struct_error"

    profile_str = json.dumps(profile, ensure_ascii=False, indent=2)

    # 2. Dialogue Generation
    d_prompt = DIALOGUE_PROMPT_TPL.format(
        profile_json=profile_str,
        doctor_style=random.choice(DOCTOR_STYLES),
        user_style=random.choice(USER_STYLES)
    )
    d_raw = call_llm(d_prompt, temperature=DIALOGUE_TEMP)
    dlg_data = extract_json(d_raw)

    # ★ 1차 수선: 물음표 2개 이상이면 잘라버림 (LLM 다시 부르지 않고 로직으로 해결)
    if "dialogue" in dlg_data and isinstance(dlg_data["dialogue"], list):
        for turn in dlg_data["dialogue"]:
            if turn.get("role") == "assistant":
                turn["content"] = sanitize_single_question(turn.get("content", ""))

    # 3. Validation
    ok, reason = validate_dialogue(dlg_data)

    # ★ 2차 수선: Summary가 없거나 User로 끝난 경우 -> Summary 턴만 생성해서 붙이기
    if not ok and (reason == "no_summary" or reason == "ends_with_user"):
        print(f"[{case_id}] Append Summary...")
        dlg_data = append_summary(profile_str, dlg_data)
        ok, reason = validate_dialogue(dlg_data) # 재검증

    if not ok:
        # 최후의 수단: 그냥 덮어놓고 리트라이 (전체 재생성보다는 나음)
        # 하지만 여기까지 오면 그냥 Fail 처리하고 다음 시드로 넘어가는 게 시간상 이득
        return None, f"dialogue_{reason}"

    return {
        "case_id": case_id,
        "seed_info": seed,
        "patient_profile": profile,
        "conversation": dlg_data
    }, "ok"


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    if not os.path.exists(INPUT_FILE):
        print(f"Input not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    # Dedup & Shuffle
    unique_map = {}
    for s in seeds:
        if "complaint" not in s:
            continue
        k = normalize_key(s["complaint"])
        if k and k not in unique_map:
            unique_map[k] = s
    seeds = list(unique_map.values())
    random.shuffle(seeds)

    print(f"Loaded {len(seeds)} unique seeds.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Resume Logic (연속 case_id 유지)
    done_keys = set()
    start_id = 1

    if OVERWRITE_OUTPUT:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as _:
            pass
        print(f"Overwrite output: {OUTPUT_FILE}")
    else:
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        cid = obj.get("case_id")
                        if isinstance(cid, int):
                            start_id = max(start_id, cid + 1)

                        comp = obj.get("seed_info", {}).get("complaint", "")
                        k = normalize_key(comp)
                        if k:
                            done_keys.add(k)
                    except:
                        pass
            print(f"Resuming from case_id {start_id}. done_seeds={len(done_keys)}")

    success = 0
    stats = Counter()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for seed in seeds:
            if MAX_CASES is not None and success >= MAX_CASES:
                break

            k = normalize_key(seed.get("complaint", ""))
            if k in done_keys:
                stats["skip_done"] += 1
                continue

            # 성공 수 기반으로 case_id 부여 (문제 없게)
            current_cid = start_id + success

            print(f"Processing [{current_cid}] {seed.get('category','')} / {seed.get('diagnosis_guess','')} ...")
            res, msg = process_case(current_cid, seed)

            if res:
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                f_out.flush()

                done_keys.add(k)
                success += 1
                stats["success"] += 1
                print("  -> Success")

                if AUTOSAVE_EVERY > 0 and success % AUTOSAVE_EVERY == 0:
                    try:
                        os.fsync(f_out.fileno())
                    except:
                        pass
                    print(f"  [Auto-Save] success={success}, stats={dict(stats)}")
            else:
                stats[msg] += 1
                print(f"  -> Fail: {msg}")

            time.sleep(0.5)

        try:
            os.fsync(f_out.fileno())
        except:
            pass

    print(f"Done. success={success}, stats={dict(stats)}")

if __name__ == "__main__":
    main()


