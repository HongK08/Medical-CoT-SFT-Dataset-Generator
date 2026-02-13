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
PORTS = [22134]  # SSH 터널링 포트들 (여러 개면 로드밸런싱)
MODEL = "gpt-oss:120b"

BASE_DIR = "/home/HongKi-Arch/Desktop/LLM_DATASET_Project/Data"
OUTPUT_FILE = os.path.join(BASE_DIR, "scenarios.json")

cycle = itertools.cycle(PORTS)

# ==========================================
# [타겟 카테고리]
# ==========================================
TARGET_CATEGORIES = [
    "호흡기내과", "소화기내과", "순환기내과", "신경과", "정형외과",
    "이비인후과", "피부과", "안과", "비뇨의학과", "산부인과",
    "정신건강의학과", "응급의학과", "내분비내과", "류마티스내과",
    "신장내과", "감염내과", "알레르기내과", "외과", "소아청소년과",
    "신경외과", "흉부외과", "재활의학과", "마취통증의학과"
]

# ==========================================
# [프롬프트 템플릿]
# ==========================================
BASE_PROMPT = """
당신은 의료 데이터 설계자입니다.
이번에는 **[{target_category}]** 영역의 환자 주호소(Chief Complaint) 시나리오 5개를 생성하십시오.

[제약 조건]
1. 진료과: 반드시 '{target_category}'에 해당하는 케이스만 작성할 것. (타 진료과 증상 금지)
2. 주호소: 환자의 자연스러운 '구어체' 사용. (예: "머리가 깨질 듯해요")
3. 위험도: {risk_instruction}
4. 출력 포맷: 반드시 아래 JSON List 형식만 출력. 설명 금지.
5. 각 항목은 반드시 다음 키를 포함: category, complaint, risk, diagnosis_guess

[출력 예시]
[
  {{"category": "{target_category}", "complaint": "가슴이 쥐어짜듯이 아파요", "risk": "high", "diagnosis_guess": "협심증"}},
  {{"category": "{target_category}", "complaint": "무릎이 시큰거려요", "risk": "low", "diagnosis_guess": "관절염"}}
]
""".strip()

# ==========================================
# [유틸리티]
# ==========================================

def call_llm(
    prompt: str,
    temperature: float = 0.85,
    timeout: int = 600,
    retries: int = 3,
    backoff_base: float = 0.8
) -> str:
    """
    Ollama /api/generate 호출 (재시도 + 간단 백오프 포함)
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        port = next(cycle)
        url = f"http://127.0.0.1:{port}/api/generate"
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_ctx": 4096
            },
        }
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            last_err = e
            sleep_s = backoff_base * attempt
            print(f"[call_llm] attempt {attempt}/{retries} failed on port {port}: {e} (sleep {sleep_s:.1f}s)")
            time.sleep(sleep_s)

    print(f"[call_llm] giving up. last_err={last_err}")
    return ""


def robust_json_parse(text: str) -> List[Dict[str, Any]]:
    """
    스택 기반 견고한 JSON List 파싱
    - 가장 앞 '['부터 balanced bracket로 닫히는 ']'까지를 잘라 json.loads
    """
    if not text:
        return []
    try:
        start = text.find("[")
        if start == -1:
            return []

        s = text[start:]
        depth = 0
        in_string = False
        escape = False

        for i, ch in enumerate(s):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return json.loads(s[: i + 1])
    except Exception:
        return []
    return []


def normalize_text(text: str) -> str:
    """
    중복 제거를 위한 텍스트 정규화
    - 공백 제거, 소문자화, 특수문자 제거(한글/영문/숫자/_ 유지)
    """
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[^\w가-힣]", "", t)
    return t[:120]


def normalize_category(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[\s\(\)\-_/]", "", s.strip())


def normalize_risk(risk_raw: Any) -> Optional[str]:
    if not isinstance(risk_raw, str):
        return None
    r = risk_raw.lower().strip()
    if r in ("moderate", "mid"):
        r = "medium"
    if r in ("emergency", "critical"):
        r = "high"
    if r not in ("low", "medium", "high"):
        return None
    return r


def validate_item(item: Dict[str, Any], target_cat: str) -> bool:
    """
    생성 배치의 각 item 검증 + 정규화(가능한 범위에서 표준화)
    """
    if not isinstance(item, dict):
        return False

    # 필수 키
    for k in ("complaint", "risk", "diagnosis_guess"):
        if k not in item:
            return False

    # complaint 체크
    comp = item.get("complaint", "")
    if not isinstance(comp, str) or len(comp.strip()) < 4:
        return False

    # category 체크 (절충안)
    cat = item.get("category", "")
    tgt_norm = normalize_category(target_cat)

    if not isinstance(cat, str) or len(cat.strip()) == 0:
        # 누락이면 주입
        item["category"] = target_cat
    else:
        cat_norm = normalize_category(cat)

        # 너무 짧은 카테고리는 신뢰하지 않음
        if len(cat_norm) < 2:
            return False

        # "소화기" vs "소화기내과" 같이 축약은 허용
        if tgt_norm in cat_norm or cat_norm in tgt_norm:
            item["category"] = target_cat  # 표준화
        else:
            return False

    # risk 정규화
    nr = normalize_risk(item.get("risk"))
    if nr is None:
        return False
    item["risk"] = nr

    # diagnosis_guess 정리
    dg = item.get("diagnosis_guess", "")
    if not isinstance(dg, str) or len(dg.strip()) < 2:
        return False
    item["diagnosis_guess"] = dg.strip()[:60]

    return True


def validate_loaded_item(item: Dict[str, Any]) -> bool:
    """
    기존 파일 로드 시 검증:
    - category가 TARGET_CATEGORIES 내인지 확인 후 validate_item로 재검증
    """
    if not isinstance(item, dict):
        return False
    cat = item.get("category", "")
    if not isinstance(cat, str) or cat not in TARGET_CATEGORIES:
        return False
    return validate_item(item, cat)


def pick_category(
    target_categories: List[str],
    cat_counter: Counter,
    mode: str = "mix",
    underfill_prob: float = 0.8
) -> str:
    """
    카테고리 선택 전략:
    - mode="random": 완전 랜덤
    - mode="underfill": 가장 적게 생성된 카테고리 우선
    - mode="mix": underfill_prob 확률로 underfill, 나머지는 랜덤
    """
    if mode == "random":
        return random.choice(target_categories)
    if mode == "underfill":
        return min(target_categories, key=lambda c: cat_counter.get(c, 0))

    # mix
    if random.random() < underfill_prob:
        return min(target_categories, key=lambda c: cat_counter.get(c, 0))
    return random.choice(target_categories)


def autosave(path: str, data: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ==========================================
# [메인 로직]
# ==========================================
def main(
    target_count: int = 1000,
    category_pick_mode: str = "mix",   # "random" | "underfill" | "mix"
    underfill_prob: float = 0.8,
    autosave_every: int = 50
) -> None:
    os.makedirs(BASE_DIR, exist_ok=True)

    all_scenarios: List[Dict[str, Any]] = []
    unique_hashes = set()

    risk_counter: Counter = Counter()
    cat_counter: Counter = Counter()
    fail_stats: Counter = Counter()
    consecutive_failures = 0

    # 기존 파일 로드
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                for item in existing:
                    if validate_loaded_item(item):
                        h = normalize_text(item["complaint"])
                        if h not in unique_hashes:
                            unique_hashes.add(h)
                            all_scenarios.append(item)
                            risk_counter[item["risk"]] += 1
                            cat_counter[item["category"]] += 1
                print(f"Loaded {len(all_scenarios)} valid unique scenarios from existing file.")
            else:
                print("Existing file format invalid (not a list). Starting fresh.")
        except Exception as e:
            print(f"File read warning: {e}. Starting fresh.")

    print(f"Target Goal: {target_count} UNIQUE scenarios")
    print(f"Category pick mode: {category_pick_mode} (underfill_prob={underfill_prob})")

    while len(all_scenarios) < target_count:
        current_cat = pick_category(TARGET_CATEGORIES, cat_counter, mode=category_pick_mode, underfill_prob=underfill_prob)

        # risk 밸런싱 (전체 high 비율)
        total_valid = sum(risk_counter.values()) or 1
        high_ratio = risk_counter["high"] / total_valid

        if high_ratio < 0.3:
            risk_instruction = "반드시 'High Risk(응급/중증)' 케이스를 3개 이상 포함할 것."
        else:
            risk_instruction = "Low, Medium, High Risk를 골고루 섞어서 구성할 것."

        prompt = BASE_PROMPT.format(
            target_category=current_cat,
            risk_instruction=risk_instruction
        )

        print(f"Requesting [{current_cat}] (HighRatio: {high_ratio:.2f})... (Unique: {len(all_scenarios)}/{target_count})")

        raw = call_llm(prompt)
        if not raw:
            fail_stats["empty_response"] += 1
            consecutive_failures += 1
        else:
            batch = robust_json_parse(raw)
            if not batch:
                fail_stats["parse_error"] += 1
                consecutive_failures += 1
                print("  ! Parse failed.")
            else:
                added = 0
                for item in batch:
                    if not validate_item(item, current_cat):
                        fail_stats["validation_error"] += 1
                        continue

                    h = normalize_text(item["complaint"])
                    if not h:
                        fail_stats["empty_key"] += 1
                        continue
                    if h in unique_hashes:
                        fail_stats["duplicate"] += 1
                        continue

                    unique_hashes.add(h)
                    all_scenarios.append(item)
                    risk_counter[item["risk"]] += 1
                    cat_counter[item["category"]] += 1
                    added += 1

                if added > 0:
                    print(f"  + Added {added} items.")
                    consecutive_failures = 0

                    if autosave_every > 0 and (len(all_scenarios) % autosave_every == 0):
                        autosave(OUTPUT_FILE, all_scenarios)
                        print(f"  [Auto-Save] {len(all_scenarios)} items saved.")
                else:
                    print("  ! Batch yielded 0 valid/unique items.")
                    consecutive_failures += 1

        if consecutive_failures >= 5:
            print(f"Too many failures. Stats: {dict(fail_stats)}. Sleeping 5s...")
            time.sleep(5)
            consecutive_failures = 0

        time.sleep(0.5)

    # 최종 저장
    autosave(OUTPUT_FILE, all_scenarios)

    print(f"\nGeneration Complete! {len(all_scenarios)} items saved to {OUTPUT_FILE}")
    print(f"Final Risk Dist: {dict(risk_counter)}")
    print(f"Final Category Dist: {dict(cat_counter)}")
    print(f"Fail Stats: {dict(fail_stats)}")


if __name__ == "__main__":
    # category_pick_mode:
    # - "mix": (기본) 부족한 카테고리 우선(확률 underfill_prob) + 랜덤 섞기
    # - "underfill": 항상 부족한 카테고리 우선
    # - "random": 완전 랜덤
    main(target_count=5000, category_pick_mode="mix", underfill_prob=0.8, autosave_every=100)
