# Medical CoT SFT Dataset Generator

**Distilling Clinical Reasoning from Giant Models to SLMs**

## Project Overview

본 프로젝트는 120B+ 규모의 거대 언어 모델(Teacher Model)이 보유한 임상적 추론 능력(Clinical Reasoning)을 20B 이하의 소형 언어 모델(Student Model)에 이식하기 위한 고품질 합성 데이터 생성 파이프라인입니다.

기존의 단순 문답(Q&A) 방식이 아닌, 의사의 사고 과정(Chain of Thought, CoT)과 환자 프로필(Ground Truth)을 포함한 구조화된 데이터를 생성함으로써, 모델의 환각(Hallucination) 현상을 방지하고 문진의 논리성을 강화하는 것을 목표로 합니다.

## Repository Structure

본 프로젝트의 디렉토리 구조는 다음과 같습니다. 데이터 생성 파이프라인의 입출력 관계를 명확히 하기 위해 데이터 디렉토리와 실행 스크립트를 분리하였습니다.

```text
.
├── Data/
│   ├── scenarios.json          # seed_GEN.py를 통해 생성된 초기 시나리오 시드 파일
│   └── medical_chat_data.jsonl # Medical_data_creater.py를 통해 생성된 최종 CoT 포함 문진 데이터
├── Medical_data_creater.py     # 메인 파이프라인: 환자 프로필 생성 및 의사-환자 대화(CoT) 생성 수행
├── seed_GEN.py                 # 시나리오 생성기: 다양한 질환 카테고리의 주호소(Chief Complaint) 시드 확보
├── requirements.txt            # 프로젝트 실행에 필요한 Python 라이브러리 명세
└── README.md                   # 프로젝트 문서

System Architecture & Requirements
본 프로젝트는 로컬 클라이언트에서 파이프라인을 제어하고, 고성능 연산이 필요한 LLM 추론은 원격 서버(Cloud GPU)에서 수행하는 Client-Server 구조로 설계되었습니다.

1. Server Environment (LLM Hosting)
거대 언어 모델(GPT-OSS 120B)을 로드하고 추론하기 위한 하드웨어 및 소프트웨어 사양입니다.

Model: GPT-OSS 120B (Quantized)

GPU: NVIDIA A100 80GB 또는 동급의 Multi-GPU 환경 (Min. 80GB VRAM 권장)

Serving Framework: Ollama 또는 VLLM

Network: SSH Tunneling을 통한 Localhost 포트 포워딩 (Port 22134)

2. Client Environment (Data Generation)
데이터 생성 스크립트를 실행하고 결과물을 수집하는 로컬 개발 환경입니다.

OS: Arch Linux (Recommended) / Linux / macOS

Language: Python 3.8+

Network: 원격 GPU 서버와의 안정적인 SSH 연결 필요
