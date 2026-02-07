# Medical-CoT-SFT-Dataset-Generator
Distilling Clinical Reasoning from Giant Models to SLMs

본 프로젝트는 120B+ 규모의 거대 언어 모델(Teacher Model)이 보유한 임상적 추론 능력(Clinical Reasoning)을 20B 이하의 소형 언어 모델(Student Model)에 이식하기 위한 고품질 합성 데이터 생성 파이프라인입니다.

단순한 문답(Q&A) 형식이 아닌, 의사의 사고 과정(Chain of Thought)과 환자 프로필(Ground Truth)을 포함하여 구조화된 데이터를 생성함으로써, 모델의 환각(Hallucination) 현상을 방지하고 문진의 논리성을 강화하는 데 중점을 두었습니다.

Repository Structure
.
├── Data/
│   ├── scenarios.json          # 생성된 시나리오 시드 파일
│   └── medical_chat_data.jsonl # 최종 결과물 (대용량 학습 데이터)
├── Medical_data_creater.py     # 메인 파이프라인 (Profile -> Dialogue 생성)
├── seed_GEN.py                 # 시나리오 시드 생성 코드
├── requirements.txt            # 필요 라이브러리 목록
└── README.md                   # 프로젝트 설명서
