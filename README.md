# Medical CoT Dataset Generator

Teacher LLM-based Clinical Reasoning Data Pipeline

---

## Overview

This project builds a structured medical dialogue dataset using a large teacher language model.

It generates:

* Realistic patient profiles
* Multi-turn medical consultations
* Expert-level clinical reasoning (Chain-of-Thought)
* Structured diagnostic JSON outputs

The final dataset is designed for:

* Chain-of-Thought (CoT) supervised fine-tuning
* Knowledge distillation into smaller student models
* Clinical reasoning chatbot training
* Structured medical QA systems

This repository implements a reproducible pipeline for generating high-quality medical reasoning data at scale.

---

## Motivation

Medical reasoning requires:

* Context comprehension
* Differential diagnosis
* Risk assessment
* Structured treatment planning

General LLM outputs often lack structural reliability.
This project enforces deterministic JSON structure while preserving reasoning depth.

The goal is to create a distillation-ready dataset suitable for training compact medical assistant models (e.g., 1B–7B parameter range).

---

## System Architecture

```
Scenario Seed
      ↓
Patient Profile Generator
      ↓
Dialogue Generator
      ↓
Expert Reasoning Generator (CoT)
      ↓
Structured JSON Validator / Repair
      ↓
JSONL Dataset
      ↓
Student Model Distillation
```

---

## Project Structure

```
LLM Dataset Project/
│
├── Data/
│   ├── scenarios.json
│   ├── medical_chat_data.jsonl
│
├── Medical_Seed_Creator.py
└── Medical_Data_Creator.py
```

### Key Components

**scenarios.json**
Defines structured clinical seeds including symptoms, demographics, and contextual factors.

**generation_script.py**
Main pipeline that:

* Loads scenario seeds
* Calls teacher LLM via HTTP API
* Enforces structured outputs
* Saves successful generations to JSONL

**medical_chat_data.jsonl**
Final dataset file (one JSON object per line).

---

## Dataset Format

Each case is stored in JSONL format:

```json
{
  "case_id": 123,
  "patient_profile": {...},
  "dialogue": [...],
  "expert_output_text": "...",
  "expert_output_json": {...},
  "created_at": "2026-02-07T02:39:22Z"
}
```

### Field Description

* `case_id` — Sequential unique identifier
* `patient_profile` — Structured patient metadata
* `dialogue` — Multi-turn patient–doctor conversation
* `expert_output_text` — Natural language clinical reasoning
* `expert_output_json` — Structured diagnosis and treatment plan
* `created_at` — UTC timestamp

---

## Generation Pipeline

### 1. Scenario Loading

Seeds are loaded from:

```
Data/scenarios.json
```

Each seed defines:

* Symptoms
* Age / Gender
* Risk factors
* Clinical background

---

### 2. Teacher Model Inference

The teacher model is accessed via HTTP API:

```bash
curl -s http://127.0.0.1:22134/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model":"gpt-oss:120b",
    "prompt":"...",
    "stream":false
  }'
```

Generation stages include:

1. Patient profile creation
2. Dialogue simulation
3. Expert-level reasoning generation
4. Structured JSON validation and repair

---

### 3. Resume-Safe Saving

* Existing scenario keys are tracked
* Duplicate cases are skipped
* `case_id` increments only on successful generation
* Output appended to `medical_chat_data.jsonl`

---

## Configuration

Example configurable parameters:

```python
MODEL = "gpt-oss:120b"

PROFILE_TEMP = 0.70
DIALOGUE_TEMP = 0.55
REPAIR_TEMP = 0.30

TIMEOUT = 600
RETRIES = 3
AUTOSAVE_EVERY = 10
```

Adjustable for:

* Creativity vs stability
* JSON strictness
* Runtime reliability

---

## Running

### 1. Start LLM Server

Example (local):

```bash
ollama run gpt-oss:120b
```

Or run your custom LLM API server.

---

### 2. Execute Generator

```bash
python generation_script.py
```

Monitor progress:

```bash
tail -f Data/medical_chat_data.jsonl
```

---

## Design Principles

* Structured reliability over free-form output
* Chain-of-thought preservation
* Resume-safe large-scale generation
* Distillation-ready format
* Clinical plausibility focus

---

## Future Work

* Automatic hallucination detection
* ICD-10 structured tagging
* Confidence estimation
* Multi-specialty dataset balancing
* Automated dataset evaluation metrics
* Student model fine-tuning experiments

---

## License

Specify license here.
