root@dd65a69d70f6:/workspace/Data# python test.py
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
/opt/venv/lib/python3.12/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2026.2.1: Fast Gpt_Oss patching. Transformers: 4.56.2.
   \\   /|    NVIDIA GeForce RTX 4090. Num GPUs = 1. Max memory: 23.51 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu130. CUDA: 8.9. CUDA Toolkit: 13.0. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.68it/s]
Unsloth: PEFT set target_parameters but found no matching parameters.
This is expected for MoE models - Unsloth handles MoE expert LoRA targeting separately.
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-02-13

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Instructions

당신은 전문 의료 상담 어시스턴트입니다. 환자의 증상을 분석하고 Thought, Intent 단계를 거쳐 체계적으로 답변하십시오.<|end|><|start|>user<|message|>요즘 손이 떨리고 심장이 빨리 뛰는 느낌이 있어요. 왜 그럴까요?<|end|><|start|>assistant<|channel|>analysis<|message|>The user reports experiencing hand tremor and palpitations. A structured approach will be used: first gather basic information (Chief Complaint, Onset, Duration, Severity, Aggravating/Relieving Factors, Associated Symptoms, Medical History, Meds, Social). Then synthesize into Thought and Intent statements.

Thought: Chief Complaint  
Intent: chief_complaint

User says they have hand tremor and heart racing.

Thought: Onset  
Intent: onset

Symptoms started 3 weeks ago, initially intermittent, now daily.

Thought: Duration  
Intent: duration

Symptoms last about 10 minutes each episode, occurring several times a day.

Thought: Severity  
Intent: severity

Severity is moderate (5/10).

Thought: Aggravating Factors  
Intent: aggravating_factors

Stress, caffeine, nicotine, dehydration, lack of sleep.

Thought: Relieving Factors  
Intent: relieving_factors

Rest, deep breathing, meditation, hydration, caffeine reduction, avoiding stimulants.

Thought: Associated Symptoms  
Intent: associated_symptoms

Lightheadedness, sweating, anxiety, palpitations, tremor, shortness of breath, nausea.

Thought: Medical History  
Intent: medical_history

Hypertension, anxiety disorder, migraine, GERD
