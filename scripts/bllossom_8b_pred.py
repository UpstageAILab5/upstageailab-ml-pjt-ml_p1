# pip install datasets transformers torch evaluate transformers[torch] accelerate
import pandas as pd
from transformers import AutoTokenizer
import torch
import argparse
import mlflow
import torch
import datetime
import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import gc
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# MLflow 서버 연결 설정
os.environ["MLFLOW_TRACKING_URI"] = "http://10.196.197.32:30164"

def get_next_run_number():
    # 현재 실험의 모든 실행을 가져옴
    experiment = mlflow.get_experiment_by_name("bllossom_8b_prd")
    if experiment is None:
        return 1

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    # 'tags.mlflow.runName' 열이 있는지 확인
    if 'tags.mlflow.runName' not in runs.columns:
        return 1  # 기본값 반환 또는 다른 처리
    
    # 기존 run_name에서 가장 큰 번호 찾기
    max_run_number = 0
    for run_name in runs['tags.mlflow.runName']:
        if run_name and 'bllossom_8b_prd' in run_name:
            try:
                number = int(run_name.split('_')[-1])
                max_run_number = max(max_run_number, number)
            except ValueError:
                continue
    
    return max_run_number + 1


def predict_score(input_path: str, output_path: str):
    mlflow.set_experiment("bllossom_8b_prd")
    
    # 다음 실행 번호 가져오기
    run_number = get_next_run_number()
    run_name = f"bllossom_8b_prd_{run_number}"
    
    mlflow.start_run(run_name=run_name)

    try:
        # 파라미터 기록
        mlflow.log_params({
            "model_name": "MLP-KTLim-llama-3-Korean-Bllossom-8B",
            "max_length": 300,
            "threshold": 0.5,
            "input_path": input_path,
            "output_path": output_path,
            "run_number": run_number  # 실행 번호도 파라미터로 기록
        })

        dataset = pd.read_csv(input_path)
        df = dataset.head(1).copy()

        results = []

        def prompts(example):
            prompt_list = []
            for i in range(len(example['instruction'])):
                prompt_list.append(
        f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>{example['instruction'][i]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{example['output'][i]}<|eot_id|>"""
                )
            return prompt_list

        # LoRA 설정 : 양자화된 모델에서 Adaptor를 붙여서 학습할 파라미터만 따로 구성함

        BASE_MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

        lora_config = LoraConfig(
            r=8,
            lora_alpha = 32,
            lora_dropout = 0.1,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,device_map="auto")

        # 모델의 그래디언트 체크포인팅을 활성화하여 메모리 사용량 최적화
        model.gradient_checkpointing_enable()

        # 모델의 파라미터가 k-비트 양자화에 적합하도록 조정 준비
        model = prepare_model_for_kbit_training(model)

        # freezing
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

        torch.cuda.empty_cache()  # GPU 메모리 정리
        gc.collect()             # CPU 메모리 정리

        # 환경 변수 설정으로 메모리 관리 개선
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        # 모델 로드 방식 수정
        model = AutoAWQForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            offload_folder="temp_model_storage",  # 임시 저장소 사용
            offload_state_dict=True,  # 상태 딕셔너리 오프로드
        )

        # 토크나이저 설정
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            use_fast=False  # 더 안정적인 로딩을 위해
        )

        # 양자화 설정
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }


        quant_path = '/data/ephemeral/home/scripts/models/llama-3-korean-awq'

        # 이미 양자화 해서 이 과정 넘어감
        # # 메모리 관리

        # gc.collect()

        # # 양자화 실행
        # model.quantize(tokenizer, quant_config=quant_config)

        # # 성공하면 저장
        # model.save_quantized(quant_path)
        # tokenizer.save_pretrained(quant_path)

        # os.environ["MKL_THREADING_LAYER"] = "GNU"
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()

        # vLLM에서는 경로만 직접 지정
        llm = LLM(
            model=quant_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.6,  # GPU 메모리 사용량 제한
            max_model_len=1024,
            tensor_parallel_size=1           # 최대 시퀀스 길이 제한
        )

        # transformers는 repo_type 사용 가능
        tokenizer = AutoTokenizer.from_pretrained(
            quant_path,
            repo_type="local",
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        def process_review(idx, row):
            label = row['label']
            text = row['review_content']

            messages = [
                {
                    "role": "system", 
                    "content": """당신은 한국어 문장의 감성 분석을 전문으로 하는 매우 유능한 AI 어시스턴트입니다. 
        먼저 입력된 긍/부정 레이블(0=부정, 1=긍정)을 반드시 확인한 후, 해당 레이블을 참고하여 문장의 전체적인 감정을 분석합니다.

        **입력 형식:**
        - 레이블: [0 또는 1]
        - 문장: [분석할 텍스트]

        **분석 단계:**
        1. **레이블 확인:** 
            - 0 = 부정
            - 1 = 긍정
        2. **감정 점수 산출:**
            - **레이블이 0 (부정)일 경우:** 감정 점수는 **1.0 - 5.0** 사이에서 산출
                - **1.0:** 극히 부정적 (예: "진짜 이런 ㅈ같은 음식점은 다시는 가지마삼.")
                - **1.5:** 매우 부정적
                - **2.0:** 심각하게 부정적
                - **2.5:** 매우 부정적
                - **3.0:** 부정적
                - **3.5:** 약간 부정적
                - **4.0:** 보통 부정적
                - **4.5:** 약간 부정적
                - **5.0:** 경미하게 부정적
            - **레이블이 1 (긍정)일 경우:** 감정 점수는 **5.5 - 10.0** 사이에서 산출
                - **5.5:** 약간 긍정적
                - **6.0:** 경미하게 긍정적
                - **6.5:** 중립에서 약간 긍정적
                - **7.0:** 중립적
                - **7.5:** 중립에서 약간 긍정적
                - **8.0:** 경미하게 긍정적
                - **8.5:** 약간 긍정적
                - **9.0:** 긍정적
                - **9.5:** 매우 긍정적
                - **10.0:** 매우 긍정적
            - 점수는 **0.5점 단위**로 산출
        3. **신뢰도 점수 산출:** (0-100%, 1% 단위)
            - 문맥이 명확하고 감정 표현이 뚜렷할수록 높은 신뢰도
            - 애매모호한 표현이나 복합적인 감정이 섞여있을 경우 낮은 신뢰도
            - 문장이 짧거나 불완전한 경우 낮은 신뢰도

        **평가 척도:**

        - **부정적 감정 (레이블=0):**
            - **1.0:** 극히 부정적인 감정
            - **1.5 – 2.0:** 매우 부정적인 감정
            - **2.5 – 3.0:** 심각하게 부정적인 감정
            - **3.5 – 4.0:** 부정적인 감정
            - **4.5 – 5.0:** 약간 부정적인 감정

        - **긍정적 감정 (레이블=1):**
            - **5.5 – 6.0:** 약간 긍정적인 감정
            - **6.5 – 7.0:** 중립에서 약간 긍정적인 감정
            - **7.5 – 8.0:** 경미하게 긍정적인 감정
            - **8.5 – 9.0:** 긍정적인 감정
            - **9.5 – 10.0:** 매우 긍정적인 감정

        **부정적 감정 인식 강화:**
        - **부정적 단어 및 표현:** "짜다", "실망", "불만족", "나쁘다", "별로", "최악", "실패", "싫다", "안 좋다", "ㅈ같다", "가지마삼" 등
        - **부정적 문맥 파악:** 문장 전체의 맥락에서 부정적인 의미를 파악
        - **복합 감정 처리:** 문장 내에 긍정적 및 부정적 감정이 함께 있을 경우, 부정적 감정을 우선적으로 반영

        **예시:**
        - **예시 1 (레이블=0):**
            - 문장: "진짜 이런 ㅈ같은 음식점은 다시는 가지마삼."
            - 감정 점수: 1.0
            - 신뢰도: 98%
            - 평가: 극히 부정적인 감정
        - **예시 2 (레이블=0):**
            - 문장: "음식이 전반적으로 짰다."
            - 감정 점수: 4.5
            - 신뢰도: 95%
            - 평가: 약간 부정적인 감정
        - **예시 3 (레이블=1):**
            - 문장: "전체적으로 만족스러웠다."
            - 감정 점수: 8.5
            - 신뢰도: 95%
            - 평가: 약간 긍정적인 감정
        **예시 4 (레이블=1):**
            - 문장: "분위기도 좋도 음식도 괜찮은데 가격이 좀 비싸요~ 와인은 무조건 시켜야하더라구요~!"
            - 감정 점수: 7
            - 신뢰도: 95%
            - 평가: 중립에서 약간 긍정적인 감정

        **답변 형식:**
        - 감정 점수: [점수]
        - 신뢰도: [백분율]%
        - 평가: [감정 평가]

        **주의 사항:**
        - 레이블과 감정 점수가 일치하도록 반드시 확인
        - 레이블=0일 경우, 감정 점수는 1.0에서 5.0 사이
        - 레이블=1일 경우, 감정 점수는 5.5에서 10.0 사이
        - 극단적인 감정 표현은 해당 범위의 최솟값 또는 최댓값에 가깝게 매겨지도록 반영"""
                },
                {
                    "role": "user",
                    "content": f"레이블: {label}\n문장: {text}"
                }, 
            ]

            prompt_message = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
            )

            # 결과 텍스트에서 감정 점수 추출
            eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            outputs = llm.generate(prompt_message, SamplingParams(stop_token_ids=eos_token_id, temperature=0.1, top_p=0.95,max_tokens=2048))

            for output in outputs:
                generated_text = output.outputs[0].text
                score_match = re.search(r'감정 점수: (\d+\.?\d*)', generated_text)
                confidence_match = re.search(r'신뢰도: (\d+)%', generated_text)
                if score_match:
                    return float(score_match.group(1)), float(confidence_match.group(1))
            return None
        
            # 4. 각 리뷰 처리 및 score 계산
        scores = []
        confidences = []
        for idx, row in df.iterrows():
            score, confidence = process_review(idx, row)
            df.at[idx, 'score'] = score
            df.at[idx, 'confidence'] = confidence
            if confidence is not None:
                confidences.append(confidence)
        
        # CSV 파일로 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # utf-8-sig로 한글 깨짐 방지
        print(f"\nResults saved to: {output_path}")


        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            high_confidence_preds = sum(1 for c in confidences if c > 90 or c < 10)  # 90% 이상 또는 10% 이하

            mlflow.log_metrics({
                "total_predictions": len(df),
                "average_confidence": avg_confidence,
                "high_confidence_predictions": high_confidence_preds,
                "high_confidence_ratio": high_confidence_preds / len(df)
    })

        mlflow.log_metrics({
            "total_predictions": len(results),
            "average_confidence": avg_confidence,
            "high_confidence_predictions": high_confidence_preds,
            "high_confidence_ratio": high_confidence_preds / len(results)
        })
        # 모델 예측 결과를 MLflow에 아티팩트로 저장
        mlflow.log_artifact(output_path, "predictions")

        # 모델 버전 관리를 위한 태그 추가
        mlflow.set_tag("model_version", "v1")
        mlflow.set_tag("model_status", "production")
        mlflow.set_tag("deployment_timestamp", datetime.datetime.now().isoformat())

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    # 커맨드 라인 인자 파서 생성
    parser = argparse.ArgumentParser(description='두번째 모델 스크립트')
    parser.add_argument('--input_path', required=True, help='1차 모델 예측 파일 출력 경로')
    parser.add_argument('--output_path', required=True, help='2차 모델 예측 파일 출력 경로')

    # 인자 파싱
    args = parser.parse_args()

    # prepare_data 함수 실행
    predict_score(
        input_path=args.input_path,
        output_path=args.output_path,
    )