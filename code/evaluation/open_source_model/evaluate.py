import os
import json
import ast
import re
from pathlib import Path
from typing import Callable
import logging
import torch
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

def main(
    model_name: str = "Llama-2-7b-chat-hf",
    is_bf16: bool = False,
    batch_size: int = 1,
    expected_cases: int = 100,
    selected_type: str = "calculation_error",
    selected_test: str = "any",
    dataset: str = "GSM8K",
):
    print(f"main start, is_bf16:{is_bf16}, batch_size:{batch_size}")
    model_path = f"/{model_name}/"

    save_dir = f"../../../evaluation/open_source_model/eval_main_results_{dataset}/{model_name}/{selected_type}/{selected_type}_{expected_cases}_{selected_test}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    eval_cases_dir = f"../../../data/generated_cases_{dataset}/{selected_type}/{selected_type}_{expected_cases}/generated_cases_clean.jsonl"
    with open(eval_cases_dir, 'r', encoding='utf-8') as file:
        eval_cases = [json.loads(line) for line in file]
        print("eval_cases loaded")

    templates_dir = f"./template_general/eval_template_{selected_test}.json"
    with open(templates_dir, 'r', encoding="utf8") as file:
        template = json.load(file)[0]

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(save_dir + 'info.log', mode='w')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    model, tokenizer = get_model(model_path, is_bf16=is_bf16)
    print("model loaded")

    batch_llama = get_batch_llama(model, tokenizer)

    eval_jsonl = Path(save_dir) / "eval.jsonl"

    for i in tqdm(range(0, len(eval_cases), batch_size)):
        cur_gsm8k_batch = eval_cases[i : i + batch_size]
        input_str_list, output_str_list = gsm8k_batch_gen(
        template, [d["question"] for d in cur_gsm8k_batch], [d["transformed_solution"] for d in cur_gsm8k_batch], batch_llama
        )
        for  j,(eval_case, output_str) in enumerate(
            zip(cur_gsm8k_batch, output_str_list)
        ):
            with open(eval_jsonl, "a") as f:
                json.dump(
                    dict(
                        **eval_case,
                        eval_result=output_str,
                    ),
                    f,
                )
                f.write("\n")

    with open(save_dir + 'eval.jsonl', encoding="utf8") as f:
        eval_datas = [json.loads(line) for line in f]

    if selected_test == "any" or selected_test == "any_simple" or selected_test == "any_misleading":
        correct_results = []
        wrong_results = []
        for eval in eval_datas:
            start_phrase = 'The solution is'
            start_index = eval["eval_result"].find(start_phrase)

            if start_index != -1:
                start_index += len(start_phrase)
                words_after_phrase = eval["eval_result"][start_index:].split()

                if words_after_phrase:
                    next_word = words_after_phrase[0]
                    if next_word == 'incorrect,':
                        correct_results.append(eval)
                    else:
                        wrong_results.append(eval)
                else:
                    wrong_results.append(eval)
            else:
                wrong_results.append(eval)

        with open(save_dir + "correct.json", "w", encoding="utf8") as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding="utf8") as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)

        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results) / (len(correct_results) + len(wrong_results))}"
        with open(save_dir + "result.json", 'w') as f:
            f.write(result)

    elif selected_test=="step_simple" or selected_test=="step_complex":
        correct_results = []
        wrong_results = []
        wrong_results_step=[]
        for eval in eval_datas:
            start_phrase = 'The solution is'
            start_index = eval["eval_result"].find(start_phrase)
            if start_index != -1:
                start_index += len(start_phrase)
                words_after_phrase = eval["eval_result"][start_index:].split()
                if words_after_phrase:
                    next_word = words_after_phrase[0]
                    if next_word == 'incorrect,':
                        match = re.search(r'the first wrong step is step (\d+)', eval["eval_result"])
                        if match:
                            step_number = int(match.group(1))
                            if step_number == eval["wrong_step"]:
                                correct_results.append(eval)
                            else:
                                wrong_results_step.append(eval)
                        else:
                            wrong_results_step.append(eval)
                    else:
                        wrong_results.append(eval)
                else:
                    wrong_results.append(eval)
            else:
                wrong_results.append(eval)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong_step.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results_step, f, ensure_ascii=False, indent=4)
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)}+{len(wrong_results_step)})={len(correct_results) / (len(correct_results) + len(wrong_results)+len(wrong_results_step))}"
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)

    elif selected_test=="type_simple" or selected_test=="type_random_simple" or selected_test=="type_reverse_simple" or selected_test=="type_complex" or selected_test=="type_random_complex" or selected_test=="type_reverse_complex":
        correct_results = []
        wrong_results = []
        wrong_results_step=[]
        for eval in eval_datas:
            start_phrase = 'The solution is'
            start_index = eval["eval_result"].find(start_phrase)

            if start_index != -1:
                start_index += len(start_phrase)
                words_after_phrase = eval["eval_result"][start_index:].split()
                if words_after_phrase:
                    next_word = words_after_phrase[0]
                    if next_word == 'incorrect,':
                        start_index1 = eval["eval_result"].find("the wrong type is ")
                        if start_index1 == -1:
                            wrong_results_step.append(eval)
                        else:
                            error_type_start = start_index1 + len("the wrong type is ")
                            error_type_end =  eval["eval_result"].find(",", error_type_start)
                            if error_type_end == -1:
                                error_type_end = len(eval["eval_result"])
                            wrong_type = eval["eval_result"][error_type_start:error_type_end].strip()
                            if wrong_type == eval["wrong_type"]:
                                correct_results.append(eval)
                            else:
                                wrong_results_step.append(eval)
                    else:
                        wrong_results.append(eval)
                else:
                    wrong_results.append(eval)
            else:
                wrong_results.append(eval)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong_type.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results_step, f, ensure_ascii=False, indent=4)
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)}+{len(wrong_results_step)})={len(correct_results) / (len(correct_results) + len(wrong_results)+len(wrong_results_step))}"
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)

    elif selected_test=="correction_simple" or selected_test=="correction_complex":
        correct_results = []
        wrong_results = []
        wrong_results_step=[]
        for eval in eval_datas:
            start_phrase = 'The solution is'
            start_index = eval["eval_result"].find(start_phrase)
            if start_index != -1:
                start_index += len(start_phrase)
                words_after_phrase = eval["eval_result"][start_index:].split()
                if words_after_phrase:
                    next_word = words_after_phrase[0]
                    if next_word == 'incorrect,':
                        match = re.search(r", the correct answer is (\d+(\.\d+)?)", eval["eval_result"])
                        if match:
                            correct_answer = float(match.group(1))
                            if correct_answer == eval["original_answer"]:
                                correct_results.append(eval)
                            else:
                                wrong_results_step.append(eval)
                        else:
                            wrong_results_step.append(eval)
                    else:
                        wrong_results.append(eval)
                else:
                    wrong_results.append(eval)
            else:
                wrong_results.append(eval)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong_correction.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results_step, f, ensure_ascii=False, indent=4)
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)}+{len(wrong_results_step)})={len(correct_results) / (len(correct_results) + len(wrong_results)+len(wrong_results_step))}"
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)


def gsm8k_batch_gen(
    template:str, gsm8k_questions: list[str],  gsm8k_questions1: list[str], batch_llm: Callable[[list[str]], list[str]]
):
    input_str_list = [template.format(question=gsm8k_questions[i],solution=gsm8k_questions1[i]) for i in range(len(gsm8k_questions))]
    output_str_list = batch_llm(input_str_list)
    return input_str_list, output_str_list


def get_batch_llama(model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
    @torch.inference_mode()
    def batch_llama(input_strs: list[str]) -> list[str]:
        input_ids_w_attnmask = tokenizer(
            input_strs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(
            input_ids=input_ids_w_attnmask.input_ids,
            attention_mask=input_ids_w_attnmask.attention_mask,
            generation_config=GenerationConfig(
                max_length=4096,
                do_sample=False,
                temperature=0.0,  # t=0.0 raise error if do_sample=True
            ),
        ).tolist()
        real_output_ids = [
            output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)
        ]
        output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
        return output_strs

    return batch_llama

def get_model(model_path: str, is_bf16: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token="[pad]"
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    model.eval()    
    return model, tokenizer


if __name__ == "__main__":
    import fire
    fire.Fire(main)

