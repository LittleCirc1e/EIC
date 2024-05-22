import openai
import json
import random
import re
import backoff
import time
import fire
import os
import logging
import ast
from zhipuai import ZhipuAI
import google.generativeai as genai

# openai.api_key = ""
# openai.api_base = ""

def askAPI(prompt, model_name):
    if model_name == "gpt-3.5-turbo-1106":
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=0,max_tokens=4096,)
        return response['choices'][0]['message']['content']
    elif model_name == "gpt-4-1106-preview":
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=0, max_tokens=4096, )
        return response['choices'][0]['message']['content']
    elif model_name == "glm-4":
        messages = [{"role": "user", "content": prompt}]
        # client = ZhipuAI(api_key="")
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.01)
        return response.choices[0].message.content
    elif model_name == "gemini-pro":
        # GOOGLE_API_KEY = ''
        generation_config = {"temperature": 0}
        genai.configure(api_key=GOOGLE_API_KEY, transport="rest")
        model = genai.GenerativeModel(model_name="gemini-pro",generation_config=generation_config)
        response = model.generate_content(prompt)
        return response.text

transformed_cases=[]

def main(
        expected_cases: int = 100,
        model_name: str = "glm-4",  # gpt-3.5-turbo-1106 or gpt-4-1106-preview or glm-4 or gemini-pro
        selected_type: str = "calculation_error",
        selected_test: str = "any",
        dataset: str = "GSM8K",
        start_epoch: int = 0,
):

    num_of_cases = start_epoch

    save_dir = f"../../../evaluation/closed_source_model/eval_main_results_{dataset}/{model_name}/{selected_type}/{selected_test}/{selected_type}_{expected_cases}_{selected_test}/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    eval_cases_dir = f"../../../data/generated_cases_{dataset}/{selected_type}/{selected_type}_{expected_cases}/generated_cases_clean.jsonl"
    with open(eval_cases_dir, 'r', encoding='utf-8') as file:
        eval_cases = [json.loads(line) for line in file]
        print("eval_cases loaded")

    templates_dir = f"./template_general/eval_template_{selected_test}.json"
    with open(templates_dir, 'r', encoding="utf8") as file:
        template = json.load(file)[0]

    if selected_test == "type_reverse_complex":
        cases1_dir = f"./template_general/type_reverse_complex.json"
        with open(cases1_dir, 'r', encoding="utf8") as file:
            type_initial_cases = json.load(file)
            print("type initial cases loaded")
    elif selected_test == "type_random_complex":
        cases1_dir = f"./template_general/type_random_complex.json"
        with open(cases1_dir, 'r', encoding="utf8") as file:
            type_initial_cases = json.load(file)
            print("type initial cases loaded")
    elif selected_test == "type_complex":
        cases1_dir = f"./template_general/type_complex.json"
        with open(cases1_dir, 'r', encoding="utf8") as file:
            type_initial_cases = json.load(file)
            print("type initial cases loaded")
    elif selected_test == "correction_complex":
        cases1_dir = f"./template_general/correction_complex.json"
        with open(cases1_dir, 'r', encoding="utf-8") as file:
            correction_initial_cases = json.load(file)
            print("correction initial cases loaded")
    elif selected_test == "step_complex":
        cases1_dir = f"./template_general/step_complex.json"
        with open(cases1_dir, 'r', encoding="utf-8") as file:
            step_initial_cases = json.load(file)
            print("step initial cases loaded")

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    if start_epoch == 0:
        file_handler = logging.FileHandler(save_dir + 'info.log', mode='w')
    else:
        file_handler = logging.FileHandler(save_dir + 'info.log', mode='a')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    if selected_test == "any" or selected_test == "any_simple" or selected_test == "any_misleading":
        while num_of_cases < expected_cases:
            logger.info("###########################")
            logger.info(f"selected_test:{selected_test}")
            logger.info(f"selected_type:{selected_type}")
            logger.info(f"total cases:{num_of_cases}/{expected_cases}")

            start_time = time.time()
            eval_case_chosen = eval_cases[num_of_cases]
            prompt = template.format(eval_case_chosen['question'],eval_case_chosen['transformed_solution'])
            #To prevent errors, try several times
            for i in range(5):
                try:
                    eval_result_original = askAPI(prompt, model_name)
                    break
                except:
                    print('\033[1;31m' + "error when askAPI" + '\033[0m')
                    if i == 4:
                        eval_result_original = askAPI(prompt, model_name)
                        # exit()
                    continue
            match = re.search(r'{(.*?)}', eval_result_original, re.DOTALL)
            transformed_case = match.group(1)
            eval_result = "{" + transformed_case + "}"
            num_of_cases += 1
            try:
                eval_result1 = ast.literal_eval(eval_result)
                with open(save_dir + 'eval.jsonl', "a") as f:
                    json.dump(
                        dict(
                            **eval_case_chosen,
                            is_correct=eval_result1["is_correct"],
                            any_explanation=eval_result1["any_explanation"],
                        ),
                        f,
                    )
                    f.write("\n")
            except:
                num_of_cases-=1
                continue

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("execution time:{:.2f}s\n".format(elapsed_time))

        with open(save_dir + 'eval.jsonl', encoding='utf-8') as f:
            eval_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        for eval in eval_datas:
            if eval["is_correct"] == "yes":
                wrong_results.append(eval)
            elif eval["is_correct"] == "Yes":
                wrong_results.append(eval)
            else:
                correct_results.append(eval)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)

        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results) / (len(correct_results) + len(wrong_results))}"
        print(result)
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)

    elif selected_test == "step_complex":
        while num_of_cases < expected_cases:
            logger.info("###########################")
            logger.info(f"selected_test:{selected_test}")
            logger.info(f"selected_type:{selected_type}")
            logger.info(f"total cases:{num_of_cases}/{expected_cases}")

            start_time = time.time()
            eval_case_chosen = eval_cases[num_of_cases]
            prompt = template.format(*step_initial_cases, eval_case_chosen['question'], eval_case_chosen['transformed_solution'])
            num_of_cases += 1
            #To prevent errors, try several times
            for i in range(5):
                try:
                    eval_result_original = askAPI(prompt, model_name)
                    break
                except:
                    print('\033[1;31m' + "error when askAPI" + '\033[0m')
                    if i == 4:
                        eval_result_original = askAPI(prompt, model_name)
                        # exit()
                    continue
            match = re.search(r'{(.*?)}', eval_result_original, re.DOTALL)
            transformed_case = match.group(1)
            eval_result = "{" + transformed_case + "}"
            try:
                eval_result1 = ast.literal_eval(eval_result)
                with open(save_dir + 'eval.jsonl', "a") as f:
                    json.dump(
                        dict(
                            **eval_case_chosen,
                            is_correct=eval_result1["is_correct"],
                            pred_wrong_step=eval_result1["pred_wrong_step"],
                            step_explanation=eval_result1["step_explanation"],
                        ),
                        f,
                    )
                    f.write("\n")
            except:
                num_of_cases -= 1
                continue

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("execution time:{:.2f}s\n".format(elapsed_time))

        with open(save_dir + 'eval.jsonl', encoding='utf-8') as f:
            eval_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        wrong_results_step = []
        for eval in eval_datas:
            if eval ["is_correct"] == "no":
                if eval["pred_wrong_step"] == eval["wrong_step"]:
                    correct_results.append(eval)
                else:
                    wrong_results_step.append(eval)
            elif eval ["is_correct"] == "No":
                if eval["pred_wrong_step"] == eval["wrong_step"]:
                    correct_results.append(eval)
                else:
                    wrong_results_step.append(eval)
            else:
                wrong_results.append(eval)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong_step.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results_step, f, ensure_ascii=False, indent=4)
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)}+{len(wrong_results_step)})={len(correct_results) / (len(correct_results) + len(wrong_results)+len(wrong_results_step))}"
        print(result)
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)

    elif selected_test == "step_simple":
        while num_of_cases < expected_cases:
            logger.info("###########################")
            logger.info(f"selected_test:{selected_test}")
            logger.info(f"selected_type:{selected_type}")
            logger.info(f"total cases:{num_of_cases}/{expected_cases}")

            start_time = time.time()
            eval_case_chosen = eval_cases[num_of_cases]
            prompt = template.format(eval_case_chosen['question'], eval_case_chosen['transformed_solution'])
            num_of_cases += 1
            #To prevent errors, try several times
            for i in range(5):
                try:
                    eval_result_original = askAPI(prompt, model_name)
                    break
                except:
                    print('\033[1;31m' + "error when askAPI" + '\033[0m')
                    if i == 4:
                        eval_result_original = askAPI(prompt, model_name)
                        # exit()
                    continue
            print("eval_result", eval_result_original)
            match = re.search(r'{(.*?)}', eval_result_original, re.DOTALL)
            transformed_case = match.group(1)
            eval_result = "{" + transformed_case + "}"
            try:
                eval_result1 = ast.literal_eval(eval_result)
                with open(save_dir + 'eval.jsonl', "a") as f:
                    json.dump(
                        dict(
                            **eval_case_chosen,
                            is_correct=eval_result1["is_correct"],
                            pred_wrong_step=eval_result1["pred_wrong_step"],
                            step_explanation=eval_result1["step_explanation"],
                        ),
                        f,
                    )
                    f.write("\n")
            except:
                num_of_cases -= 1
                continue

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("execution time:{:.2f}s\n".format(elapsed_time))

        with open(save_dir + 'eval.jsonl', encoding='utf-8') as f:
            eval_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        wrong_results_step = []
        for eval in eval_datas:
            if eval ["is_correct"] == "no":
                if eval["pred_wrong_step"] == eval["wrong_step"]:
                    correct_results.append(eval)
                else:
                    wrong_results_step.append(eval)
            elif eval ["is_correct"] == "No":
                if eval["pred_wrong_step"] == eval["wrong_step"]:
                    correct_results.append(eval)
                else:
                    wrong_results_step.append(eval)
            else:
                wrong_results.append(eval)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong_step.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results_step, f, ensure_ascii=False, indent=4)
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)}+{len(wrong_results_step)})={len(correct_results) / (len(correct_results) + len(wrong_results)+len(wrong_results_step))}"
        print(result)
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)
    
    elif selected_test == "type_complex" or selected_test == "type_reverse_complex" or selected_test == "type_random_complex":
        while num_of_cases < expected_cases:
            logger.info("###########################")
            logger.info(f"selected_test:{selected_test}")
            logger.info(f"selected_type:{selected_type}")
            logger.info(f"total cases:{num_of_cases}/{expected_cases}")

            start_time = time.time()
            eval_case_chosen = eval_cases[num_of_cases]
            prompt = template.format(*type_initial_cases,eval_case_chosen['question'], eval_case_chosen['transformed_solution'])
            num_of_cases += 1
            #To prevent errors, try several times
            for i in range(5):
                try:
                    eval_result_original = askAPI(prompt, model_name)
                    break
                except:
                    print('\033[1;31m' + "error when askAPI" + '\033[0m')
                    if i == 4:
                        eval_result_original = askAPI(prompt, model_name)
                        # exit()
                    continue
            #print("eval_result", eval_result_original)
            match = re.search(r'{(.*?)}', eval_result_original, re.DOTALL)
            transformed_case = match.group(1)
            # print("transformed_case",transformed_case)
            try:
                eval_result = "{" + transformed_case + "}"
                eval_result1 = ast.literal_eval(eval_result)
                with open(save_dir + 'eval.jsonl', "a") as f:
                    json.dump(
                        dict(
                            **eval_case_chosen,
                            is_correct=eval_result1["is_correct"],
                            pred_wrong_type=eval_result1["pred_wrong_type"],
                            type_explanation=eval_result1["type_explanation"],
                        ),
                        f,
                    )
                    f.write("\n")
            except:
                num_of_cases -= 1
                continue

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("execution time:{:.2f}s\n".format(elapsed_time))

        with open(save_dir + 'eval.jsonl', encoding='utf-8') as f:
            eval_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        wrong_results_type = []
        for eval in eval_datas:
            if eval["is_correct"] == "no":
                if eval["pred_wrong_type"] == eval["wrong_type"]:
                    correct_results.append(eval)
                else:
                    wrong_results_type.append(eval)
            elif eval["is_correct"] == "No":
                if eval["pred_wrong_type"] == eval["wrong_type"]:
                    correct_results.append(eval)
                else:
                    wrong_results_type.append(eval)
            else:
                wrong_results.append(eval)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong_type.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results_type, f, ensure_ascii=False, indent=4)
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)}+{len(wrong_results_type)})={len(correct_results) / (len(correct_results) + len(wrong_results) + len(wrong_results_type))}"
        print(result)
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)

    elif selected_test == "type_reverse_simple" or selected_test == "type_random_simple" or selected_test == "type_simple":
        while num_of_cases < expected_cases:
            logger.info("###########################")
            logger.info(f"selected_test:{selected_test}")
            logger.info(f"selected_type:{selected_type}")
            logger.info(f"total cases:{num_of_cases}/{expected_cases}")

            start_time = time.time()
            eval_case_chosen = eval_cases[num_of_cases]
            prompt = template.format(eval_case_chosen['question'], eval_case_chosen['transformed_solution'])
            num_of_cases += 1
            #To prevent errors, try several times
            for i in range(5):
                try:
                    eval_result_original = askAPI(prompt, model_name)
                    break
                except:
                    print('\033[1;31m' + "error when askAPI" + '\033[0m')
                    if i == 4:
                        eval_result_original = askAPI(prompt, model_name)
                        # exit()
                    continue
            match = re.search(r'{(.*?)}', eval_result_original, re.DOTALL)
            transformed_case = match.group(1)
            try:
                eval_result = "{" + transformed_case + "}"
                eval_result1 = ast.literal_eval(eval_result)
                with open(save_dir + 'eval.jsonl', "a") as f:
                    json.dump(
                        dict(
                            **eval_case_chosen,
                            is_correct=eval_result1["is_correct"],
                            pred_wrong_type=eval_result1["pred_wrong_type"],
                            type_explanation=eval_result1["type_explanation"],
                        ),
                        f,
                    )
                    f.write("\n")
            except:
                num_of_cases -= 1
                continue

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("execution time:{:.2f}s\n".format(elapsed_time))

        with open(save_dir + 'eval.jsonl', encoding='utf-8') as f:
            eval_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        wrong_results_type = []
        for eval in eval_datas:
            if eval["is_correct"] == "no":
                if eval["pred_wrong_type"] == eval["wrong_type"]:
                    correct_results.append(eval)
                else:
                    wrong_results_type.append(eval)
            elif eval["is_correct"] == "No":
                if eval["pred_wrong_type"] == eval["wrong_type"]:
                    correct_results.append(eval)
                else:
                    wrong_results_type.append(eval)
            else:
                wrong_results.append(eval)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong_type.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results_type, f, ensure_ascii=False, indent=4)
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)}+{len(wrong_results_type)})={len(correct_results) / (len(correct_results) + len(wrong_results) + len(wrong_results_type))}"
        print(result)
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)

    elif selected_test == "correction_complex":
        while num_of_cases < expected_cases:
            logger.info("###########################")
            logger.info(f"selected_test:{selected_test}")
            logger.info(f"selected_type:{selected_type}")
            logger.info(f"total cases:{num_of_cases}/{expected_cases}")

            start_time = time.time()
            eval_case_chosen = eval_cases[num_of_cases]
            prompt = template.format(*correction_initial_cases, eval_case_chosen['question'], eval_case_chosen['transformed_solution'])
            num_of_cases += 1
            #To prevent errors, try several times
            for i in range(5):
                try:
                    eval_result_original = askAPI(prompt, model_name)
                    break
                except:
                    print('\033[1;31m' + "error when askAPI" + '\033[0m')
                    if i == 4:
                        eval_result_original = askAPI(prompt, model_name)
                        # exit()
                    continue
            match = re.search(r'{(.*?)}', eval_result_original, re.DOTALL)
            transformed_case = match.group(1)
            eval_result = "{" + transformed_case + "}"
            try:
                eval_result1 = ast.literal_eval(eval_result)
                with open(save_dir + 'eval.jsonl', "a") as f:
                    json.dump(
                        dict(
                            **eval_case_chosen,
                            is_correct=eval_result1["is_correct"],
                            corrected_solution=eval_result1["corrected_solution"],
                            corrected_answer=eval_result1["corrected_answer"],
                            corrected_explanation=eval_result1["corrected_explanation"],
                        ),
                        f,
                    )
                    f.write("\n")
            
            except:
                print(eval_result_original)
                num_of_cases -= 1
                print(eval_result)
                continue

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("execution time:{:.2f}s\n".format(elapsed_time))

        with open(save_dir + 'eval.jsonl', encoding='utf-8') as f:
            eval_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        wrong_correction = []
        for eval_data in eval_datas:
            if eval_data["is_correct"] == "no":
                if eval_data["corrected_answer"] == "none":
                    wrong_correction.append(eval_data)
                elif abs(eval_data["original_answer"] - float(eval_data["corrected_answer"])) < 1e-3:
                    correct_results.append(eval_data)
                else:
                    wrong_correction.append(eval_data)
            elif eval_data["is_correct"] == "No":
                if eval_data["corrected_answer"] == "none":
                    wrong_correction.append(eval_data)
                elif abs(eval_data["original_answer"] - float(eval_data["corrected_answer"])) < 1e-3:
                    correct_results.append(eval_data)
                else:
                    wrong_correction.append(eval_data)
            else:
                wrong_results.append(eval_data)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong_correction.json", "w", encoding='utf-8') as f:
            json.dump(wrong_correction, f, ensure_ascii=False, indent=4)
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)}+{len(wrong_correction)})={len(correct_results) / (len(correct_results) + len(wrong_results) + len(wrong_correction))}"
        print(result)
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)

    elif selected_test == "correction_simple":
        while num_of_cases < expected_cases:
            logger.info("###########################")
            logger.info(f"selected_test:{selected_test}")
            logger.info(f"selected_type:{selected_type}")
            logger.info(f"total cases:{num_of_cases}/{expected_cases}")

            start_time = time.time()
            eval_case_chosen = eval_cases[num_of_cases]
            prompt = template.format(eval_case_chosen['question'], eval_case_chosen['transformed_solution'])
            num_of_cases += 1
            # To prevent errors, try several times
            for i in range(5):
                try:
                    eval_result_original = askAPI(prompt, model_name)
                    break
                except:
                    print('\033[1;31m' + "error when askAPI" + '\033[0m')
                    if i == 4:
                        eval_result_original = askAPI(prompt, model_name)
                        # exit()
                    continue
            # print("eval_result", eval_result_original)
            match = re.search(r'{(.*?)}', eval_result_original, re.DOTALL)
            transformed_case = match.group(1)
            eval_result = "{" + transformed_case + "}"
            try:
                eval_result1 = ast.literal_eval(eval_result)
                with open(save_dir + 'eval.jsonl', "a") as f:
                    json.dump(
                        dict(
                            **eval_case_chosen,
                            is_correct=eval_result1["is_correct"],
                            corrected_solution=eval_result1["corrected_solution"],
                            corrected_answer=eval_result1["corrected_answer"],
                            corrected_explanation=eval_result1["corrected_explanation"],
                        ),
                        f,
                    )
                    f.write("\n")

            except:
                num_of_cases -= 1
                continue

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("execution time:{:.2f}s\n".format(elapsed_time))

        with open(save_dir + 'eval.jsonl', encoding='utf-8') as f:
            eval_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        wrong_correction = []
        for eval_data in eval_datas:
            if eval_data["is_correct"] == "no":
                if eval_data["corrected_answer"] == "none":
                    wrong_correction.append(eval_data)
                elif abs(eval_data["original_answer"] - float(eval_data["corrected_answer"])) < 1e-3:
                    correct_results.append(eval_data)
                else:
                    wrong_correction.append(eval_data)
            elif eval_data["is_correct"] == "No":
                if eval_data["corrected_answer"] == "none":
                    wrong_correction.append(eval_data)
                elif abs(eval_data["original_answer"] - float(eval_data["corrected_answer"])) < 1e-3:
                    correct_results.append(eval_data)
                else:
                    wrong_correction.append(eval_data)
            else:
                wrong_results.append(eval_data)
        with open(save_dir + "correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        with open(save_dir + "wrong_correction.json", "w", encoding='utf-8') as f:
            json.dump(wrong_correction, f, ensure_ascii=False, indent=4)
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)}+{len(wrong_correction)})={len(correct_results) / (len(correct_results) + len(wrong_results) + len(wrong_correction))}"
        print(result)
        with open(save_dir + "result.json", 'w', encoding='utf-8') as f:
            f.write(result)


if __name__ == "__main__":
    fire.Fire(main)
