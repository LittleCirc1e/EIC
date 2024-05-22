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

#openai.api_key = ""
#openai.api_base = ""

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
        #client = ZhipuAI(api_key="")
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.01)
        return response.choices[0].message.content
    elif model_name == "gemini-pro":
        #GOOGLE_API_KEY = ''
        generation_config = {"temperature": 0}
        genai.configure(api_key=GOOGLE_API_KEY, transport="rest")
        model = genai.GenerativeModel(model_name="gemini-pro",generation_config=generation_config)
        response = model.generate_content(prompt)
        return response.text


def main(
        expected_cases: int = 1,
        model_name: str = "glm-4",  # gpt-3.5-turbo-1106 or gpt-4-1106-preview
        selected_type: str = "calculation_error",
        dataset: str = "GSM8K",
):
    print(f"selected_type: {selected_type}")
    global type
    num_of_iteration = 1
    num_of_cases = 0
    transformed_cases = []
    transformed_cases1 = []

    save_dir = f"../../data/generated_cases_{dataset}/{selected_type}/{selected_type}_{expected_cases}/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    original_cases_dir = f"./step_data_{dataset}/total.jsonl"
    with open(original_cases_dir, 'r', encoding='utf-8') as file:
        original_cases = [json.loads(line) for line in file]
        print("original_question loaded")

    cases_dir = f"./in_context_learning/{selected_type}/initial_cases.json"
    with open(cases_dir, 'r', encoding="utf8") as file:
        initial_cases = json.load(file)
        print("initial cases loaded")

    templates_dir = f"./in_context_learning/{selected_type}/in_context_learning.json"
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
    
    while num_of_cases < expected_cases:
        logger.info("###########################")
        logger.info(f"selected_type:{selected_type}")
        logger.info(f"iteration:{num_of_iteration}")
        logger.info(f"total cases:{num_of_cases}/{expected_cases}")
        logger.info(f"len of original_cases:{len(original_cases)}")

        num_of_iteration += 1

        start_time = time.time()
        
        original_case_chosen=random.sample(original_cases, 1)
        prompt = template.format(*initial_cases,original_case_chosen[0]['question'],original_case_chosen[0]['answer'])
        text = askAPI(prompt, model_name)
        match = re.search(r'{(.*?)}', text, re.DOTALL)
        transformed_case = match.group(1)
        text2 = "{" + transformed_case + "}"
        try:
            text1=ast.literal_eval(text2)
        except:
            print('\033[1;31m' + f"error when ast.literal_eval" + '\033[0m')
            continue
        
        transformed_cases.append(text)
        with open(save_dir + 'generated_cases.json', 'a', encoding="utf8") as file:
            json.dump(
                text,
                file,
                ensure_ascii=False,
                indent=4
            )
            file.write("\n")

        if text1["original_answer"] != text1["transformed_answer"]:
            transformed_cases1.append(text1)
            with open(save_dir + 'generated_cases_clean.jsonl', 'a', encoding="utf8") as file:
                json.dump(
                    text1,
                    file,
                    ensure_ascii=False
                )
                file.write("\n")
            original_cases.remove(original_case_chosen[0])
            num_of_cases += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("execution time:{:.2f}s\n".format(elapsed_time))

if __name__ == "__main__":
    fire.Fire(main)
