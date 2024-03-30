import pandas as pd
from bs4 import BeautifulSoup
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import threading
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForQuestionAnswering, AutoTokenizer, pipeline

logging.disable(logging.CRITICAL + 1)

def get_roberta(job_description:str, question:str, model, tokenizer):
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {
        'question': question,
        'context': job_description
    }
    res = nlp(QA_input)
    if res['score'] > 0.001:
        return res['answer']
    return "No answer"
    
def get_flan(job_description:str, question:str, model, tokenizer):
    input_text = f"Job Description: {job_description} question={question}, Answer:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=20, max_length=50)
    result = BeautifulSoup(tokenizer.decode(outputs[0]), "html.parser").get_text(separator=' ', strip=True)
    return result

def text_to_num(text):
    num_dict = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    return num_dict.get(text.lower())

def find_highest_lowest(s):
    number_patterns = re.findall(r'\b[a-zA-Z]+\b|\d+', s)
    processed_numbers = []
    for num in number_patterns:
        if num.isalpha():
            number = text_to_num(num)
            if number is not None:
                processed_numbers.append(number)
        else:
            processed_numbers.append(int(num))
    processed_numbers = [x for x in processed_numbers if x <= 10]
    if not processed_numbers:
        return 0, 0
    return max(processed_numbers), min(processed_numbers)
    
def get_model(name:str, size:str):
    if size == "medium": size = "base"
    print(f"Parsing job experience information with {name}-{size} model...\n")
    if name == "roberta":
        model = AutoModelForQuestionAnswering.from_pretrained(f"deepset/roberta-{size}-squad2")
        tokenizer = AutoTokenizer.from_pretrained(f"deepset/roberta-{size}-squad2")
        func = lambda job_description, question: get_roberta(job_description=job_description, question=question, model=model, tokenizer=tokenizer)
    elif name == "flan-t5":
        tokenizer = T5Tokenizer.from_pretrained(f"google/flan-t5-{size}", legacy=True)
        model = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{size}")
        func = lambda job_description, question: get_flan(job_description=job_description, question=question, model=model, tokenizer=tokenizer)
    return tokenizer, model, func

def parse_title(tokenizer, model, job_title, question):
    input_text = f"Job title: {job_title} question={question}, Answer with yes or no:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=20, max_length=50)
    result = BeautifulSoup(tokenizer.decode(outputs[0]), "html.parser").get_text(separator=' ', strip=True)
    return result

def progress_bar(current:int, total:int):
    progress_length = 100
    percentage = int(current / total * 100)
    progress = '#' * percentage + '>' + '=' * (progress_length - percentage)
    bar = f"[{progress}] {percentage}% completed; rows processed:{current}"
    sys.stdout.write('\r' + bar)
    sys.stdout.flush()

def process_row(row, index, parser, column_name:str):
    if pd.notnull(row[column_name]):
        return index, None
    question = "How many years of work experience are required for this role according to the job description? Answer with a number."
    job_description = row["description"]
    job_description = ' '.join(job_description.split())
    text = parser(job_description, question) 
    _, low = find_highest_lowest(text)
    return index, low


def process_linkedin_jobs(file_name, model_name:str, model_size:str):
    df = pd.read_csv(file_name)
    exp_column_name = "experience"
    if exp_column_name not in df.columns:
        df[exp_column_name] = pd.Series(dtype='int64')

    _, _, parser = get_model(model_name, model_size)
    number_of_threads = 10

    with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        futures = [executor.submit(process_row, row, index, parser, exp_column_name) for index, row in df.iterrows()]
        for i, future in enumerate(as_completed(futures)):
            progress_bar(total=len(df), current=i)
            index, result = future.result()
            if result != None:
                df.at[index, exp_column_name] = result
            progress_bar(current=i + 1, total=len(df))

    df.to_csv(file_name, index=False)
    sys.stdout.write('\r')
