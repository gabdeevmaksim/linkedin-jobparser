from jobParser import process_linkedin_jobs
from scrape import get_job_details
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--time", type=int, help="Time window of search")
parser.add_argument("--exp", type=int, help="maximum experience allowed for the jobs ")
parser.add_argument("--size", type=str, help="size of the model used to parse the job experience")
parser.add_argument("--model", type=str, help="name of the model used to parse the job experience")

args = parser.parse_args()

source_file_name = 'all_linkedinjobs.csv'
target_file = 'filtered_jobs.csv'

max_experience = 2              # the maximum experience allowed for the jobs
if args.exp:
    max_experience = args.exp

filter_time = 48 * 60 * 60      # filter time is specified in milliseconds
if args.time:
    filter_time = args.time * 60 * 60

model_name = "roberta"          # It can be robert or flan-t5
if args.model:
    model_name = args.model

model_size = "large"            # size of the model used to parse job experience it can be medium or large
if args.size:
    model_size = args.size

# This is a list of roles that are searched for on linkedin
roles = ['Full Stack', '"Software Engineer"', '"Software Developer"', 'Founding Engineer', 'Frontend', 'Backend', 'Data Engineer']
get_job_details(file_name=source_file_name, roles=roles, filter_time=filter_time)
process_linkedin_jobs(file_name=source_file_name, model_name=model_name, model_size=model_size)

df = pd.read_csv(source_file_name)
df['experience'] = pd.to_numeric(df['experience'], errors='coerce')
df = df[df['experience'] <= max_experience]
df['target_url'] = df['job_id'].apply(lambda x: f"https://www.linkedin.com/jobs/view/{x}")
df = df.astype(str)
group_columns = ['company', 'job-title', 'description', 'level', 'html', 'experience']
df['_temp_order'] = range(len(df))
agg_functions = {col: 'min' if col == '_temp_order' else lambda x: ', '.join(x) for col in df.columns.difference(group_columns)}
df = df.groupby(group_columns).agg(agg_functions).reset_index()
df = df.sort_values(by='_temp_order')
df = df.drop(columns=['_temp_order', 'job_id', 'html', 'level', 'description'])

df.to_csv(target_file, mode='w', index=False, header=True, encoding='utf-8', lineterminator='\n')



