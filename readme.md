## Python Code for LinkedIn Job Scraping and Filtering

This code automates the process of finding relevant job postings on LinkedIn. 

**Functionality Breakdown**

1. **Argument Parsing**

* Uses `argparse` to handle command-line arguments for customizing:
    * `--time`: Time window (in hours) for filtering job postings.
    * `--exp`: Maximum years of experience allowed for the jobs.
    * `--size`: Size of the machine learning model used for parsing job experience ("medium" or "large").
    * `--model`: Name of the model used for parsing job experience ("roberta" or "flan-t5").

2. **File Handling and Configuration**

* `source_file_name`: CSV file to store all scraped LinkedIn job data.
* `target_file`: CSV file to save filtered job data.
* `max_experience`, `filter_time`, `model_name`, `model_size`: Default values for script parameters (can be overridden by command-line arguments).

3. **Job Scraping and Filtering**

* `roles`: List of job roles to search for on LinkedIn.
* `get_job_details(...)`:
    * Scrapes job postings from LinkedIn based on specified roles and `filter_time`.
    * Saves scraped data to `source_file_name`.
* `process_linkedin_jobs(...)`:
    * Processes scraped job data in `source_file_name`.
    * Uses a machine learning model to extract required years of experience from job descriptions.
    * Adds extracted experience information to the CSV file.

4. **Data Processing and Refinement**

* Reads data from `source_file_name` into a pandas DataFrame.
* Converts 'experience' column to numeric values.
* Filters out jobs with experience exceeding `max_experience`.
* Adds a 'target_url' column with direct links to LinkedIn job postings.
* Aggregates data by grouping and combining 'target_url' values.
* Sorts DataFrame and drops unnecessary columns.

5. **Saving Filtered Results**

* Saves filtered and processed job data to `target_file`.

**Key Features**

* Searches for specific job roles.
* Filters out jobs based on experience level.
* Uses machine learning to extract years of experience from job descriptions.
* Provides a clean CSV file with links to filtered job postings.


### Install  Dependencies

```
pip install -r requirements.txt
```

### Run the code with this command
```
python3 valid_job_extractor.py --time 24 --exp 2 --size small
```
#### Arguments
- `size` can be medium or large
- `time` is the time in hours for the oldest allowed job
- `exp` is the number of maximum number of years of experience allowed for a job
- `model` is for specifying which model to use for inferencing(currently supports roberta and flan-t5).