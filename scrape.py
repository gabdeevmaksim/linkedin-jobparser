import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from urllib.parse import quote, unquote
import time
from jobParser import parse_title, get_model


MAX_RETRIES = 5 # number of retries while looking for jobs of a particular role

# if any of these keywords exist in the job description that job is ignored
keywords = [
        "U.S. Citizenship",
        "U.S. Only",
        "US Only",
        "TS/SCI",
        "Polygraph",
        "security clearance",
        "Defence",
        "Aerospace",
        "US Work Authorization",
        "U.S. Work Authorization",
        "US Citizen",
        "U.S. Citizen",
        "U.S citizen",
        "Green Card",
        "Clearance Required",
        "Secret clearance"
    ]

# This is a list of companies that are ignored 
blacklist = [
            'Akraya', 'Reperio', 'Latitude', 'Get It', 'CACI', 'Actalent', 'Test Company', 'Test', 'Infoscitex Corporation',
            'Peterson Technologies', 'Embedded', 'DigiFlight', 'Ursus', 'Flint Hills', 'Belay Technologies', 'Jobot',
            'SynergisticIT', 'Altamira', 'Professional Diversity Network', 'Lockheed Martin', 'TS/SCI', 'SAIC', 'Dice',
            'Intern', 'FOTOMILL STUDIOS LIMITED', 'BAE Systems', 'Fingerprint for Success', 'L3Harris Technologies',
            'Northrop Grumman', 'RemoteWorker US', 'ManTech', 'Talentify.io', 'Lead', 'Sr.', 'Senior', 'ClearanceJobs',
            'Braintrust', 'Apex Systems', 'Diverse Lynx', 'ClickJobs.io', 'Team Remotely Inc', 'LiveMarket', 'The Judge Group', 'TEKsystems',
            "Andriod", 'Raytheon', 'Epic', 'Honeywell', 'Systems Engineer', 'Systems Reliability Engineer', 'Tata Consultancy Services',
            'Control Engineer', 'Collins Aerospace', 'Georgia Tech', 'System Engineer', 'Systems Software Engineer', 'Leidos',
            'Site Reliability Engineer', 'Piper Companies', 'ios', 'Energy Jobline', 'Firmware', 'Anduril', '.NET', 'Pratt & Whitney',
            'Metrea', 'Dev10', 'Patterned Learning Career', 'get.it', 'HireMeFast', 'Phoenix Recruitment LLC', 'Genie Healthcare',
            'Sicredi', 'TekWissen', 'Creative Financial Staffing', 'Proconex', 'American Cruise Lines', 'eteam', 'Primary Talent Partners',
            'Fidelity TalentSource', 'Cottingham & Butler', 'Phoenix Recruitment', 'Wabtec Corporation', 'Oak Ridge National Laboratory',
            'Luminate', 'Freddie Mac', 'Prudential Financial', 'VPNforAndroid', 'Almac Group', 'Hertz', 'Jobs via eFinancialCareers', 'TalentBurst',
            'Procter & Gamble', 'Medpace', 'Robert Half', 'Russell Tobin', 'Genesis10', 'Aditi Consulting', 'Skiltrek', 'SPECTRAFORCE', 'ZetaChain',
            'Infinite Computer Solutions', 'Listopro', 'RIT Solutions, Inc.', 'Crystal Equation Corporation'
        ]

def load_existing_jobIDs(file_name):
    if not os.path.exists(file_name):
        return set()
    try:
        df = pd.read_csv(file_name)
        return set(df['job_id'])
    except FileNotFoundError:
        return set()

def scrape_job_postings(job_ids, filter_time, roles):
    jobs = []
    target_url='https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords="{}"&location=United%20States&geoId=103644278&f_E=2&f_TPR=r{}&start={}'
    
    roles = [quote(role) for role in roles]
    # relevance = []
    # tokenizer, model = get_model("large")
    total = 0
    for role in roles:
        alljobs_on_this_page={}
        retries = 0
        i = 0
        print(f"Searching for jobs of {unquote(role)} roles...")
        
        while True:
            try:
                time.sleep(1)
                res = requests.get(target_url.format(role, filter_time, i))
                soup = BeautifulSoup(res.text,'html.parser')
                alljobs_on_this_page=soup.find_all("li")
                if len(alljobs_on_this_page) == 0:
                    raise ValueError('No jobs found on this page')
                i += len(alljobs_on_this_page) + 1
                print(f"Try[{retries + 1}/{MAX_RETRIES}]: Scraped {i} jobs", end='\r', flush=True)
                for x in range(0,len(alljobs_on_this_page)):
                    job_id = alljobs_on_this_page[x].find("div",{"class":"base-card"}).get('data-entity-urn').split(":")[3]
                    jobcard = alljobs_on_this_page[x].find("div",{"class":"base-card"}).text.strip()
                    job_title = alljobs_on_this_page[x].find("h3", {"class": "base-search-card__title"}).text.strip()
                    keywords = ['engineer', 'developer', 'scientist', 'software', 'frontend', 'backend']
                    flag = True
                    for keyword in keywords:
                        if keyword in jobcard.lower():
                            flag = False
                            break
                    all_roles = ', '.join([unquote(role) for role in roles])
                    # answer = parse_title(tokenizer, model, job_title, f"Is this job title relevant to any of these roles ${all_roles}")
                    # relevance.append([job_title, answer])
                    if int(job_id) in job_ids:
                        flag = True
                    for blocked in blacklist:
                        if blocked.lower() in jobcard.lower():
                            flag = True
                            break
                    if flag == False:
                        jobs.append(job_id)
            except Exception as e:
                if retries < MAX_RETRIES:
                    retries += 1
                    continue
                print()
                break
        total += i
    # df1 = pd.DataFrame(relevance)
    # df1.columns = ["Job title", "Relevance"]
    # df1.to_csv("relevance.csv", index=False) 
    print("Jobs filtered:", len(set(jobs)), " out of:", total)
    return list(set(jobs))

def progress_bar(index, length):
    progress_length = 100
    percentage = (index + 1) / length
    num_equals = int(percentage * progress_length)
    progress = '#' * num_equals + ">" + '=' * (progress_length - num_equals)
    print("[" + "".join(progress) + "]", f"{percentage * 100:.2f}%", end='\r', flush=True)

def scrape_job_detail(job_id):
    target_url = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{}"
    try:
        time.sleep(1)
        resp = requests.get(target_url.format(job_id))
        soup = BeautifulSoup(resp.text, 'html.parser')
        o = {"job_id": job_id}
        o["company"] = soup.find("div", {"class": "top-card-layout__card"}).find("a").find("img").get('alt', None)
        o["job-title"] = soup.find("div", {"class": "top-card-layout__entity-info"}).find("a").text.strip()
        o["level"] = soup.find("ul", {"class": "description__job-criteria-list"}).find("li").text.replace("Seniority level", "").strip()
        o["description"] = soup.find("div", {"class": "description__text description__text--rich"}).text.strip()
        o["html"] = str(soup.find(class_="description__text description__text--rich"))
        for keyword in keywords:
            if keyword.lower() in o["description"].lower():
                return {}
            if keyword.lower() in o["job-title"].lower():
                return {}
        return o
    except Exception as e:
        return None

def get_job_details(file_name, roles, filter_time=86400):
    existing_job_ids = load_existing_jobIDs(file_name)
    job_ids = scrape_job_postings(existing_job_ids, filter_time, roles)
    print("Fetching job descriptions...")
    job_details = []
    failed_job_ids = []
    MAX_RETRIES = 3
    for retry in range(MAX_RETRIES):
        for i, job_id in enumerate(job_ids):
            progress_bar(i, len(job_ids))
            job_detail = scrape_job_detail(job_id)
            if job_detail == None:
                failed_job_ids.append(job_id)
            elif job_detail == {}:
                continue
            else:
                job_details.append(job_detail)
        if len(failed_job_ids) == 0: break
        job_ids = failed_job_ids
        failed_job_ids = []
        print("\nRetry {}: retrying {} jobs".format(retry + 1, len(job_ids)))

    df = pd.DataFrame(job_details)
    if os.path.exists(file_name):
        # df.to_csv(file_name, mode='a', index=False, header=False, encoding='utf-8')
        with open(file_name, 'a', encoding='utf-8') as f:
            f.seek(0, os.SEEK_END)
            df.to_csv(f, index=False, header=False)
    else:
        df.to_csv(file_name, mode='w', index=False, header=True, encoding='utf-8')
    print("\nDone scraping")
    return df
