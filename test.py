from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def get_roberta(job_description:str, question:str):
    model_name = "deepset/roberta-large-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': question,
        'context': job_description
    }
    res = nlp(QA_input)
    return res

job_description = '''
"Randstad is looking for an experienced Software engineer for healthcare application development in support of several in-flight and planned capital projects. Individual will work as part of WellMed's Clinical Data Integration team supporting clinical data integration projects. FHIR experience will set you apart from other candidates.Required Skills - BizTalk experience 2+ Development experience with HL7 and/or FHIR healthcare data standards (FHIR will set them apart from others) Health care industry experienceJob Duties - Analyzes, develops, and implements computer application enhancements for end users. Performs all tasks in the development life cycle including requirements analysis, design, development, and testing. Utilizes available programming methodologies, languages and tools, and adheres to governing development and documentation standards, processes and procedures. Supports existing applications including developing fixes as necessary and providing troubleshooting support for complex issues. Performs all other related duties as assigned.Job Requirements - Development experience with BizTalk (2017 or 2020) 2+ Development experience with HL7 and/or FHIR healthcare data standards, formats and transaction processing Development experience with Cloud-based development techniques and tools, including Azure (Logic Apps, Service Bus, Data Factory, Event Grid), Snowflake, Kafka and Kubernetes Development experience with one or more of the following technologies, languages and tools: .NET (Core, Framework, ASP) C# XML JSON SQL Server (2016 or 2019) SSIS, SSRS and T-SQL Visual Studio GitHub ServiceNow Health care industry experience, preferably with provider/care delivery organizations rather than payer/health insurance organizationsDesired Skills & Experience - Experience with electronic medical records (EMR) and healthcare information exchange (HIE) technologies Experience working with service-oriented architecture (SOA) Knowledge of REST APIsRequired Skills : C#
        


        
            Show more
          

          



        
            Show less"
'''
job_description = '''
"SS&C is a global provider of investment and financial services and software for the financial services and healthcare industries. Named to Fortune 1000 list as top U.S. company based on revenue, SS&C is headquartered in Windsor, Connecticut and has 20,000+ employees in over 90 offices in 35 countries. Some 18,000 financial services and healthcare organizations, from the world's largest institutions to local firms, manage and account for their investments using SS&C's products and services.Job DescriptionFull Stack Developer (MEAN)Locations: Hybrid ‚Äì Massachusetts Get To Know The Team:We are seeking a full stack web developer. This position will require strong experience in hands-on development and will participate in all phases of the software development lifecycle. You will work closely with our Product team to understand requirements and business specifications around Portfolio Management, Trading, Analytics, and Investment Accounting.In this position, you will work as part of a larger Scrum development team focused on driving innovation across the Aloha application. You will write Angular, TypeScript, HTML, and CSS code in a Node.JS environment, with a MongoDB data store. You will also attend sprint meetings with team members to define and analyze development requirements and provide development work breakdowns and estimatesWhy You Will Love It Here! Flexibility: Hybrid Work Model & a Business Casual Dress Code, including jeansYour Future: 401k Matching Program, Professional Development ReimbursementWork/Life Balance: Flexible Personal/Vacation Time Off, Sick Leave, Paid HolidaysYour Wellbeing: Medical, Dental, Vision, Employee Assistance Program, Parental LeaveDiversity & Inclusion: Committed to Welcoming, Celebrating and Thriving on DiversityTraining: Hands-On, Team-Customized, including SS&C UniversityExtra Perks: Discounts on fitness clubs, travel and more!What You Will Get To Do:Collaborate with Product Owners to develop the web-based Aloha applicationWrite complete, tested, performant, and documented Angular, TypeScript, HTML, and CSSCreate, enhance, and maintain RESTful APIs using Nest.JS and MongoDBCollaborate with stakeholders, users, build teams, QA, and other development partnersParticipate actively in discussions, presentations and decisions about front-end and API developmentImplement new technologies to maximize application performanceWork on bug fixing and improving application performance What You Will Bring:Bachelor‚Äôs Degree (Computer Science or related degree)2+ years of front-end development experience & 1+ years of Angular development experienceUnderstanding of MEAN application technology stackStrong JavaScript and TypeScript, Angular 2+ frameworks, MongoDB, HTML 5, CSS, Webpack, Babel, NPM, JSON, BootstrapCSS pre-compilers (like Sass and LESS)RESTful APIs, GitContinuous Integration using JenkinsKnowledge of Docker is desirable but not necessaryExcellent conceptual and critical thinking capabilitiesSelf-directed and self-motivated with the ability to take charge or play a supporting roleStrong understanding of product developmentClear written and verbal communications skills (English), including presentation skillsThank you for your interest in SS&C! To further explore this opportunity, please apply through our careers page on the corporate website at www.ssctech.com/careers.Unless explicitly requested or approached by SS&C Technologies, Inc. or any of its affiliated companies, the company will not accept unsolicited resumes from headhunters, recruitment agencies, or fee-based recruitment services. SS&C offers excellent benefits including health, dental, 401k plan, tuition and professional development reimbursement plan. SS&C Technologies is an Equal Employment Opportunity employer and does not discriminate against any applicant for employment or employee on the basis of race, color, religious creed, gender, age, marital status, sexual orientation, national origin, disability, veteran status or any other classification protected by applicable discrimination laws.



        
            Show more
          

          



        
            Show less"
            '''
job_description = ' '.join(job_description.split())
question = "At the very least how many years of work experience are required for this role according to the job description? Answer with a number which should not be a calendar year."
output = get_roberta(job_description=job_description, question=question)
print(output)
