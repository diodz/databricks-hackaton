# Databricks notebook source
# MAGIC %pip install 'crewai[tools]'==0.30.11
# MAGIC %pip install --upgrade langchain databricks-sql-connector
# MAGIC %pip install python-dotenv==1.0.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

!openssl s_client -showcerts -servername dbc-0b27798d-9689.cloud.databricks.com -connect dbc-0b27798d-9689.cloud.databricks.com:443 > certout </dev/null
!cat certout
!cat certout |  sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > cert.pem
!cat cert.pem
!pwd

# COMMAND ----------

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

os.environ['SSL_CERT_FILE' ] ='./cert.pem'
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
dbrx_llm = ChatOpenAI(
    model="databricks-dbrx-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=DATABRICKS_TOKEN,
    base_url="https://dbc-0b27798d-9689.cloud.databricks.com/serving-endpoints"
)

search_tool = SerperDevTool()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  # You can pass an optional llm attribute specifying what model you wanna use.
  llm=dbrx_llm,
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  llm=dbrx_llm,
  allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
  expected_output="Full blog post of at least 4 paragraphs",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Peer review example

# COMMAND ----------

!pip install pypdf
dbutils.library.restartPython()

# COMMAND ----------

from pypdf import PdfReader 

def get_text_from_pdf(path):
    reader = PdfReader(path) 
    page = reader.pages[0] 
    text = ''
    for page in reader.pages: 
        text += page.extract_text() 
    return text

# COMMAND ----------

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileReadTool, DirectoryReadTool, PDFSearchTool
from dotenv import load_dotenv
import os
from openai import OpenAI

os.environ['SSL_CERT_FILE' ] ='./cert.pem'
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# os.environ['OPENAI_API_KEY'] = DATABRICKS_TOKEN

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="databricks-dbrx-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=DATABRICKS_TOKEN,
    base_url="https://dbc-0b27798d-9689.cloud.databricks.com/serving-endpoints",
)

# Initialize the tool with a specific file path, so the agent can only read the content of the specified file
file_read_tool = FileReadTool(file_path='uploads/example_paper.pdf')
dir_read_tool = DirectoryReadTool(directory='uploads')

pdf_path = 'uploads/example_paper.pdf'
if not os.path.isfile(pdf_path):
  raise FileNotFoundError(f"The file {pdf_path} does not exist")

search_tool = SerperDevTool()

pdf_text = get_text_from_pdf(pdf_path)

# pdf_tool = PDFSearchTool(pdf='uploads/example_paper.pdf', llm=llm)

# Define your agents with roles and goals
researcher = Agent(
  role='',
  goal="""
    Provide peer review of papers submitted to a journal for publication.
  """,
  backstory="""You work at a leading university.
  Your expertise lies in writing exceptional peer reviews.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=llm
)
writer = Agent(
  role='University professor',
  goal="""
    Provide peer review of papers submitted to a journal for publication.
  """,
  backstory="""You work at a leading university.
  Your expertise lies in writing exceptional peer reviews.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=True,
  llm=llm
)

task1 = Task(
  description=f"""
    Conduct a comprehensive review of the paper.
    Identify key insights, data used, and potential academic impact.
    The content of the paper is provided here:
    `{pdf_text}`
  """,
  expected_output="Full peer review report, including a short summary paragraph, contributions to the literature paragraph, feedback paragraph, review of the data used paragraph, publication decision paragraph and explanation of how the authors can improve the work paragraph.",
  agent=researcher
)

task2 = Task(
  description="""
    Based on the comprehensive review and your expert knowledge, develop a detailed peer review report of the paper. The review should include the following sections:
    1. **Overall Evaluation**: Provide a general assessment of the paper's quality, originality, and relevance to the field.
    2. **Detailed Feedback**: Offer in-depth comments on each section of the paper (e.g., introduction, literature review, methodology, results, discussion, conclusion). Highlight strengths and identify specific areas for improvement.
    3. **Data Analysis**: Examine the data used in the study. Evaluate the quality, reliability, and appropriateness of the data. Identify any potential issues such as biases, limitations, or errors. Suggest ways to address these issues if applicable.
    4. **Methodology Assessment**: Analyze the methods used in the paper. Discuss whether the methods are appropriate for the research questions posed and if they are applied correctly. Recommend any alternative methods if necessary.
    5. **Results Interpretation**: Assess how well the results are presented and interpreted. Check for any overgeneralization or misinterpretation of the data. Suggest improvements in the presentation of results if needed.
    6. **Discussion and Conclusion Review**: Evaluate the discussion and conclusion sections. Ensure that the conclusions are supported by the data and analysis. Check for any unsupported claims or speculative statements.
    7. **Recommendations**: Based on the detailed feedback, recommend one of the following publication decisions:
       - **Reject**: Explain the reasons for rejection, focusing on critical flaws or deficiencies.
       - **Revise and Resubmit**: Outline the major revisions required and suggest how the authors can address the identified issues.
       - **Accept for Publication**: Justify why the paper meets the standards for publication without further major revisions.

    The final report should be approximately 5000 words, providing a balanced and thorough evaluation.

    Use the analysis provided as a foundation for your review.
  """,
  expected_output="Full peer review report, including a short summary paragraph, contributions to the literature paragraph, feedback paragraph, review of the data used paragraph, publication decision paragraph and explanation of how the authors can improve the work paragraph.",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
