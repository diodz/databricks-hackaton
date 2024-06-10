# Databricks notebook source
!pip install openai
!pip install langchain_openai
!pip install langchain_community
!pip install crewai[tools]
%restart_python

# COMMAND ----------

!openssl s_client -showcerts -servername dbc-0b27798d-9689.cloud.databricks.com -connect dbc-0b27798d-9689.cloud.databricks.com:443 > certout </dev/null
!cat certout
!cat certout |  sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > cert.pem
!cat cert.pem
!pwd

# COMMAND ----------

from openai import OpenAI
import os
os.environ['SSL_CERT_FILE' ] ='./cert.pem'
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

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

# COMMAND ----------

output = llm.invoke("what is databricks")

# COMMAND ----------

output

# COMMAND ----------

def get_text_from_pdf(path):
    from pypdf import PdfReader 
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
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
import os

from openai import OpenAI
import os
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

# load_dotenv()
os.chdir('/Workspace/Users/nitinappiah@gmail.com/databricks-hackaton-nitin/')
print(os.getcwd())
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
  role='University professor',
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
  # tools=[pdf_tool],
  llm=llm
)

# Create tasks for your agents
task1 = Task(
  description=f"""
    Conduct a comprehensive review of the paper.
    Identify key insights, data used, and potential academic impact.
    The content of the paper is provided here:
    `{pdf_text}`
  """,
  expected_output="Full analysis report",
  agent=researcher
)

task2 = Task(
  description="""Using your knowledge and the analysis provided, develop an expert peer review of the paper.
  Then make a decision on publication, such as reject, revise and resubmit, or accept for publication.""",
  expected_output="Full peer review report of approximately 2000 words, with publication decision.",
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


