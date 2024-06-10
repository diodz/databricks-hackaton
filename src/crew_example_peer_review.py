import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileReadTool, DirectoryReadTool, PDFSearchTool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the tool with a specific file path, so the agent can only read the content of the specified file
file_read_tool = FileReadTool(file_path='uploads/10-K_-_Marsh__Mclennan_Companies_INC_-_02-12-2024.pdf')
dir_read_tool = DirectoryReadTool(directory='codespaces-flask/output')

SERPER_API_KEY = os.getenv('SERPER_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# You can choose to use a local model through Ollama for example. See https://docs.crewai.com/how-to/LLM-Connections/ for more information.


search_tool = SerperDevTool()
llm_gemini=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1,
                                  google_api_key=GEMINI_API_KEY)

pdf_tool = PDFSearchTool(pdf='uploads/10-K_-_Marsh__Mclennan_Companies_INC_-_02-12-2024.pdf', llm=llm_gemini)

# Define your agents with roles and goals
researcher = Agent(
  role='University professor',
  goal='Provide peer review of papers submitted to a journal for publication',
  backstory="""You work at a leading university.
  Your expertise lies in writing exceptional peer reviews.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool, pdf_tool],
  # You can pass an optional llm attribute specifying what model you wanna use.
  # It can be a local model through Ollama / LM Studio or a remote
  # model like OpenAI, Mistral, Antrophic or others (https://docs.crewai.com/how-to/LLM-Connections/)
  #
  # import os
  
  #
  # OR
  #
  # from langchain_openai import ChatOpenAI
  llm=llm_gemini
)
writer = Agent(
  role='University professor',
  goal='Provide peer review of papers submitted to a journal for publication',
  backstory="""You work at a leading university.
  Your expertise lies in writing exceptional peer reviews.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=True,
  tools=[pdf_tool],
  llm=llm_gemini
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive review of the paper comprised of the files in the directory.
  Identify key insights, data used, and potential academic impact.""",
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