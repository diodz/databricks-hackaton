# Databricks notebook source
# MAGIC %pip install langchain_community
# MAGIC %pip install 'crewai[tools]'

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token="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",
    task="text-generation",
)

agent = Agent(
    role="HuggingFace Agent",
    goal="Generate text using HuggingFace",
    backstory="A diligent explorer of GitHub docs.",
    llm=llm,
    max_iter=100
)

# COMMAND ----------

llm.invoke("What is databricks")

# COMMAND ----------

agent

# COMMAND ----------

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key

# You can choose to use a local model through Ollama for example. See https://docs.crewai.com/how-to/LLM-Connections/ for more information.

# os.environ["OPENAI_API_BASE"] = 'http://localhost:11434/v1'
# os.environ["OPENAI_MODEL_NAME"] ='openhermes'  # Adjust based on available model
# os.environ["OPENAI_API_KEY"] ='sk-111111111111111111111111111111111111111111111111'

# You can pass an optional llm attribute specifying what model you wanna use.
# It can be a local model through Ollama / LM Studio or a remote
# model like OpenAI, Mistral, Antrophic or others (https://docs.crewai.com/how-to/LLM-Connections/)
#
# import os
# os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'
#
# OR
#
# from langchain_openai import ChatOpenAI

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
  llm=llm,
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  llm=llm,
  verbose=True,
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
  agents=[agent], # [researcher, writer],
  tasks=[task1, task2],
  verbose=1, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)

# COMMAND ----------

result

# COMMAND ----------

from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv()

# Install the required library for Yahoo Finance News Tool
# !pip install langchain_community

yahoo_finance_news_tool = YahooFinanceNewsTool()

# Define your agents with roles and goals
financial_analyst = Agent(
    role='Financial Analyst',
    goal='Analyze current financial news to identify market trends and investment opportunities',
    backstory="""You are an experienced financial analyst adept at interpreting market data and news
  to forecast financial trends and advise on investment strategies.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[yahoo_finance_news_tool]
)

communications_specialist = Agent(
    role='Corporate Communications Specialist',
    goal='Communicate financial insights and market trends to company stakeholders',
    backstory="""As a communications specialist in a corporate setting, your expertise lies in
  crafting clear and concise messages from complex financial data for stakeholders and the public.""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    # (optional) llm=another_llm
)

# Create tasks for your agents
task1 = Task(
    description="""Review the latest financial news using the Yahoo Finance News Tool. Identify key market trends and potential investment opportunities relevant to our company's portfolio.""",
    agent=financial_analyst
)

task2 = Task(
    description="""Based on the financial analyst's report, prepare a press release for the company. The release should
  highlight the identified market trends and investment opportunities, tailored for our stakeholders and the general public.""",
    agent=communications_specialist
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[financial_analyst, communications_specialist],
    tasks=[task1, task2],
    verbose=2
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)

# COMMAND ----------


