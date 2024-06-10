# Databricks notebook source
!pip install -r ../requirements.txt
%restart_python

# COMMAND ----------

import os
os.environ['OPENAI_API_KEY'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
print(len(os.environ['OPENAI_API_KEY']))

os.environ['OPENAI_API_BASE'] = 'https://dbc-0b27798d-9689.cloud.databricks.com/serving-endpoints'

os.environ['OPENAI_MODEL_NAME'] = 'databricks-dbrx-instruct'

import finance_analyst_crew.py

# COMMAND ----------


