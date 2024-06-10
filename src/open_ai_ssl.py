# Databricks notebook source
!pip install openai
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


# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
#DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://dbc-0b27798d-9689.cloud.databricks.com/serving-endpoints"
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": "Tell me about Large Language Models"
  }
  ],
  model="databricks-dbrx-instruct",
  max_tokens=256
)

print(chat_completion.choices[0].message.content)

# COMMAND ----------


