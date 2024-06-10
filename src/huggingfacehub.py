from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/databricks/dbrx-instruct",
    huggingfacehub_api_token="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",
    task="text-generation",
)

agent = Agent(
    role="HuggingFace Agent",
    goal="Generate text using HuggingFace",
    backstory="A diligent explorer of GitHub docs.",
    llm=llm
)