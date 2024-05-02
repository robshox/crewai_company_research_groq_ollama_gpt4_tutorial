import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileReadTool
from langchain_openai import ChatOpenAI

os.environ["SERPER_API_KEY"] = "yourkeyhere"  # serper.dev API key

llm = ChatOpenAI(
    model="crewaiv2-llama3", base_url="http://localhost:11434/v1", api_key="NA"
)


search_tool = SerperDevTool()
file_read_tool = FileReadTool(file_path="./emp_details.csv")

# Define your agents with roles and goals
researcher = Agent(
    role="Data Research",
    goal="Gather information on Engineering Companies",
    backstory="""You are a research and data expert. Using existing samples you find similar info on new companies via the search tool to pass to the data entry agent """,
    verbose=True,
    allow_delegation=False,
    tools=[file_read_tool, search_tool],
    max_rpm=2,  # Groq API rate limit
    llm=llm,
)

data_entry = Agent(
    role="Data Entry",
    goal="Enter data from researcher agent into the file",
    backstory="""You are a data entry expert. Taking the data from the research agent you add it to the file as a new column """,
    verbose=True,
    allow_delegation=False,
    tools=[file_read_tool],
    llm=llm,
)

# Create tasks for your agent
research_task = Task(
    description="""Using the example file provided, research answers for each of the rows for a company called {company}.""",
    expected_output="New inputs for {company} so the data entry agent can add them to the file.",
    tools=[file_read_tool, search_tool],
    allow_delegation=False,
    agent=researcher,
    output_file="empDetails_output_gpt4.csv",  # Example of output customization
)

entry_task = Task(
    description="""Take the research from the researcher agent and add it to the file for {company}.""",
    expected_output="New inputs for {company} as a new column similar to the sample data",
    tools=[file_read_tool],
    allow_delegation=False,
    agent=data_entry,
    output_file="emp_details.csv",  # Example of output customization
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, data_entry],
    tasks=[research_task, entry_task],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff(inputs={"company": "blueorigin.com"})

print("######################")
print(result)