
from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage,AIMessage
 #from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent,AgentExecutor
from todoist_api_python.api import  TodoistAPI

load_dotenv()

todoist_api_key = os.getenv('TODOIST_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

todoist = TodoistAPI(todoist_api_key)

@tool
def add_task(task,disp=None):
       """Add a new task to the user task's list. Use this when the user wants to add or create a task"""
       todoist.add_task(content=task, description=disp)

@tool
def show_task():
    """Returns a list of all existing tasks for the user."""
    result_paginator = todoist.get_tasks()
    tasks = []
    for task_list in result_paginator:
        for task in task_list:
            tasks.append(task.content)


    return tasks


tools = [add_task,show_task]

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key =gemini_api_key,
    temperature=0.3
)


system_prompt = """You are going to be a useful assistance
i want the existing tasks to be in bullet format
 """
prompt =ChatPromptTemplate ([
    ('system',system_prompt),
    MessagesPlaceholder("history"),
    ('user',"{input}"),
    MessagesPlaceholder("agent_scratchpad"),

])

#chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)


#response =  chain.invoke({"input":user_input})

history= []
while True:
    user_input = input("you: ")
    response = agent_executor.invoke({"input": user_input, "history": history})
    print(response['output'])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(response['output']))

