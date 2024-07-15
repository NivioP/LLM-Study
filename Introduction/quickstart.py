from dotenv import find_dotenv, load_dotenv
#from langchain.llms import OpenAI
from langchain_openai import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI

# This will be the first step to studying LLMs and LangChain
# I will split the functions here to understand each part 


# Load environment variables
load_dotenv(find_dotenv())

# --------------------------------------------------------------
# LLMs: Get predictions from a language model
# --------------------------------------------------------------

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
prompt = "Write a cake recipe"
print(llm(prompt))


# --------------------------------------------------------------
# Prompt Templates: Manage prompts for LLMs
# --------------------------------------------------------------

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is the best way to learn how to {product}?",
)

prompt.format(product="Fight Muay Thai")


# --------------------------------------------------------------
# Memory: Add State to Chains and Agents
# --------------------------------------------------------------

llm = OpenAI()
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hello IA!")
print(output)

output = conversation.predict(
    input="I'm all right! Just having a conversation and learning AI."
)
print(output)


# --------------------------------------------------------------
# Chains: Combine LLMs and prompts in multi-step workflows
# --------------------------------------------------------------

llm = OpenAI()
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("Action figures for engineers"))
# This for exemple could be a website that generates names for companies
# -> CompanyNameGenarator.ai

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write an email subject for this topic {topic}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("AI Session"))


# --------------------------------------------------------------
# Agents: Dynamically Call Chains Based on User Input
# --------------------------------------------------------------


llm = OpenAI()

get_all_tool_names()
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Now let's test it out!
result = agent.run(
    "In what year was python released? and who is the original creator? Multiply the year by 3"
)
print(result)

# Now let's test it out!
result = agent.run(
    "In what year was Tesla released and who is the original creator? Multiply the year by 3"
)
print(result)

# Now let's test it out!
result = agent.run(
    "In what year was Tesla born? and who is the original creator? Multiply the year by 3"
)
print(result)
