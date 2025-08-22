from crewai import Agent as CrewAgent, Task, Crew
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.utilities import SerpAPIWrapper
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ Base LLM
llm = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"))

# ✅ FAISS Retriever
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Tools
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
tools = [
    Tool(name="Search", func=search.run, description="Searches for current events or topics"),
    Tool(name="LongTermQA", func=qa_chain.run, description="Answers from FAISS-stored memory")
]

# ✅ Define CrewAI Agents
planner = CrewAgent(
    role="Planner",
    goal="Break down the user's query into steps",
    backstory="Expert in task decomposition and strategy",
    llm=llm,
)

researcher = CrewAgent(
    role="Researcher",
    goal="Research using tools and gather relevant facts",
    backstory="Knows how to use internet tools and memory to gather accurate data",
    tools=tools,
    llm=llm,
)

analyzer = CrewAgent(
    role="Analyzer",
    goal="Summarize and conclude with a useful answer",
    backstory="Can interpret raw data and craft useful answers",
    llm=llm,
)

# ✅ Define the Tasks
task1 = Task(description="Break this query into actionable steps.", agent=planner)
task2 = Task(description="Use memory and tools to gather information.", agent=researcher)
task3 = Task(description="Analyze everything and answer the user's query clearly.", agent=analyzer)

# ✅ Set up the Crew
crew = Crew(
    agents=[planner, researcher, analyzer],
    tasks=[task1, task2, task3],
    verbose=True
)

# ✅ Execute
if __name__ == "__main__":
    user_query = input("🔹 Ask your question: ")
    final_output = crew.run(input=user_query)
    print("\n✅ Final Answer:\n", final_output)
