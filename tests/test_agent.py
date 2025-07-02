# test_agent.py

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool

# --------------------------
# ✅ 1️⃣ Il tuo tool con parametri strutturati
# --------------------------

PDF_PATH = "./414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf"

def pdf_query(query: str) -> str:
    """
    Simula una query sul PDF.
    In un caso reale useresti Qdrant o PyPDFLoader.
    """
    print(f"📄 TOOL chiamato --> query: '{query}' | PDF: '{PDF_PATH}'")
    return f"Found answer for '{query}' in PDF '{PDF_PATH}'"

pdf_query_tool = StructuredTool.from_function(pdf_query)

# --------------------------
# ✅ 2️⃣ Prompt super robusto
# --------------------------

def get_prompt_template():
    return PromptTemplate.from_template(
        """
You are an AI agent with access to tools.

TOOLS:
{tools}

You MUST follow this format:

Question: the question you must answer

Thought: always think about what to do next

Action: the action to take, must be one of [{tool_names}]

Action Input: the input to the action, as JSON ONLY. Never write plain text here.

Observation: the result of the action

(Repeat Thought / Action / Action Input / Observation as needed)

Thought: I know the final answer

Final Answer: the final answer to the original question.

---

EXAMPLE

Question: Who is the CEO?

Thought: I should use the PDF tool.

Action: pdf_query

Action Input: {{ "query": "CEO name" }}

Observation: The CEO is John Donahoe.

Thought: I know the final answer.

Final Answer: The CEO is John Donahoe.

---

Begin!

Question: {query}

Thought:{agent_scratchpad}
"""
    )


# --------------------------
# ✅ 3️⃣ Setup e run
# --------------------------

if __name__ == "__main__":
    # LLM locale OLLAMA
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.0,  # 🔑 così segue esattamente l'esempio!
    )

    tools = [pdf_query_tool]
    prompt = get_prompt_template()

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="force",
    )

    # ESECUZIONE
    result = agent_executor.invoke({
        "query": "Who are the executive officers of Nike?"
    })

    print("\n✅ FINAL RESULT:")
    print(result)
