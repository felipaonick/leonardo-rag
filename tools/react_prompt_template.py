from langchain.prompts import PromptTemplate

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

IMPORTANT:
- The input PDF document is dynamic: it can change every time.
- Always pass the `pdf_path` in the Action Input together with the `query`
- If the Observation already contains the answer, do NOT write an Action or Action Input.
- In this case, write only:
  Thought: I know the final answer
  Final Answer: [your final answer]

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



if __name__ == "__main__":
    prompt_template = get_prompt_template()
    print(prompt_template.format(
        tools="pdf_query",
        tool_names="pdf_query",
        query="Who is the CEO?",
        agent_scratchpad=""
    ))
