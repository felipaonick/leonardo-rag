from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from tools.react_prompt_template import get_prompt_template
from tools.pdf_query_tools import pdf_query
import os


def agent(pdf_path: str, query: str, ollama_model: str = "llama3.2:3b"):
    """
    Create and run an agent with either Groq or OLLAMA LLM

    Args:
        pdf_path (str): The pdf document from which get the chunks
        query (str): The user's query
        ollama_model (str): OLLAMA model to use (defautl: llama3.2:3b)
    """

    
    os.environ["USE_OLLAMA"] = "true"
    os.environ["OLLAMA_MODEL"] = ollama_model

    base_url = "http://host.docker.internal:11434"

    LLM = ChatOllama(
        model=ollama_model,
        base_url=base_url,
        temperature=0.1,
        timeout=120,  # Increase timeout for local models
    )

    print(f"Using OLLAMA model: {ollama_model} at {base_url}")


    tools = [pdf_query]

    prompt_template = get_prompt_template()

    agent = create_react_agent(
        LLM, 
        tools, 
        prompt_template
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10, # Limit iterations for local models
        early_stopping_method="force"
    )


    try:
        result = agent_executor.invoke({"query": query, "pdf_path": pdf_path})
        return result['output']
    
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return f"Sorry, I encountered an error while processing your query: {str(e)}"
    

def get_available_ollama_models():
    """Get list of available OLLAMA models"""
    try:
        import requests
        url = "http://host.docker.internal:11434"
        print(f"DEBUG: Checking Ollama models at {url}/api/tags")
        response = requests.get(f"{url}/api/tags", timeout=5)

        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model['name'] for model in models]
        else:
            return []
    except Exception as e:
        print(f"Could not connect to OLLAMA: {e}")
        return []

    

def check_ollama_connection():
    """
    Check if OLLAMA is running and accessible
    """
    try:
        import requests
        url = "http://host.docker.internal:11434"
        print(f"DEBUG: Checking Ollama at {url}/api/version")
        response = requests.get(f"{url}/api/version", timeout=5)
        print(f"DEBUG: Ollama response: {response.status_code} {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"DEBUG: Ollama check failed: {e}")
        return False
    

if __name__ == "__main__":

    res = get_available_ollama_models()

    print(res)