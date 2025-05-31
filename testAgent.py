import pandas as pd # Still used for structuring data if needed, but not for main querying
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any # Added Any
from dotenv import load_dotenv
import os
import re

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 not installed. Please install it: pip install PyPDF2")
    # You might want to exit or handle this more gracefully
    PdfReader = None

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
# --- Global variable to store parsed PDF data ---
PDF_CHUNKS = []

def parse_pdf_to_chunks(pdf_path: str) -> List[str]:
    """
    Parses a PDF and splits it into chunks.
    Parse it heading by heading.
    """
    chunks = []
    if not PdfReader:
        print("PDF parsing skipped as PyPDF2 is not available.")
        return chunks
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                question_blocks = re.split(r'(Question\s*\d+[:.\s])', page_text)
                current_block = ""
                for i, part in enumerate(question_blocks):
                    if i > 0 and question_blocks[i-1].startswith("Question"):
                        if current_block.strip():
                             chunks.append(current_block.strip())
                        current_block = question_blocks[i-1] + part
                    else:
                        current_block += part
                if current_block.strip():
                    chunks.append(current_block.strip())

        if not chunks and full_text:
            chunks.append(full_text.strip())

        if not chunks:
            print(f"Warning: No text chunks extracted from {pdf_path}. Check PDF content and parsing logic.")
    except Exception as e:
        print(f"Error parsing PDF {pdf_path}: {e}")
    return chunks


PDF_FILE_PATH = "Nursing_Quick_Guide.pdf"
PDF_CHUNKS = parse_pdf_to_chunks(PDF_FILE_PATH)


class AgentState(TypedDict):
    user_input: str
    search_keywords: List[str]
    retrieved_chunks: List[str]
    prompt: str
    response: str
    message_history: List[str]
    retry: bool
    retry_count: int
    original_user_input: str

def prepare_retrieval_node(state: AgentState) -> AgentState:
    user_input = state['user_input'].lower()
    # Simple keyword extraction: split by space.
    # For better results, use NLP libraries for more sophisticated keyword extraction.
    keywords = [kw for kw in user_input.split() if len(kw) > 2] # Basic stopword filter
    return {**state, "search_keywords": keywords, "original_user_input": state['user_input']}

# Node to run retrieval from PDF chunks
def retrieve_from_pdf_node(state: AgentState) -> AgentState:
    """
    Retrieves relevant chunks from PDF_CHUNKS based on keywords.
    Replace this with semantic search for better performance.
    """
    keywords = state["search_keywords"]
    retrieved = []
    if not PDF_CHUNKS:
        print("No PDF chunks available for retrieval.")
        return {**state, "retrieved_chunks": ["Error: No PDF content loaded or parsed."]}

    if not keywords:
        retrieved = PDF_CHUNKS[:2] # Return first 2 chunks as a fallback
        print("No keywords for search, returning first few chunks as fallback.")
    else:
        for chunk in PDF_CHUNKS:
            chunk_lower = chunk.lower()
            if any(keyword in chunk_lower for keyword in keywords):
                retrieved.append(chunk)
                if len(retrieved) >= 3:
                    break
    
    if not retrieved:
        retrieved = ["No specific information found for your query in the document. Here's a general overview or the start of the document: " + PDF_CHUNKS[0] if PDF_CHUNKS else "No content available."]
        print("No specific chunks found, providing generic fallback.")


    return {**state, "retrieved_chunks": retrieved}


def prompt_template_node(state: AgentState) -> AgentState:
    separator = "\n\n---\n\n"
    context = separator.join(state["retrieved_chunks"]) 
    user_query = state.get("original_user_input", state['user_input']) # Use original for clarity
    
    history = state.get("message_history", [])
    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n" + "\n".join(history)
    
    prompt = f"""You are a helpful nursing knowledge assistant.
Use the following extracted sections from a nursing Q&A document to answer the user's query.
Focus on directly answering the query based on the provided context.
If the context does not contain the answer, state that the information is not found in the provided excerpts.
Do not make up information beyond the provided context.
{history_text}

Context from the document:
{context}

User query: {user_query}

Answer:"""
    return {**state, "prompt": prompt}

def llm_node(state: AgentState) -> AgentState:
    try:
        result = model.generate_content(state["prompt"])
        response_text = result.text.strip()
        
        # Clean up any code blocks or extra quotation marks from the response
        if response_text.startswith("```python"):
            response_text = response_text[9:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        if response_text.endswith('"""'):
            response_text = response_text[:-3]
        if response_text.startswith('"') and response_text.endswith('"'):
            response_text = response_text[1:-1]
            
        response_text = response_text.strip()
    except Exception as e:
        print(f"Error during LLM call: {e}")
        response_text = "Sorry, I encountered an error trying to generate a response."
    return {**state, "response": response_text}

def review_node(state: AgentState) -> AgentState:
    user_query = state.get("original_user_input", state['user_input'])    # Define the separator outside the f-string
    separator = "\n\n---\n\n"
    review_prompt = f"""
You are a strict reviewer. Read the assistant's response and judge if it:
1. Directly answers the user's query.
2. Uses only the provided context from the document.
3. Avoids making up information.
4. Is clearly formulated.

User Query: {user_query}
Provided Context:
{separator.join(state["retrieved_chunks"])}

Assistant's Response: {state['response']}

Based on these criteria, answer YES or NO. If NO, provide a brief 1-line reason related to the criteria.
Example:
NO - The response includes information not present in the context.
YES - The response is well-supported by the context and answers the query.

Your Verdict (YES/NO and reason if NO):
"""
    try:
        review_result_obj = model.generate_content(review_prompt)
        review_result_text = review_result_obj.text.strip().lower()
    except Exception as e:
        print(f"Error during review call: {e}")
        review_result_text = "no - review failed" # Default to retry if review fails

    if review_result_text.startswith("yes") or state.get("retry_count", 0) >= 1: # Reduced retries for faster demo
        return {**state, "retry": False}

    print("Review failed. Will attempt to improve.")
    # Construct more specific feedback for the retry
    feedback = f"The previous response was not adequate. Reviewer feedback: '{review_result_text}'. Please ensure your new response strictly adheres to the provided document context to answer the query: '{user_query}'."

    # Option 2: Construct a more targeted new prompt for retry
    separator = "\n\n---\n\n"
    context_for_retry = separator.join(state["retrieved_chunks"])
    updated_prompt = f"""You are a helpful nursing knowledge assistant. Your previous attempt to answer was not sufficient.
Reviewer's Feedback: {feedback}
Focus on directly answering the query based on the provided context.
Understand the text yoy have recieved and you can test the user on the text if asked as well.
If the context does not contain the answer, state that the information is not found in the provided excerpts.

Context from the document:
{context_for_retry}

User query: {user_query}

Revised Answer:"""

    return {**state, "prompt": updated_prompt, "retry": True, "retry_count": state.get("retry_count", 0) + 1}


def check_retry_node(state: AgentState) -> str:
    if state.get("retry"):
        return "call_llm"
    else:
        return "track_message"

def track_message_node(state: AgentState) -> AgentState:
    history = state.get("message_history", [])
    # Use original_user_input if available for history
    user_msg = state.get("original_user_input", state['user_input'])
    history.append(f"User: {user_msg}")
    history.append(f"Bot: {state['response']}")
    print("✍️ Message tracked in history.")
    return {**state, "message_history": history}

def keep_history_node(state: AgentState) -> AgentState:
    # No summarization, just pass through the history as is
    print("✍️ History maintained without summarization")
    return state


def output_node(state: AgentState):
    print("\nFinal Response:\n")
    print(state["response"])
    return state

# Build the graph
graph = StateGraph(AgentState)

graph.add_node("prepare_retrieval", prepare_retrieval_node)
graph.add_node("retrieve_chunks", retrieve_from_pdf_node) # Changed from run_query
graph.add_node("create_prompt", prompt_template_node)
graph.add_node("call_llm", llm_node)
graph.add_node("review", review_node)
graph.add_node("track_message", track_message_node)
graph.add_node("keep_history", keep_history_node)
graph.add_node("output", output_node)

graph.set_entry_point("prepare_retrieval")
graph.add_edge("prepare_retrieval", "retrieve_chunks")
graph.add_edge("retrieve_chunks", "create_prompt")
graph.add_edge("create_prompt", "call_llm")
graph.add_edge("call_llm", "review")
graph.add_conditional_edges("review", check_retry_node, {
    "call_llm": "call_llm",
    "track_message": "track_message"
})
graph.add_edge("track_message", "keep_history") # Just maintain history without summarization
graph.add_edge("keep_history", "output") # Output after maintaining history
graph.add_edge("output", END)

agent_executor = graph.compile()

# Initial state
initial_state = {
    "message_history": [],
    "retry": False,
    "retry_count": 0,
}

if __name__ == "__main__":
    if not PDF_CHUNKS:
        print("\n" + "="*50)
        print("WARNING: No data loaded from the PDF. The agent might not function as expected.")
        print("Please ensure '40 Qs.pdf' is present and readable, and PyPDF2 is installed.")
        print("If the PDF is complex, the basic parsing logic might need improvement.")
        print("="*50 + "\n")

    current_state = initial_state.copy()
    #while True:
        # query = input("\nAsk something about the nursing Q&A document (or type 'exit' to quit): ")
        # if query.lower() in ["exit", "quit"]:
        #     print("Goodbye!")
        #     break
        #   # Update only the necessary parts of the state for a new query
        # current_state["user_input"] = query # No .lower() here, let nodes handle casing
        # current_state["retry"] = False # Reset retry for new input
        # current_state["retry_count"] = 0 # Reset retry count for new input        
        
        # try:
        #     result_state = agent_executor.invoke(current_state)
        #     # Persist history back to current_state for the next turn
        #     current_state["message_history"] = result_state.get("message_history", [])
        # except Exception as e:
        #     print(f"\nAn error occurred in the agent execution: {e}")
        #     print("The agent state might be inconsistent. Please consider restarting.")