import os
import sys
import logging
import datetime
from dotenv import load_dotenv
from llm_ai import get_gemini_llm
from chunks import LocalChunkStorage, process_pdf_to_chunks
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StdOutCallbackHandler
from flask import Flask, render_template, request, jsonify

# Configure logging
log_filename = os.path.join("agent_logs", f"bhagavad_gita_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
os.makedirs("agent_logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration from .env file
load_dotenv()

# Define paths
pdf_path = os.path.join("Data", "The_Bhagavad_Gita.pdf")
storage_dir = "data_blocks"

# Ensure PDF exists
if not os.path.exists(pdf_path):
    logger.error(f"PDF file not found at {pdf_path}")
    print(f"Error: PDF file not found at {pdf_path}")
    sys.exit(1)

# Initialize the local storage
def initialize_storage():
    """Initialize or load the local storage"""
    logger.info("Initializing local storage...")
    
    # Check if data needs to be processed
    if not os.path.exists(os.path.join(storage_dir, "index.json")):
        logger.info("Processing PDF and creating chunks...")
        process_pdf_to_chunks(pdf_path, storage_dir)
    else:
        logger.info("Using existing chunks from storage")
    
    # Initialize embedding model
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Using sentence-transformers/all-MiniLM-L6-v2 embedding model")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise

    # Initialize local storage
    storage = LocalChunkStorage(storage_dir, embeddings)
    
    return storage

# Custom callback handler to log agent steps
class LoggingCallbackHandler(StdOutCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        """Log agent action decision"""
        logger.info(f"Agent decided to use: {action.tool}")
        logger.info(f"Tool input: {action.tool_input}")
        
    def on_agent_finish(self, finish, **kwargs):
        """Log agent finish"""
        logger.info(f"Agent finished: {finish.return_values}")

# Create a simple calculator tool
def simple_calculator(query):
    """Simple calculator function"""
    try:
        logger.info(f"Calculator input: {query}")
        result = str(eval(query))
        logger.info(f"Calculator result: {result}")
        return result
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return "Calculation error."

def setup_rag_agent():
    """Set up the RAG agent with tools"""
    # Initialize the local storage
    storage = initialize_storage()
    
    # Initialize the Gemini LLM
    llm = get_gemini_llm()
    
    # Create the RAG function that uses local storage
    def run_rag_pipeline(query):
        """
        Run the Retrieval-Augmented Generation pipeline using local storage
        """
        logger.info(f"RAG Query: {query}")
        
        # Check if this is an "explain more" request
        if query.lower().startswith("explain more:"):
            # Extract the original query
            original_response = query.split(":", 1)[1].strip()
            explain_prompt = f"Please elaborate on the following information from the Bhagavad Gita in about 50-60 words, providing deeper insights: {original_response}"
            
            logger.info(f"Explain more request for: {original_response[:50]}...")
            # Call Gemini directly for elaboration
            response = llm.invoke(explain_prompt)
            return {"output_text": response.content}
        
        # Get relevant documents using similarity search
        retrieved_chunks = storage.similarity_search(query, k=3)
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        if not retrieved_chunks:
            return {"output_text": "I couldn't find any relevant information in the Bhagavad Gita to answer this question."}
        
        # Format retrieved chunks for LLM
        from langchain_core.documents import Document
        docs = []
        for chunk in retrieved_chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            docs.append(doc)
        
        # Set up QA chain
        qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        
        # Run the chain
        result = qa_chain({"input_documents": docs, "question": query})
        return result
    
    # Define tools
    tools = [
        Tool(
            name="Calculator", 
            func=simple_calculator, 
            description="For math/calculation tasks, input should be a mathematical expression."
        ),
        Tool(
            name="RAG_QA", 
            func=run_rag_pipeline, 
            description="For questions about the Bhagavad Gita. Use this for most questions."
        )
    ]
    
    # Initialize agent with callback handler for logging
    callback_handler = LoggingCallbackHandler()
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        callbacks=[callback_handler]
    )
    
    return agent

# Set up Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    """Process a query and return the response"""
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    logger.info(f"Web query: {query}")
    
    try:
        # Use the agent chain with callbacks to capture intermediate steps
        from langchain.callbacks import get_openai_callback
        from langchain.callbacks.base import BaseCallbackManager
        from langchain.callbacks.tracers import LangChainTracer
        
        # Create a custom callback handler to capture tool usage and references
        class ReferenceCapturingHandler(StdOutCallbackHandler):
            def __init__(self):
                super().__init__()
                self.tool_usage = []
                self.references = []
                self.current_tool = None
            
            def on_tool_start(self, serialized, input_str, **kwargs):
                self.current_tool = {
                    "tool": serialized["name"],
                    "input": input_str
                }
            
            def on_tool_end(self, output, **kwargs):
                if self.current_tool:
                    self.current_tool["output"] = output
                    if self.current_tool["tool"] == "RAG_QA":
                        # Try to extract references from the output
                        if isinstance(output, dict) and "output_text" in output:
                            self.current_tool["answer"] = output["output_text"]
                            # Extract sources if available
                            if "SOURCES" in output.get("output_text", ""):
                                sources_section = output["output_text"].split("SOURCES:", 1)
                                if len(sources_section) > 1:
                                    self.references = [s.strip() for s in sources_section[1].split("\n") if s.strip()]
                        
                    self.tool_usage.append(self.current_tool)
                    self.current_tool = None
        
        # Initialize the callback handler
        reference_handler = ReferenceCapturingHandler()
        
        try:
            # Run the agent with the callback handler
            response = agent.run(query, callbacks=[reference_handler])
            
            # Check if the response indicates the agent couldn't answer
            unable_phrases = [
                "i am unable to answer",
                "i don't know",
                "i cannot answer",
                "i do not have",
                "not available in",
                "not found in the bhagavad gita"
            ]
            
            # If the response says the agent can't answer, use direct Gemini approach
            if any(phrase in response.lower() for phrase in unable_phrases):
                raise ValueError("Agent indicated it cannot answer")
            
            # Otherwise, return the normal response
            references = reference_handler.references
            tools_used = [{"name": tool["tool"], "input": tool["input"]} for tool in reference_handler.tool_usage]
            
            return jsonify({
                'response': response,
                'references': references,
                'tools_used': tools_used
            })
            
        except Exception as agent_error:
            logger.warning(f"Agent could not answer. Falling back to direct Gemini: {agent_error}")
            
            # Get the Gemini LLM instance
            llm = get_gemini_llm()
            
            # Create direct prompt to Gemini with Bhagavad Gita context
            direct_prompt = f"With reference to the Bhagavad Gita, please answer the following question in 60- 70 words: {query}"
            direct_response = llm.invoke(direct_prompt)
            
            return jsonify({
                'response': direct_response.content,
                'references': [],
                'tools_used': [{"name": "Direct Gemini", "input": "Fallback mode - using Gemini directly"}],
                'is_fallback': True
            })
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

# CLI demo
def run_cli_demo():
    """Run the command-line interface demo"""
    logger.info("Starting CLI demo")
    
    print("\n" + "=" * 60)
    print("Welcome to the Bhagavad Gita Q&A System")
    print("Ask questions about the Bhagavad Gita and get answers")
    print("=" * 60 + "\n")
    
    last_response = ""
    
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        # Check if user wants more explanation of the previous answer
        if query.lower() == 'explain more' and last_response:
            logger.info("User requested elaboration on previous response")
            try:
                explain_query = f"Explain more: {last_response}"
                elaborate_response = agent.run(explain_query)
                print("\n" + elaborate_response)
            except Exception as e:
                logger.error(f"Error during elaboration: {e}")
                print(f"\nError while elaborating: {str(e)}")
            continue
        
        logger.info(f"User query: {query}")
        try:
            response = agent.run(query)
            print("\n" + response)
            last_response = response
            print("\nType 'explain more' if you want a more detailed explanation.")
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            print(f"\nError: {str(e)}")

# Create agent for both CLI and web use
agent = setup_rag_agent()

# Create templates directory and ensure it exists
def setup_templates():
    """Set up the templates directory"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)

# Main execution
if __name__ == "__main__":
    logger.info("Bhagavad Gita Q&A Assistant starting up")
    
    # Create templates if they don't exist (for web mode)
    setup_templates()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        logger.info("Starting in web mode")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        run_cli_demo()

