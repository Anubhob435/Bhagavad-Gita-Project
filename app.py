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
        response = agent.run(query)
        return jsonify({'response': response})
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

# Create templates directory and index.html file
def setup_templates():
    """Set up the templates directory and HTML files"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bhagavad Gita Q&A</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            header {
                background-color: #f8b400;
                color: #fff;
                padding: 1rem;
                text-align: center;
                border-radius: 5px 5px 0 0;
            }
            .chat-container {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
                margin-top: 20px;
            }
            #chat-box {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
            }
            .user-message {
                background-color: #e6f7ff;
                padding: 8px 12px;
                margin: 5px 0;
                border-radius: 15px 15px 0 15px;
                max-width: 80%;
                align-self: flex-end;
                margin-left: auto;
            }            .bot-message {
                background-color: #f0f0f0;
                padding: 8px 12px;
                margin: 5px 0;
                border-radius: 15px 15px 15px 0;
                max-width: 80%;
            }
            .message-container {
                display: flex;
                flex-direction: column;
                margin-bottom: 10px;
            }
            .user-container {
                align-items: flex-end;
            }
            .explain-more-btn {
                font-size: 12px;
                color: #1a73e8;
                background: none;
                border: none;
                text-decoration: underline;
                cursor: pointer;
                align-self: flex-start;
                margin-top: 4px;
                margin-left: 12px;
                padding: 0;
            }
            .explain-more-btn:hover {
                color: #0d47a1;
            }
            .form-control {
                display: flex;
            }
            #user-input {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px 0 0 5px;
                font-size: 16px;
            }
            #send-btn {
                padding: 10px 20px;
                background-color: #f8b400;
                color: white;
                border: none;
                border-radius: 0 5px 5px 0;
                cursor: pointer;
                font-size: 16px;
            }
            #send-btn:hover {
                background-color: #e6a700;
            }
            .loading {
                text-align: center;
                margin: 10px 0;
                color: #666;
            }
            footer {
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                color: #666;
                font-size: 12px;
            }
            .source {
                font-size: 12px;
                color: #666;
                margin-top: 5px;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Bhagavad Gita Q&A System</h1>
                <p>Ask questions about the Bhagavad Gita and get AI-powered responses</p>
            </header>
            
            <div class="chat-container">
                <div id="chat-box">
                    <div class="message-container">
                        <div class="bot-message">
                            Welcome! I'm your Bhagavad Gita assistant. Ask me any question about the sacred text!
                        </div>
                    </div>
                </div>
                
                <div id="loading" class="loading" style="display: none;">
                    Thinking...
                </div>
                
                <div class="form-control">
                    <input type="text" id="user-input" placeholder="Ask a question..." autofocus>
                    <button id="send-btn">Send</button>
                </div>
            </div>
            
            <footer>
                <p>Bhagavad Gita Q&A System powered by Gemini AI &copy; 2025</p>
            </footer>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const chatBox = document.getElementById('chat-box');
                const userInput = document.getElementById('user-input');
                const sendBtn = document.getElementById('send-btn');
                const loading = document.getElementById('loading');
                  // Function to add a message to the chat
                function addMessage(message, isUser = false) {
                    const messageContainer = document.createElement('div');
                    messageContainer.className = 'message-container';
                    if (isUser) messageContainer.className += ' user-container';
                    
                    const messageDiv = document.createElement('div');
                    messageDiv.className = isUser ? 'user-message' : 'bot-message';
                    messageDiv.textContent = message;
                    
                    messageContainer.appendChild(messageDiv);
                    
                    // Add "Explain more" button for bot messages
                    if (!isUser) {
                        const explainBtn = document.createElement('button');
                        explainBtn.className = 'explain-more-btn';
                        explainBtn.textContent = 'Explain more';
                        explainBtn.onclick = function() {
                            const explainQuery = `Explain more: ${message}`;
                            sendQuery(explainQuery);
                        };
                        messageContainer.appendChild(explainBtn);
                    }
                    
                    chatBox.appendChild(messageContainer);
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
                
                // Function to send query to backend
                async function sendQuery(query) {
                    loading.style.display = 'block';
                    
                    try {
                        const response = await fetch('/query', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ query: query }),
                        });
                        
                        const data = await response.json();
                        loading.style.display = 'none';
                        
                        if (response.ok) {
                            addMessage(data.response);
                        } else {
                            addMessage('Sorry, I encountered an error: ' + (data.error || 'Unknown error'));
                        }
                    } catch (error) {
                        loading.style.display = 'none';
                        addMessage('Sorry, an error occurred while processing your request.');
                        console.error('Error:', error);
                    }
                }
                
                // Send button click event
                sendBtn.addEventListener('click', function() {
                    const query = userInput.value.trim();
                    if (query) {
                        addMessage(query, true);
                        userInput.value = '';
                        sendQuery(query);
                    }
                });
                
                // Enter key press event
                userInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        const query = userInput.value.trim();
                        if (query) {
                            addMessage(query, true);
                            userInput.value = '';
                            sendQuery(query);
                        }
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    
    index_path = os.path.join(templates_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)

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

