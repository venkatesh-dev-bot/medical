import json
from pathlib import Path
import streamlit as st
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
import logging
import time
from langchain.callbacks.base import BaseCallbackHandler

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Pediatric Medical Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LLM
gpt4 = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    seed=42
)

# Define storage path for custom prompts
PROMPTS_DIR = Path("stored_prompts")
PROMPTS_FILE = PROMPTS_DIR / "custom_prompts.json"

# Create directory if it doesn't exist
PROMPTS_DIR.mkdir(exist_ok=True)

# Function to load stored prompts
def load_stored_prompts():
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to save prompt
def save_prompt(name, prompt_text):
    prompts = load_stored_prompts()
    prompts[name] = prompt_text
    with open(PROMPTS_FILE, 'w') as f:
        json.dump(prompts, f, indent=2)

# Function to delete prompt
def delete_prompt(name):
    prompts = load_stored_prompts()
    if name in prompts:
        del prompts[name]
        with open(PROMPTS_FILE, 'w') as f:
            json.dump(prompts, f, indent=2)

# Initialize session states
if 'custom_prompt' not in st.session_state:
    st.session_state.custom_prompt = """You are a professional pediatric medical expert. Follow these guidelines strictly:

1. IMPORTANT: Always preface responses with appropriate medical disclaimers
2. Verify symptoms and conditions carefully before responding
3. Use clear, parent-friendly language while maintaining medical accuracy
4. Provide evidence-based information with current medical guidelines
5. Include red flags or warning signs when relevant
6. Recommend when immediate medical attention is needed
7. Explain preventive care and healthy practices
8. NEVER provide specific medication dosages
9. Always encourage consultation with a healthcare provider

Medical Query: {input}

Response Format:
DISCLAIMER: [Appropriate medical disclaimer]
ASSESSMENT: [Initial evaluation of the query]
EXPLANATION: [Detailed medical information]
RECOMMENDATIONS: [General guidance and next steps]
IMPORTANT NOTES: [Key points and warnings]
WHEN TO SEEK IMMEDIATE CARE: [Emergency indicators]

Begin response:
"""

if 'default_prompt' not in st.session_state:
    st.session_state.default_prompt = st.session_state.custom_prompt

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'stored_prompts' not in st.session_state:
    st.session_state.stored_prompts = load_stored_prompts()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Add tabs for prompt management
    tab1, tab2 = st.tabs(["Edit Prompt", "Saved Prompts"])
    
    with tab1:
        st.subheader("Customize Prompt Template")
        
        # Prompt editing area
        custom_prompt = st.text_area(
            "Edit prompt template",
            value=st.session_state.custom_prompt,
            height=300,
            key="prompt_editor"
        )
        
        # Save prompt section
        col1, col2 = st.columns([2, 1])
        with col1:
            prompt_name = st.text_input("Prompt Name", 
                placeholder="Enter a name to save this prompt")
        with col2:
            if st.button("Save Prompt", type="primary"):
                if prompt_name and custom_prompt.strip():
                    save_prompt(prompt_name, custom_prompt)
                    st.session_state.stored_prompts = load_stored_prompts()
                    st.success(f"✅ Prompt '{prompt_name}' saved!")
                else:
                    st.error("Please enter both name and prompt!")
    
    with tab2:
        st.subheader("Saved Prompts")
        
        # Load and display saved prompts
        stored_prompts = load_stored_prompts()
        if stored_prompts:
            selected_prompt = st.selectbox(
                "Select a saved prompt",
                options=list(stored_prompts.keys())
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Prompt"):
                    st.session_state.custom_prompt = stored_prompts[selected_prompt]
                    st.success("Prompt loaded!")
                    st.rerun()
            with col2:
                if st.button("Delete Prompt", type="secondary"):
                    delete_prompt(selected_prompt)
                    st.session_state.stored_prompts = load_stored_prompts()
                    st.warning(f"Prompt '{selected_prompt}' deleted!")
                    st.rerun()
        else:
            st.info("No saved prompts yet!")
    
    st.divider()
    
    # Original prompt controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset to Default"):
            st.session_state.custom_prompt = st.session_state.default_prompt
            st.rerun()
    with col2:
        if st.button("Clear Prompt"):
            st.session_state.custom_prompt = ""
            st.rerun()
    
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Update prompt template with custom prompt
math_prompt = PromptTemplate(
    template=st.session_state.custom_prompt,
    input_variables=["input"]
)

# Add StreamHandler class
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.full_response += token
        self.container.markdown(self.full_response)

# Main content
st.title("🏥 Pediatric Medical Assistant")
st.markdown("""
---
⚕️ **IMPORTANT MEDICAL DISCLAIMER**
- This is an AI assistant providing general medical information
- Not a replacement for professional medical advice
- Always consult with qualified healthcare providers
- Seek immediate medical attention for emergencies
---
""")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
question = st.chat_input("Enter your vector mathematics question...")

# Add medical validation function
def validate_medical_query(query: str) -> tuple[bool, str]:
    """Validate medical queries for safety and appropriateness"""
    # List of keywords indicating emergency situations
    emergency_keywords = [
        "unconscious", "breathing", "choking", "seizure", "bleeding",
        "head injury", "poison", "overdose", "suicide", "abuse"
    ]
    
    # Check for emergency situations
    for keyword in emergency_keywords:
        if keyword in query.lower():
            return False, "⚠️ EMERGENCY DETECTED: Please call emergency services (911) immediately!"
    
    return True, ""

# Update the chat interface with medical validation
if question:
    # Validate the medical query first
    is_safe, warning_message = validate_medical_query(question)
    
    if not is_safe:
        with st.chat_message("assistant"):
            st.error(warning_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": warning_message
            })
    else:
        # Show user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Show thinking message with medical context
        with st.chat_message("assistant"):
            answer_container = st.empty()
            thinking_placeholder = answer_container.text("🩺 Analyzing medical query...")
            stream_handler = StreamHandler(answer_container)
            
            try:
                # Initialize streaming LLM with medical context
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.2,  # Lower temperature for more conservative medical advice
                    streaming=True,
                    callbacks=[stream_handler],
                    verbose=True
                )
                
                # Update chain with streaming LLM
                medical_chain = LLMChain(
                    llm=llm, 
                    prompt=math_prompt, 
                    verbose=True
                )
                
                # Clear thinking message
                thinking_placeholder.empty()
                
                # Get streaming response
                result = medical_chain.invoke({"input": question})
                
                # Store in chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": stream_handler.full_response
                })
                    
            except Exception as e:
                thinking_placeholder.empty()
                error_msg = f"❌ Error: Unable to process medical query. Please try again or consult a healthcare provider."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Add custom CSS
st.markdown("""
<style>
    .stChat {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stChatInput {
        border-radius: 20px;
        border: 2px solid #f0f2f6;
    }
    .stButton button {
        border-radius: 20px;
        padding: 10px 20px;
    }
    .stRadio [role="radiogroup"] {
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .medical-disclaimer {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .emergency-warning {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        font-weight: bold;
    }
    .medical-recommendation {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        line-height: 1.6;
    }
    .medical-reference {
        font-size: 0.8em;
        color: #666;
        margin-top: 10px;
        border-top: 1px solid #eee;
        padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Add MathJax support
components.html("""
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true
            }
        };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
""", height=0) 
