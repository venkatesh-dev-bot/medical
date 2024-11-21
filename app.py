import streamlit as st
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
import logging
import time
import json
from pathlib import Path

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Vector Math Calculator",
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
    st.session_state.custom_prompt = """You are a specialized vector mathematics calculator. Follow these rules strictly:
1. Use only plain text in your responses
2. Write vectors as: 2i + 3j + 4k
3. Show calculations in simple arithmetic form
4. Express all numbers as decimals
5. No LaTeX or special mathematical notation

Question: {input}

Follow this format for your answer:
STEP 1: [First calculation step]
STEP 2: [Second calculation step]
STEP 3: [Third calculation step]
FINAL ANSWER: [Result in simplest form]

Begin solution:
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

# Create chain
gpt4_chain = LLMChain(llm=gpt4, prompt=math_prompt, verbose=True)

# Main content
st.title("🔢 Vector Mathematics Calculator")
st.markdown("---")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
question = st.chat_input("Enter your vector mathematics question...")

if question:
    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Show thinking message
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.text("🤔 Thinking...")
        
        try:
            result = gpt4_chain.invoke({"input": question})
            response_text = result['text']
            
            thinking_placeholder.empty()
            st.write(response_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text
            })
                
        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"❌ Error: {str(e)}"
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
