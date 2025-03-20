import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import tempfile
import os
import base64
from gtts import gTTS
from dotenv import load_dotenv
import google.generativeai as genai
import re  # Needed for clean_tool_call
from langchain_openai import ChatOpenAI
# === LangGraph & LangChain Imports ===
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine, text
import psycopg2
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
import random
from Prompt import agent_prompt, tool_prompt

# --- Load environment variables ---
load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure the ChatGroq model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
# --- Retrieve database credentials from environment variables ---
REAL_DB_USER = os.environ['REAL_DB_USER']
REAL_DB_PASSWORD = os.environ['REAL_DB_PASSWORD']
REAL_DB_HOST = os.environ['REAL_DB_HOST']
REAL_DB_NAME = os.environ['REAL_DB_NAME']
REAL_DB_PORT = os.environ['REAL_DB_PORT']

# --- Create engine for PostgreSQL connection ---
real_engine = create_engine(
    f"postgresql+psycopg2://{REAL_DB_USER}@{REAL_DB_HOST}:{REAL_DB_PORT}/{REAL_DB_NAME}"
)

Base = declarative_base()

# --- Define Table for Loan Applications ---
class LoanApplication(Base):
    __tablename__ = 'loan_applications'
    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String)
    last_name = Column(String)
    phone=Column(String)
    email=Column(String)
    monthly_income=Column(String)
    preferred_loan_tenure=Column(String)
    loan_reason = Column(String)
    loan_amount = Column(String)  # Adjust if you use Numeric in DB

# --- Function to Store Conversation Details in Database ---
def store_conversation_details(first_name:str, last_name:str, phone:str,email:str,monthly_income:str,preferred_loan_tenure:str,loan_reason:str, loan_amount:str):
    insert_query = """
    INSERT INTO loan_applications 
    (first_name, last_name, phone, email, monthly_income, preferred_loan_tenure, loan_reason, loan_amount)
    VALUES 
    (:first_name, :last_name, :phone, :email, :monthly_income, :preferred_loan_tenure, :loan_reason, :loan_amount)
    """
    with real_engine.connect() as connection:
        result=connection.execute(text(insert_query), {
            "first_name": first_name,
            "last_name": last_name,
            "phone": phone,
            "email": email,
            "monthly_income": monthly_income,
            "preferred_loan_tenure": preferred_loan_tenure,
            "loan_reason": loan_reason,
            "loan_amount": loan_amount
        })
        print("Loan Details----------------", result)
        connection.commit()

# === Define the Agent State for LangGraph ===
class AgentState(BaseModel):
    user_input: Optional[str] = None
    messages: List[Any] = Field(default_factory=list)
    loan_details: Optional[dict] = None

# === Define a function to clean internal tool call wrappers and chain-of-thought ---
def clean_tool_call(response_text: str) -> str:
    """
    Extracts the final text from a tool call wrapper if present.
    This function removes any <tool_call>...</|tool▁calls▁end|> wrappers and any <think>...</think> sections.
    """
    print("clean tool call -------------------------------")
    # Remove the tool call wrapper if it exists
    cleaned_text = re.sub(r"<tool_call>.*?<\|tool▁calls▁end\|>", "", response_text, flags=re.DOTALL)
    # Remove any <think> sections
    cleaned_text = re.sub(r"<think>.*?</think>", "", cleaned_text, flags=re.DOTALL)
    return cleaned_text.strip()


def send_loan_confirmation(receiver_email: str) -> str:
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSCODE")
    print("send_loan_confirmation -------------------------------")
    
    if not sender_email or not sender_password:
        return "Error: Sender credentials not set."
    
    context = ssl.create_default_context()
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Loan Application Submitted Successfully"
    
    body = (
        "Dear Customer,\n\n"
        "Your loan application has been submitted successfully. Thank you for choosing our service.\n\n"
        "Best regards,\nLoan Application Team"
    )
    message.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        return "Confirmation email sent successfully!"
    except Exception as e:
        return f"Error sending email: {str(e)}"

# === Define Tools ===
@tool
def normalize_email(recognized_email: str) -> str:
    """
    Normalizes a voice-recognized email address.
    """
    email = recognized_email.lower()
    # Replace variations of "at" with "@"
    email = re.sub(r'\s*at\s*(the\s*rate\s*)?', '@', email)
    # Replace variations of "dot" or "period" with a period.
    # Using a word boundary to catch cases like "dotkom" (which becomes ".kom")
    email = re.sub(r'\b(dot|period)', '.', email)
    # Remove any remaining spaces.
    email = email.replace(' ', '')
    # Correct misrecognized ".kom" to ".com"
    email = email.replace('.kom', '.com')
    print("email -------------------------------", email)
    return email

@tool
def normalize_phone(recognized_phone: str) -> str:
    """
    Normalizes a voice-recognized phone number.
    
    Example:
      Input: "70442 60709"
      Output: "7044260709"
    
    Raises a ValueError if the normalized number is not exactly 10 digits.
    """
    # Remove any non-digit characters.
    phone = re.sub(r'\D', '', recognized_phone)
    print("Normalized phone:", phone)
    
    # Validate that the phone number has exactly 10 digits.
    if len(phone) != 10:
        raise ValueError("InvalidPhoneNumber")  # Raise an error if not valid.
    
    return phone
@tool
def loan_application_tool(text: str):
    """
    This tool handles the conversational part of the loan application process.
    It ensures a structured and smooth interaction while strictly following the instructions.
    """
    system_prompt=tool_prompt
    # Combine system prompt and user text
    combined_text = system_prompt + "\nUser: " + text
    result = model.invoke(combined_text).content  # Generate response using the model
    return {"messages": [{"role": "assistant", "content": result}]}


@tool
def database_insertion_tool(loan_details: dict):
    """
    Inserts the collected loan details into the database.
    """
    print("database_insertion_tool -------------------------------",loan_details)
    if not loan_details:
        return {"messages": [{"role": "assistant", "content": "Error: Missing loan details."}]}
    
    try:
        store_conversation_details(
            first_name=loan_details.get("first_name", ""),
            last_name=loan_details.get("last_name", ""),
            phone=loan_details.get("phone", ""),
            email=loan_details.get("email", ""),
            monthly_income=loan_details.get("monthly_income", ""),
            preferred_loan_tenure=loan_details.get("preferred_loan_tenure", ""),
            loan_reason=loan_details.get("loan_reason", ""),
            loan_amount=loan_details.get("loan_amount", "")
        )
        
        # After successful insertion, send a confirmation email if an email address exists
        receiver_email = loan_details.get("email", "")
        print("receiver_email -------------------------------", receiver_email)
        if receiver_email:
            email_status = send_loan_confirmation(receiver_email)
        else:
            email_status = "No email provided; confirmation email not sent."
        
        confirmation_msg = f"Loan details created successfully. {email_status}"
        return {"messages": [{"role": "assistant", "content": confirmation_msg}]}
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"Database Error: {str(e)}"}]}

loan_application_agent = create_react_agent(
    model,
    tools=[loan_application_tool, database_insertion_tool,normalize_email,normalize_phone],
    name="loan_application_agent",
     prompt=agent_prompt
)


# === Define Graph Node ===
def loan_application_node(state: AgentState):
    # Invoke the unified agent with the conversation.
    response = loan_application_agent.invoke({"messages": state.messages + [{"role": "user", "content": state.user_input}]})
    print("loan_application_node -------------------------------",response)
    # Update state's messages with the agent's response.
    if isinstance(response, dict) and "messages" in response:
        state.messages = response["messages"]
    else:
        state.messages = [{"role": "assistant", "content": str(response)}]
    state.loan_details = response
    # Return the full state as a dictionary.
    return state.dict()

# === Create and Compile the LangGraph ===
graph = StateGraph(AgentState)
graph.add_node("loan_application_node", loan_application_node)
graph.set_entry_point("loan_application_node")
graph.add_edge("loan_application_node", END)
memory=MemorySaver()
compiled_graph = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
# === Streamlit UI Code ===

st.set_page_config(page_title="AI Voice Assistant", layout="wide", page_icon="🤖")

dark_mode_css = """
<style>
body { background-color: #2E2E2E; color: white; }
.chat-message { border-radius: 15px; padding: 10px; margin: 5px; }
.user-message { background-color: #5DADE2; color: white; }
.assistant-message { background-color: #58D68D; color: white; }
.recording { animation: pulse 1s infinite; }
@keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
</style>
"""
light_mode_css = """
<style>
body { background-color: white; color: black; }
</style>
"""

with st.sidebar:
    st.markdown("## Appearance")
    dark_mode = st.checkbox("Dark Mode", value=False)
    if dark_mode:
        st.markdown(dark_mode_css, unsafe_allow_html=True)
    else:
        st.markdown(light_mode_css, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False
if "pending_clear" not in st.session_state:
    st.session_state.pending_clear = False

with st.sidebar:
    st.markdown('<h1 style="font-size: 1.5rem;">Settings</h1>', unsafe_allow_html=True)
    
    mode_options = {"⌨️ Text Mode": False, "🎤 Voice Mode": True}
    selected_mode = st.selectbox(
        label="Select input mode",
        options=list(mode_options.keys()),
        index=1 if st.session_state.voice_mode else 0,
        label_visibility="collapsed"
    )
    st.session_state.voice_mode = mode_options[selected_mode]
    
    with st.expander("Voice Settings", expanded=True):
        voice_option = st.selectbox(
            "Assistant Voice",
            options=["Default (US)", "British Accent", "Australian Accent", "Indian Accent"],
            index=0
        )
        voice_map = {
            "Default (US)": {"lang": "en", "tld": "com"},
            "British Accent": {"lang": "en", "tld": "co.uk"},
            "Australian Accent": {"lang": "en", "tld": "com.au"},
            "Indian Accent": {"lang": "en", "tld": "co.in"}
        }
        st.session_state.current_voice = voice_map[voice_option]
    
    st.markdown("### Chat History")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.pending_clear = True
    if st.session_state.pending_clear:
        st.warning("Are you sure you want to clear the chat history?")
        if st.button("Confirm Clear"):
            st.session_state["messages"] = []
            st.session_state.pending_clear = False
            st.success("Chat history cleared.")
        if st.button("Cancel Clear"):
            st.session_state.pending_clear = False

    st.markdown("---")
    st.markdown("### About AI Voice Assistant")
    st.write("This assistant uses state-of-the-art AI to generate responses. Interact via voice or text.")
    st.markdown("### Features")
    st.markdown("- 🎤 Voice recognition")
    st.markdown("- 🔊 Text-to-speech responses")
    st.markdown("- 💬 Natural language understanding")

st.markdown("<h1 style='text-align: center;'>🤖 AI Voice Assistant</h1>", unsafe_allow_html=True)
status_placeholder = st.empty()

def text_to_speech(text):
    # Provide a fallback if text is empty.
    if not text.strip():
        text = "No response received."
    current_voice = st.session_state.current_voice
    tts = gTTS(text=text, lang=current_voice.get("lang", "en"), tld=current_voice.get("tld", "com"), slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def autoplay_audio(audio_filepath):
    with open(audio_filepath, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay controls style="width:100%; display:none;">
      <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """
    st.components.v1.html(audio_html, height=60)

def process_user_input(user_text):
    # Append the new user message to the conversation history.
    conversation = st.session_state["messages"].copy()
    conversation.append({"role": "user", "content": user_text})
    
    # Create an initial agent state with the updated conversation.
    state = AgentState(user_input=user_text, messages=conversation)
    
    new_state = compiled_graph.invoke({"messages": conversation},config)
    print("New state:", new_state)
    
    response_text = ""
    if new_state.get("loan_details") and "messages" in new_state.get("loan_details"):
        messages = new_state["loan_details"].get("messages", [])
        if messages:
            response_text = messages[-1].get("content", "")
    
    # Clean the response text to remove internal chain-of-thought.
    response_text = clean_tool_call(response_text)
    
    # Fallback: if no response is generated, provide a default message.
    if not response_text.strip():
        if user_text.lower() in ["hi", "hello", "hey"]:
            response_text = "Hello! How can I assist you with your loan application? Please reply 'yes' if you need help."
        else:
            response_text = "I'm sorry, could you please provide more details?"
    
    st.session_state["messages"].append({"role": "user", "content": user_text})
    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    
    audio_file = text_to_speech(response_text)
    return audio_file, response_text

user_text = ""
if st.session_state.voice_mode:
    st.subheader("🎤 Speak Now")
    status_placeholder.info("Status: Listening...")
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#6366f1",
        icon_name="microphone",
        icon_size="2x",
    )
    if audio_bytes is not None:
        status_placeholder.info("Status: Processing voice input...")
        with st.spinner("Processing voice input..."):
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio_file.write(audio_bytes)
            temp_audio_file.close()
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(temp_audio_file.name) as source:
                    audio_data = recognizer.record(source)
                user_text = recognizer.recognize_google(audio_data)
                st.success(f"**You said:** {user_text}")
            except Exception:
                st.error("Could not recognize the audio. Please try again.")
                user_text = ""
            os.remove(temp_audio_file.name)
        status_placeholder.empty()
else:
    if user_text == "":
        user_text = st.chat_input("Type a message or switch to Voice Mode")

if user_text:
    with st.spinner("Generating response..."):
        audio_file, response_text = process_user_input(user_text)
        autoplay_audio(audio_file)
        os.remove(audio_file)

from streamlit_chat import message
for idx, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{idx}_{hash(msg['content'])}",avatar_style="no-avatar")
    else:
        message(msg["content"], is_user=False, key=f"assistant_{idx}_{hash(msg['content'])}",avatar_style="no-avatar")