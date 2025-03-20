# AI Voice Assistant Loan Application
This project is an AI-powered voice assistant designed to help users with loan applications. It integrates voice recognition, text-to-speech, email notifications, and database operations with a conversational AI agent. The assistant leverages multiple libraries including Streamlit for the web UI, LangGraph & LangChain for conversational flow, and SQLAlchemy for database interactions.
## Features
1.Voice and Text Interaction: Users can interact using voice or text.
   
2.Speech Recognition & Text-to-Speech: Convert spoken input into text and generate audio responses.

3.Conversational AI Agent: Uses a generative AI model to handle conversation and process loan application details.

4.Database Integration: Stores loan application details in a PostgreSQL database.

5.Email Confirmation: Sends confirmation emails upon successful loan application submission.


# 1.Installation
``` Clone the repository:
git clone https://github.com/yourusername/ai-voice-assistant.git
cd ai-voice-assistant
```

# 2.Set up a virtual environment (optional but recommended):

```
python3 -m venv venv
source venv/bin/activate
```

# 3.Install the required dependencies:
	pip install -r requirements.txt

## Environment Variables
Create a .env file in the root directory with the following variables (adjust as needed):
# API Keys
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```
# Database Credentials
```
REAL_DB_USER=your_db_username
REAL_DB_PASSWORD=your_db_password
REAL_DB_HOST=your_db_host
REAL_DB_NAME=your_db_name
REAL_DB_PORT=your_db_port
```

# Email Credentials
```
SENDER_EMAIL=your_email@example.com
SENDER_PASSCODE=your_email_passcode
```

To launch the Streamlit web application, run:
```
streamlit run your_script_name.py
```
Replace your_script_name.py with the name of your Python file containing the above code.


