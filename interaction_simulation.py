from openai import OpenAI
import time
import json
import pandas as pd
from textblob import TextBlob
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.metrics import precision_score, recall_score, f1_score
from flask import Flask, render_template
from datetime import datetime



# Set OpenAI API Key
client = OpenAI(
    api_key ="api-key"
)

# Initialize SQLAlchemy
Base = declarative_base()
engine = create_engine('sqlite:///interaction_logs.db')
Session = sessionmaker(bind=engine)
session = Session()

# Interaction model for database
class Interaction(Base):
    __tablename__ = 'interactions'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=time.time)
    customer_query = Column(String)
    llm_response = Column(String)
    response_time_ms = Column(Float)
    model_name = Column(String)
    sentiment_score = Column(Float)

Base.metadata.create_all(engine)

def customer_is_satisfied(message):
    """
    Check if the customer is satisfied by looking for specific keywords in their message.
    
    :param message: The customer's message.
    :return: True if the customer says thanks or indicates satisfaction, False otherwise.
    """
    satisfied_phrases = ["exit", "that's all", "bye"]
    return any(phrase in message.lower() for phrase in satisfied_phrases)

def generate_customer_response(history, model):
    # Use ChatGPT to generate a customer's response
    customer_prompt = "What would a customer say in response to this? Just say the answer and do not include 'a customer might say'"
    history.append({"role": "user", "content": customer_prompt})
    
    response = client.chat.completions.create(
        model=model, 
        messages=history
    )
    
    customer_message = response.choices[0].message.content
    history.append({"role": "assistant", "content": customer_message})
    
    return customer_message

def chat_with_gpt(prompt, history, model):
    # Add the new prompt to the conversation history
    history.append({"role": "user", "content": prompt})

    # Call the ChatGPT API with the current conversation history
    response = client.chat.completions.create(
        model=model,  
        messages=history
    )

    # Extract the response from the assistant
    message = response.choices[0].message.content
    
    # Add the assistant's response to the history
    history.append({"role": "assistant", "content": message})

    return message

# Function to simulate interactions
def simulate_interaction(history, model):

    cus_res = []
    llm_res = []

    while True:
        
        # Generate the customer's response
        customer_response = generate_customer_response( history, model)
        print(f"Customer: {customer_response}")
        cus_res.append(customer_response)

        # Get the assistant's response
        assistant_response = chat_with_gpt(customer_response, history, model)
        print(f"Assistant: {assistant_response}")
        llm_res.append(assistant_response)

        if customer_is_satisfied(customer_response):
                print("Conversation ended: Customer is satisfied.")
                history = []
                break
        
    return llm_res, cus_res

# Function to calculate sentiment of response
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # -1 (negative) to 1 (positive)

# Function to log interaction in the database
def log_interaction(customer_query, llm_response, response_time, model_name, sentiment_score):
    interaction = Interaction(
        timestamp=datetime.now(),  # Correctly pass the current datetime
        customer_query=json.dumps(customer_query),
        llm_response=json.dumps(llm_response),
        response_time_ms=response_time * 1000,
        model_name=model_name,
        sentiment_score=sentiment_score,
    )
    session.add(interaction)
    session.commit()


# Running interactions and logging performance
def run_simulation():
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4-turbo"]  # Use different models
    
    for model in models:
        print("\n\n\n\n",model,"\n\n\n\n\n" )

        history = [
                {"role": "system", 
                "content": (
                    "You are a customer service chatbot for a website called ert.com. "
                    "Your job is to assist customers with their queries regarding online retail, "
                    "including product details, order status, returns, and any other support requests. "
                    "Always be polite and professional."
                )},
                {"role":"user",
                "content": ("You are a customer that is having issue with your order that you placed on ert.com"
                    "It can be, that you could have been charged twice,"
                    "that you never recieved the product,"
                    "a damaged product,"
                    "or any other issues that happen with the order"
                    "act as if you are the customer, don't say 'a customer might say this' "
                    "Once satisfied the customer must say 'exit' or use 'that's all' to exit the chat"
                )}
            ]
        start_time = time.time()
        response, query = simulate_interaction(history, model) 
        end_time = time.time()
        response_time = end_time - start_time
        string_response=' '.join(map(str, response))  
        # Sentiment analysis
        sentiment_score = get_sentiment(string_response)


        # Log interaction in the database
        log_interaction(query, response, response_time, model, sentiment_score)

    

# Reporting and metrics (advanced)
def generate_report():
    df = pd.read_sql("SELECT * FROM interactions", con=engine)

    # Calculate average response time
    avg_response_time = df['response_time_ms'].mean()

    # Metrics for each model
    model_group = df.groupby('model_name').agg(
        avg_response_time=('response_time_ms', 'mean'),
        avg_sentiment_score=('sentiment_score', 'mean')
    )
    
    print("\nAverage Response Time (ms):", avg_response_time)
    print("\nModel Performance:")
    print(model_group)

# Flask app for reporting
app = Flask(__name__)

@app.route('/')
def dashboard():
    df = pd.read_sql("SELECT * FROM interactions", con=engine)
    avg_response_time = df['response_time_ms'].mean()
    model_group = df.groupby('model_name').agg(
        avg_response_time=('response_time_ms', 'mean'),
        avg_sentiment_score=('sentiment_score', 'mean')
    )
    return render_template('dashboard.html', avg_response_time=avg_response_time, model_group=model_group)

if __name__ == '__main__':
    # Uncomment below to run the simulation and log data
    run = 500
    for i in range(run):
        #run_simulation()

    # Run Flask app
    #app.run(debug=True)

    # Generate a text-based report
    #generate_report()
