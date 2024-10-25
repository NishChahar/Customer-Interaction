import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the SQLite database
db_path = 'interaction_logs.db'
conn = sqlite3.connect(db_path)

# Load data from the 'interactions' table
interactions_df = pd.read_sql("SELECT * FROM interactions", conn)

# Remove the 'accuracy_score' column
interactions_df.drop(columns=['accuracy_score'], inplace=True)

# Count the number of entries in the database
number_of_entries = len(interactions_df)
print(f"Number of entries in the database: {number_of_entries}")

# Grouping data by model name to generate performance metrics (mean sentiment score)
grouped_metrics = interactions_df.groupby('model_name').agg(
    mean_sentiment=('sentiment_score', 'mean'),
    response_time_avg=('response_time_ms', 'mean')
).reset_index()

# Plot comparisons (Sentiment and Response Time)
plt.figure(figsize=(12, 6))
plt.bar(grouped_metrics['model_name'], grouped_metrics['mean_sentiment'], alpha=0.7, label='Mean Sentiment Score')
plt.xlabel('Model Name')
plt.ylabel('Mean Sentiment Score')
plt.title('Model Performance Comparison - Mean Sentiment')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(grouped_metrics['model_name'], grouped_metrics['response_time_avg'], alpha=0.7, label='Average Response Time (ms)', color='orange')
plt.xlabel('Model Name')
plt.ylabel('Average Response Time (ms)')
plt.title('Model Performance Comparison - Response Time')
plt.legend()
plt.show()

# Plotting sentiment scores by model
plt.figure(figsize=(12, 6))
for model in interactions_df['model_name'].unique():
    model_data = interactions_df[interactions_df['model_name'] == model]
    plt.plot(model_data['timestamp'], model_data['sentiment_score'], label=model)

plt.xlabel('Timestamp')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Score by Model Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Adding subjectivity analysis using TextBlob
interactions_df['subjectivity'] = interactions_df['llm_response'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Plotting subjectivity by model
plt.figure(figsize=(12, 6))
for model in interactions_df['model_name'].unique():
    model_data = interactions_df[interactions_df['model_name'] == model]
    plt.plot(model_data['timestamp'], model_data['subjectivity'], label=model)

plt.xlabel('Timestamp')
plt.ylabel('Subjectivity Score')
plt.title('Subjectivity Score by Model Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Plotting average subjectivity by model
average_subjectivity_by_model = interactions_df.groupby('model_name')['subjectivity'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.bar(average_subjectivity_by_model['model_name'], average_subjectivity_by_model['subjectivity'], color='blue', alpha=0.7)
plt.xlabel('Model Name')
plt.ylabel('Average Subjectivity Score')
plt.title('Average Subjectivity Score by Model')
plt.show()

# Calculating complexity of "customer_query"
# Here, complexity is measured as the number of words in the query
interactions_df['query_complexity'] = interactions_df['customer_query'].apply(lambda x: len(x.split()))

# Plotting complexity of customer queries over time
plt.figure(figsize=(12, 6))
plt.plot(interactions_df['timestamp'], interactions_df['query_complexity'], label='Query Complexity', color='green')
plt.xlabel('Timestamp')
plt.ylabel('Query Complexity (Word Count)')
plt.title('Complexity of Customer Queries Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Step 8: Comparing sentiment and subjectivity of "llm_response" against complexity of "customer_query"
plt.figure(figsize=(12, 6))
plt.scatter(interactions_df['query_complexity'], interactions_df['sentiment_score'], alpha=0.5, label='Sentiment Score vs Query Complexity')
plt.scatter(interactions_df['query_complexity'], interactions_df['subjectivity'], alpha=0.5, label='Subjectivity vs Query Complexity', color='red')
plt.xlabel('Query Complexity (Word Count)')
plt.ylabel('Score')
plt.title('Sentiment and Subjectivity vs Query Complexity')
plt.legend()
plt.show()

# Efficacy analysis of each model
# Define efficacy as a combination of high sentiment scores and lower response times
interactions_df['efficacy_score'] = interactions_df['sentiment_score'] / interactions_df['response_time_ms']

# Group by model to calculate the mean efficacy score
efficacy_metrics = interactions_df.groupby('model_name').agg(
    mean_efficacy=('efficacy_score', 'mean')
).reset_index()

# Plotting efficacy by model
plt.figure(figsize=(12, 6))
plt.bar(efficacy_metrics['model_name'], efficacy_metrics['mean_efficacy'], alpha=0.7, label='Mean Efficacy Score', color='purple')
plt.xlabel('Model Name')
plt.ylabel('Mean Efficacy Score')
plt.title('Model Efficacy Comparison')
plt.legend()
plt.show()

# Scatter plot of Sentiment Score for each model
plt.figure(figsize=(12, 6))
for model in interactions_df['model_name'].unique():
    model_data = interactions_df[interactions_df['model_name'] == model]
    plt.scatter(model_data.index, model_data['sentiment_score'], label=model, alpha=0.6)

plt.xlabel('Index')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Score Scatter Plot for Each Model')
plt.legend()
plt.show()

# Scatter plot of Query Complexity for each query
# Calculate query complexity (word count)
interactions_df['query_complexity'] = interactions_df['customer_query'].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 6))
plt.scatter(interactions_df.index, interactions_df['query_complexity'], alpha=0.6, color='green')
plt.xlabel('Index')
plt.ylabel('Query Complexity (Word Count)')
plt.title('Query Complexity Scatter Plot for Each Query')
plt.show()

# Emotional Analysis using VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
interactions_df['emotion_score'] = interactions_df['llm_response'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Plotting Emotion Score by Query Complexity
plt.figure(figsize=(12, 6))
plt.scatter(interactions_df['query_complexity'], interactions_df['emotion_score'], alpha=0.6, color='blue')
plt.xlabel('Query Complexity (Word Count)')
plt.ylabel('Emotion Score (Compound)')
plt.title('Emotion Score vs Query Complexity')
plt.show()

# Calculate average emotion score by model
average_emotion_by_model = interactions_df.groupby('model_name')['emotion_score'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.bar(average_emotion_by_model['model_name'], average_emotion_by_model['emotion_score'], color='orange', alpha=0.7)
plt.xlabel('Model Name')
plt.ylabel('Average Emotion Score')
plt.title('Average Emotion Score by Model')
plt.show()

# Grouping data by model name to generate performance metrics
grouped_metrics = interactions_df.groupby('model_name').agg(
    mean_sentiment=('sentiment_score', 'mean'),
    response_time_avg=('response_time_ms', 'mean'),
    mean_subjectivity=('subjectivity', 'mean'),  # Assuming subjectivity is calculated beforehand
    mean_emotion=('emotion_score', 'mean'),  # Emotion score using VADER
    mean_efficacy=('efficacy_score', 'mean')
).reset_index()

# Printing the results for each model
print("Mean Efficacy, Sentiment, Response Times, Subjectivity, and Emotion Score by Model:")
print(grouped_metrics)

# Close the connection
conn.close()