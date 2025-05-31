# AI-Ticket-Classifier
This project is an AI-powered classifier for customer support tickets, built to automatically identify issue types and urgency levels from raw ticket text.

We start by cleaning and preprocessing the data — removing nulls, standardizing text, and extracting key features like length, urgency keywords, sentiment polarity, and product mentions.

A custom text cleaning function lemmatizes tokens and removes noise. Then, we apply feature engineering to extract whether the ticket contains urgent words or a question, and calculate basic stats like word count.

For entity extraction, we parse out products, dates, and complaint keywords using custom logic and regex patterns — making the model more context-aware.

Next, we define two pipelines:

One predicts the issue type using a Random Forest Classifier.

The other predicts urgency level using a Gradient Boosting Classifier with SMOTE to handle class imbalance.

Each pipeline includes a TfidfVectorizer for text, plus standard scaling and one-hot encoding for numerical and categorical features.

After training on labeled data, we evaluate both models using classification reports and confusion matrices, giving us clear insights into performance.

Finally, we deploy the model using Gradio, creating an interactive web app where users can input ticket text and instantly receive predictions, along with confidence scores and extracted entities.
