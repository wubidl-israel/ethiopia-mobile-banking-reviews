# Ethiopia Mobile Banking Reviews
This project contains a full pipeline for:

Task 1: Data Cleaning & Preprocessing
Task 2: Sentiment Analysis + Thematic Analysis

DistilBERT

VADER fallback

TF-IDF keyword extraction

KMeans clustering

Theme assignment per bank

Task 3: PostgreSQL Storage

Banks table

Reviews table

Upsert logic

SQLAlchemy schema

Task 4: Visualization

Rating distribution

Sentiment distribution

Sentiment trends

Word clouds

Files
File	Purpose
task2_sentiment_theme.py	Main sentiment + themes pipeline
task3_db_insert.py	Database schema + insert
task4_visualize.py	Plots and wordclouds
requirements.txt	Dependencies
themes_per_bank.json	Output from clustering
all_reviews_with_sentiment_and_themes.csv	Final processed dataset
