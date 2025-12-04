import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

INPUT = "all_reviews_with_sentiment_and_themes.csv"
df = pd.read_csv(INPUT)
df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
sns.set(style="whitegrid")

# 1) Rating distribution per bank
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='rating', hue='app_name', order=sorted(df['rating'].dropna().unique()))
plt.title("Rating distribution by bank")
plt.savefig("plot_rating_distribution.png", dpi=300)
plt.close()

# 2) Sentiment distribution per bank
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='sentiment_label', hue='app_name', order=['positive','neutral','negative'])
plt.title("Sentiment labels by bank")
plt.savefig("plot_sentiment_labels.png", dpi=300)
plt.close()

# 3) Sentiment trend over time (monthly avg)
df['month'] = df['review_date'].dt.to_period('M')
trend = df.groupby(['app_name','month']).sentiment_score.mean().reset_index()
trend['month'] = trend['month'].dt.to_timestamp()
plt.figure(figsize=(12,6))
for app, g in trend.groupby('app_name'):
    plt.plot(g['month'], g['sentiment_score'], marker='o', label=app)
plt.legend()
plt.title("Monthly average sentiment score by bank")
plt.xlabel("Month")
plt.ylabel("Avg Sentiment Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_sentiment_trend.png", dpi=300)
plt.close()

# 4) Wordcloud of negative reviews per bank (example for each bank)
for app, g in df.groupby('app_name'):
    neg_text = " ".join(g[g['sentiment_label']=='negative']['clean_review'].dropna().astype(str).tolist())[:200000]
    if len(neg_text) < 10:
        continue
    wc = WordCloud(width=800, height=400, collocations=False).generate(neg_text)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Negative review wordcloud: {app}")
    plt.savefig(f"wordcloud_negative_{app.replace(' ','_')}.png", dpi=300)
    plt.close()
