"""
task3_db_insert.py
- Reads all_reviews_with_sentiment_and_themes.csv
- Creates DB schema and inserts rows
"""
import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float, Text, ForeignKey
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from sqlalchemy.exc import ProgrammingError
from dotenv import load_dotenv

load_dotenv()

USER = os.getenv("PG_USER","postgres")
PASSWORD = os.getenv("PG_PASSWORD","postgres")
HOST = os.getenv("PG_HOST","localhost")
PORT = os.getenv("PG_PORT","5432")
DB = os.getenv("PG_DB","bank_reviews")

# connection URL
url = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"
# Create DB if not exists: SQLAlchemy cannot create DB-level easily across RDBMS. If DB not created, instruct user to run:
# createdb bank_reviews
# or use psql to create DB manually. We'll attempt connection assuming DB exists.
engine = create_engine(url)
conn = engine.connect()
meta = MetaData()

# Define tables
banks = Table(
    'banks', meta,
    Column('bank_id', Integer, primary_key=True, autoincrement=True),
    Column('bank_name', String(255), nullable=False, unique=True),
    Column('app_name', String(255))
)

reviews = Table(
    'reviews', meta,
    Column('review_id', String(255), primary_key=True),
    Column('bank_id', Integer, ForeignKey('banks.bank_id', ondelete='CASCADE')),
    Column('review_text', Text),
    Column('rating', Integer),
    Column('review_date', String(50)),
    Column('sentiment_label', String(20)),
    Column('sentiment_score', Float),
    Column('assigned_theme', String(255)),
    Column('source', String(255))
)

# Create tables
meta.create_all(engine)
print("Tables created (if not existing).")

# Load processed CSV
df = pd.read_csv("all_reviews_with_sentiment_and_themes.csv")

# Map app_name to banks table and upsert banks
unique_apps = df["app_name"].dropna().unique().tolist()
existing_banks = {r[0]: r[1] for r in conn.execute("SELECT bank_id, bank_name FROM banks").fetchall()} if engine.dialect.has_table(conn, "banks") else {}

# Insert banks if needed
for app in unique_apps:
    # Attempt to find by bank_name equal to app (simple). You can change mapping if desired.
    sel = conn.execute(f"SELECT bank_id FROM banks WHERE bank_name = %s", (app,))
    res = sel.fetchone()
    if not res:
        ins = banks.insert().values(bank_name=app, app_name=app)
        conn.execute(ins)

# Build bank_name -> bank_id map
bank_map = {row['bank_name']: row['bank_id'] for row in conn.execute("SELECT bank_id, bank_name FROM banks").mappings()}

# Prepare review inserts (upsert to avoid duplicates)
to_insert = []
for _, r in df.iterrows():
    rid = str(r.get("review_id") or r.get("review_id"))
    app = r.get("app_name")
    bank_id = bank_map.get(app)
    to_insert.append({
        "review_id": rid,
        "bank_id": bank_id,
        "review_text": r.get("clean_review"),
        "rating": int(r.get("rating")) if pd.notna(r.get("rating")) else None,
        "review_date": str(r.get("review_date")),
        "sentiment_label": r.get("sentiment_label"),
        "sentiment_score": float(r.get("sentiment_score")) if pd.notna(r.get("sentiment_score")) else None,
        "assigned_theme": r.get("assigned_theme"),
        "source": r.get("source")
    })

# Bulk upsert with SQLAlchemy core for Postgres
with engine.begin() as conn:
    for row in to_insert:
        stmt = insert(reviews).values(**row)
        stmt = stmt.on_conflict_do_update(
            index_elements=['review_id'],
            set_={
                'review_text': stmt.excluded.review_text,
                'rating': stmt.excluded.rating,
                'review_date': stmt.excluded.review_date,
                'sentiment_label': stmt.excluded.sentiment_label,
                'sentiment_score': stmt.excluded.sentiment_score,
                'assigned_theme': stmt.excluded.assigned_theme,
                'source': stmt.excluded.source
            }
        )
        conn.execute(stmt)

print(f"Inserted/updated {len(to_insert)} reviews into Postgres.")
