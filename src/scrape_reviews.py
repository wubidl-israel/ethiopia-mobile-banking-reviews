# src/scrape_reviews.py
from google_play_scraper import reviews, Sort
import csv, time
from tqdm import tqdm

TARGET_PER_BANK = 450
SLEEP_BETWEEN_PAGES = 1.2
OUTPUT_DIR = "data/raw/"

apps = {
    "cbe": {"app_id": "com.combanketh.mobilebanking", "bank": "CBE"},
    "boa": {"app_id": "com.boa.boaMobileBanking", "bank": "BOA"},
    "dashen": {"app_id": "com.dashen.dashensuperapp", "bank": "Dashen"}
}

def fetch_reviews_for_app(app_id, bank_label, target=TARGET_PER_BANK):
    all_reviews = []
    continuation_token = None
    pbar = tqdm(total=target, desc=f"Fetching {bank_label}")
    while len(all_reviews) < target:
        n = min(200, target - len(all_reviews))
        try:
            result, continuation_token = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=n,
                continuation_token=continuation_token
            )
        except Exception as e:
            print("Error:", e)
            time.sleep(3)
            continue

        if not result:
            print("No more results.")
            break

        for r in result:
            text = r.get('content') or r.get('review') or ""
            rating = r.get('score') or r.get('rating') or None
            date = r.get('at') or r.get('date') or None
            all_reviews.append({"review": text, "rating": rating, "date_raw": date, "bank": bank_label, "source": "google_play"})

        pbar.update(len(result))
        if continuation_token is None:
            break
        time.sleep(SLEEP_BETWEEN_PAGES)
    pbar.close()
    return all_reviews

def save_csv(rows, out_path):
    keys = ["review","rating","date_raw","bank","source"]
    with open(out_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    for slug, meta in apps.items():
        rows = fetch_reviews_for_app(meta['app_id'], meta['bank'], TARGET_PER_BANK)
        out = f"{OUTPUT_DIR}{slug}_raw.csv"
        save_csv(rows, out)
        print(f"Saved {len(rows)} rows -> {out}")
