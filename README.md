## Smart Reading Tracker Project Plan (Metadata-Driven)

## Project Goal

Build a Smart Reading Tracker using Goodreads export data (CSV) to analyze reading patterns, extract insights from metadata, and generate reading recommendations—all without needing personal notes or reviews.

### 1. Reading Behavior Insights

- Reading frequency: Books read per month/year; visualize streaks and seasonal patterns
- Rating patterns: Your average rating vs. Goodreads averages; trends over time
- Time-to-read: Time between "date added" and "date read"; average wait time on TBR
- Page count analysis: Avg. book length; longest/shortest reads; changes over time

### 2. Author & Genre Exploration

- Top authors: Most read, highest rated, author diversity
- Genre/shelf trends: Most common genres/moods based on bookshelf tags

### 3. Clustering and Recommendations

- Topic clustering: Use NLP on titles, tags, or fetched descriptions (TF-IDF + KMeans)
- Book recommender: Based on similarities in metadata to your highly-rated books (KNN or cosine similarity)

### Optional Enhancements

- Fetch book descriptions/genres: Use Google Books, Open Library, or Goodreads API
- TBR optimizer: Prioritize unread books most aligned with past 5-star reads

## Setup
1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Add your reading data to `data/reading_log.csv` or link your Google Sheet.
3. Run the notebook or scripts in `notebooks/` to generate insights.

## Data Format
```csv
Book Id,Title,Author,Author l-f,Additional Authors,ISBN,ISBN13,My Rating,Average Rating,Publisher,Binding,Number of Pages,Year Published,Original Publication Year,Date Read,Date Added,Bookshelves,Bookshelves with positions,Exclusive Shelf,My Review,Spoiler,Private Notes,Read Count,Owned Copies
200982339,Under the Same Stars,Libba Bray,"Bray, Libba",,"=""0374388946""","=""9780374388942""",0,4.17,"Farrar, Straus and Giroux (BYR)",Hardcover,480,2025,2025,2025/03/29,2024/03/21,,,read,,,,1,0
205181098,The Thirteenth Child,Erin A. Craig,"Craig, Erin A.",,"=""0593482581""","=""9780593482582""",0,4.08,Delacorte Press,Hardcover,512,2024,2024,2025/03/03,2025/02/24,,,read,,,,1,0

## Folders
- `data/`: your input CSV or Google Sheet connector
- `scripts/`: Python modules for preprocessing, sentiment analysis, clustering, and recommendations
- `notebooks/`: exploratory analysis and visualization

