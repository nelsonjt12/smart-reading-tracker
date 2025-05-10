# smart-reading-tracker
A Python-based tracker that analyzes your reading list to provide sentiment analysis, topic clustering, and personalized reading recommendations.
## Features
- Sentiment analysis on notes/highlights
- Topic clustering of your reads
- Visualization of emotional trends
- Bonus: Recommend future reads based on previous themes

## Setup
1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Add your reading data to `data/reading_log.csv` or link your Google Sheet.
3. Run the notebook or scripts in `notebooks/` to generate insights.

## Data Format
```csv
Title,Author,Source,Date Read,Notes,Genre,Link
"Atomic Habits","James Clear","Book","2023-03-12","Loved the habit loop concept.","Self-help","https://..."
```

## Folders
- `data/`: your input CSV or Google Sheet connector
- `scripts/`: Python modules for preprocessing, sentiment analysis, clustering, and recommendations
- `notebooks/`: exploratory analysis and visualization

