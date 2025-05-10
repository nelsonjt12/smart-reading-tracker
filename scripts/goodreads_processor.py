import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_goodreads_data(filepath):
    """Load and clean the Goodreads export data"""
    # Load the data
    df = pd.read_csv(filepath)
    
    # Clean the dataframe
    df['ISBN'] = df['ISBN'].str.replace('=""', '').str.replace('""', '')
    df['ISBN13'] = df['ISBN13'].str.replace('=""', '').str.replace('""', '')
    
    # Convert date columns
    df['Date Read'] = pd.to_datetime(df['Date Read'], format='%Y/%m/%d', errors='coerce')
    df['Date Added'] = pd.to_datetime(df['Date Added'], format='%Y/%m/%d', errors='coerce')
    
    # Convert ratings to numeric
    df['My Rating'] = pd.to_numeric(df['My Rating'], errors='coerce')
    df['Average Rating'] = pd.to_numeric(df['Average Rating'], errors='coerce')
    df['Number of Pages'] = pd.to_numeric(df['Number of Pages'], errors='coerce')
    
    # Create read status
    df['Read Status'] = df['Exclusive Shelf']
    
    # Calculate days to read (for read books only)
    df_read = df[df['Read Status'] == 'read'].copy()
    df_read['Days to Read'] = (df_read['Date Read'] - df_read['Date Added']).dt.days
    
    # Merge back
    df = pd.merge(df, df_read[['Book Id', 'Days to Read']], on='Book Id', how='left')
    
    # Create metadata text field for similarity calculations
    df['Metadata'] = df.apply(lambda row: f"{row['Title']} {row['Author']} {row['Publisher'] if not pd.isna(row['Publisher']) else ''} {row['Bookshelves'] if not pd.isna(row['Bookshelves']) else ''}", axis=1)
    
    return df

def analyze_reading_behavior(df):
    """Analyze reading behavior and return insights"""
    results = {}
    
    # Filter read books
    read_books = df[df['Read Status'] == 'read'].copy()
    
    # Reading frequency by year/month
    if not read_books.empty and not read_books['Date Read'].isna().all():
        read_books['Year'] = read_books['Date Read'].dt.year
        read_books['Month'] = read_books['Date Read'].dt.month
        results['books_by_year'] = read_books.groupby('Year').size().reset_index(name='count')
        results['books_by_month'] = read_books.groupby(['Year', 'Month']).size().reset_index(name='count')
    
    # Rating patterns
    if not read_books.empty:
        results['avg_my_rating'] = read_books['My Rating'].mean()
        results['avg_goodreads_rating'] = read_books['Average Rating'].mean()
        results['rating_difference'] = results['avg_my_rating'] - results['avg_goodreads_rating']
    
    # Time to read analysis
    if 'Days to Read' in read_books.columns and not read_books['Days to Read'].isna().all():
        results['avg_days_to_read'] = read_books['Days to Read'].mean()
        results['min_days_to_read'] = read_books['Days to Read'].min()
        results['max_days_to_read'] = read_books['Days to Read'].max()
    
    # Page count analysis
    if not read_books.empty and not read_books['Number of Pages'].isna().all():
        results['avg_pages'] = read_books['Number of Pages'].mean()
        results['min_pages'] = read_books['Number of Pages'].min()
        results['max_pages'] = read_books['Number of Pages'].max()
    
    # TBR waiting time
    tbr_books = df[df['Read Status'] == 'to-read'].copy()
    if not tbr_books.empty and not tbr_books['Date Added'].isna().all():
        tbr_books['Days on TBR'] = (datetime.now().date() - tbr_books['Date Added'].dt.date).dt.days
        results['avg_days_on_tbr'] = tbr_books['Days on TBR'].mean()
        results['books_on_tbr'] = len(tbr_books)
    
    return results, read_books, tbr_books

def analyze_authors_genres(df):
    """Analyze author and genre patterns"""
    results = {}
    
    # Read books only
    read_books = df[df['Read Status'] == 'read'].copy()
    
    # Top authors
    if not read_books.empty:
        results['top_authors'] = read_books['Author'].value_counts().head(10).reset_index()
        results['top_authors'].columns = ['Author', 'Count']
    
    # Author diversity
    if not read_books.empty:
        results['author_diversity'] = len(read_books['Author'].unique())
        results['author_diversity_ratio'] = results['author_diversity'] / len(read_books)
    
    # Genre/bookshelf analysis
    if not read_books.empty and not read_books['Bookshelves'].isna().all():
        # Expand bookshelves into rows
        bookshelves = read_books['Bookshelves'].dropna().str.split(', ').explode()
        results['top_shelves'] = bookshelves.value_counts().head(10).reset_index()
        results['top_shelves'].columns = ['Shelf', 'Count']
    
    return results

def generate_recommendations(df, n_recommendations=5, similarity_method='metadata'):
    """Generate book recommendations based on reading history"""
    # Get read and to-read books
    read_books = df[df['Read Status'] == 'read'].copy()
    tbr_books = df[df['Read Status'] == 'to-read'].copy()
    
    if read_books.empty or tbr_books.empty:
        return pd.DataFrame()
    
    # Focus on highly rated books for similarity
    if 'My Rating' in read_books.columns and not read_books['My Rating'].isna().all():
        reference_books = read_books[read_books['My Rating'] >= 4].copy()
        if reference_books.empty:  # If no high ratings, use all read books
            reference_books = read_books
    else:
        reference_books = read_books
    
    # Method 1: TF-IDF on metadata
    if similarity_method == 'metadata':
        # Compute TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        all_books = pd.concat([reference_books, tbr_books])
        tfidf_matrix = tfidf.fit_transform(all_books['Metadata'])
        
        # Calculate similarity between read books and tbr books
        ref_idx = list(range(len(reference_books)))
        tbr_idx = list(range(len(reference_books), len(all_books)))
        
        # Compute similarity scores
        similarity_scores = cosine_similarity(tfidf_matrix[tbr_idx], tfidf_matrix[ref_idx])
        
        # Average similarity to all reference books
        avg_similarity = np.mean(similarity_scores, axis=1)
        
        # Add similarity score to tbr_books
        tbr_books_scored = tbr_books.copy()
        tbr_books_scored['similarity_score'] = avg_similarity
        
        # Sort by similarity
        recommendations = tbr_books_scored.sort_values('similarity_score', ascending=False).head(n_recommendations)
        return recommendations
    
    # Method 2: KNN on features
    else:
        # Create features for KNN
        features = ['Average Rating', 'Number of Pages']
        valid_features = [f for f in features if not df[f].isna().all()]
        
        if not valid_features:
            return pd.DataFrame()
            
        # Scale features
        scaler = MinMaxScaler()
        all_books = pd.concat([reference_books, tbr_books])
        scaled_features = scaler.fit_transform(all_books[valid_features])
        
        # Fit KNN model
        knn = NearestNeighbors(n_neighbors=min(5, len(scaled_features)), algorithm='auto')
        knn.fit(scaled_features[:len(reference_books)])
        
        # Get recommendations
        tbr_features = scaled_features[len(reference_books):]
        recommendations = []
        
        for i, book_features in enumerate(tbr_features):
            distances, indices = knn.kneighbors([book_features])
            sim_score = 1 / (1 + np.mean(distances))
            tbr_books.iloc[i, tbr_books.columns.get_indexer(['similarity_score'])] = sim_score
            recommendations.append((i, sim_score))
        
        # Sort and get top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_indices = [rec[0] for rec in recommendations[:n_recommendations]]
        return tbr_books.iloc[top_indices].sort_values('similarity_score', ascending=False)
