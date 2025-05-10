import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

def set_style():
    """Set the visualization style for consistent appearance"""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_reading_frequency(results, period='year'):
    """Plot reading frequency by year or month"""
    set_style()
    
    if period == 'year' and 'books_by_year' in results:
        data = results['books_by_year']
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Year', y='count', data=data)
        plt.title('Books Read by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Books')
        return plt.gcf()
    
    elif period == 'month' and 'books_by_month' in results:
        # Convert month numbers to names
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        data = results['books_by_month'].copy()
        data['Month_Name'] = data['Month'].map(month_names)
        
        # Group by year and plot with different colors for each year
        plt.figure(figsize=(14, 7))
        for year, group in data.groupby('Year'):
            sns.lineplot(x='Month_Name', y='count', data=group, marker='o', label=str(year))
        
        plt.title('Books Read by Month Over Years')
        plt.xlabel('Month')
        plt.ylabel('Number of Books')
        plt.legend(title='Year')
        return plt.gcf()

def plot_rating_comparison(results):
    """Plot comparison between user ratings and Goodreads average ratings"""
    set_style()
    
    if 'avg_my_rating' in results and 'avg_goodreads_rating' in results:
        labels = ['Your Average', 'Goodreads Average']
        values = [results['avg_my_rating'], results['avg_goodreads_rating']]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e'])
        
        # Add rating values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Your Ratings vs. Goodreads Average')
        plt.ylabel('Rating (out of 5)')
        plt.ylim(0, 5.5)  # Set y-axis limit to 5.5 to make room for text
        
        return plt.gcf()

def plot_page_count_distribution(read_books):
    """Plot distribution of book lengths"""
    set_style()
    
    if 'Number of Pages' in read_books.columns and not read_books['Number of Pages'].isna().all():
        plt.figure(figsize=(10, 6))
        sns.histplot(read_books['Number of Pages'].dropna(), bins=20, kde=True)
        plt.title('Distribution of Book Lengths')
        plt.xlabel('Number of Pages')
        plt.ylabel('Count')
        
        # Add vertical line for average
        avg_pages = read_books['Number of Pages'].mean()
        plt.axvline(x=avg_pages, color='r', linestyle='--', label=f'Average: {avg_pages:.0f} pages')
        plt.legend()
        
        return plt.gcf()

def plot_top_authors(author_results):
    """Plot top authors by book count"""
    set_style()
    
    if 'top_authors' in author_results and not author_results['top_authors'].empty:
        data = author_results['top_authors'].sort_values('Count')
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(data['Author'], data['Count'], color=sns.color_palette('viridis', len(data)))
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.0f}', va='center')
        
        plt.title('Top Authors in Your Library')
        plt.xlabel('Number of Books')
        plt.tight_layout()
        
        return plt.gcf()

def plot_genre_distribution(author_results):
    """Plot genre/shelf distribution"""
    set_style()
    
    if 'top_shelves' in author_results and not author_results['top_shelves'].empty:
        data = author_results['top_shelves'].sort_values('Count', ascending=False)
        
        plt.figure(figsize=(12, 8))
        plt.pie(data['Count'], labels=data['Shelf'], autopct='%1.1f%%', 
               startangle=90, shadow=False, 
               colors=sns.color_palette('viridis', len(data)))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Genre Distribution')
        plt.tight_layout()
        
        return plt.gcf()

def plot_reading_time(read_books):
    """Plot time taken to read books"""
    set_style()
    
    if 'Days to Read' in read_books.columns and not read_books['Days to Read'].isna().all():
        # Filter out extreme outliers
        q1 = read_books['Days to Read'].quantile(0.25)
        q3 = read_books['Days to Read'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        filtered_data = read_books[read_books['Days to Read'] <= upper_bound]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=filtered_data['Days to Read'])
        plt.title('Time to Read Books (Days)')
        plt.ylabel('Days')
        
        return plt.gcf()

def plot_tbr_recommendations(recommendations):
    """Plot recommended books from TBR list"""
    set_style()
    
    if not recommendations.empty and 'similarity_score' in recommendations.columns:
        # Sort by similarity score
        data = recommendations.sort_values('similarity_score', ascending=True).copy()
        
        # Truncate long titles
        data['Short_Title'] = data['Title'].apply(lambda x: (x[:40] + '...') if len(x) > 40 else x)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(data['Short_Title'], data['similarity_score'], 
                        color=sns.color_palette('viridis', len(data)))
        
        # Add score labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', va='center')
        
        plt.title('Recommended Books from Your TBR List')
        plt.xlabel('Similarity Score')
        plt.xlim(0, 1.1)  # Set x-axis limit to 1.1 to make room for text
        plt.tight_layout()
        
        return plt.gcf()
