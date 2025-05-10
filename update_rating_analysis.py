import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime

# Set up directories for output
os.makedirs('output', exist_ok=True)
os.makedirs('output/images', exist_ok=True)

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_goodreads_data(filepath):
    """Load and clean the Goodreads export data"""
    print(f"Loading data from {filepath}...")
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
    
    return df

def analyze_ratings(df):
    """Analyze rating patterns with proper handling of null values"""
    print("Analyzing rating patterns...")
    results = {}
    
    # Filter read books
    read_books = df[df['Read Status'] == 'read'].copy()
    
    # Count total read books
    total_read_books = len(read_books)
    
    # Count books with valid ratings
    rated_books = read_books[read_books['My Rating'] > 0].copy()
    total_rated_books = len(rated_books)
    
    # Calculate percentage of rated books
    rating_percentage = (total_rated_books / total_read_books) * 100 if total_read_books > 0 else 0
    
    print(f"Total read books: {total_read_books}")
    print(f"Books with ratings: {total_rated_books} ({rating_percentage:.1f}%)")
    
    # Calculate average ratings ONLY for books that have ratings
    if not rated_books.empty:
        # Your average rating (only for books you've rated)
        results['avg_my_rating'] = rated_books['My Rating'].mean()
        
        # Goodreads average for the SAME books you've rated
        results['avg_goodreads_rating_for_rated'] = rated_books['Average Rating'].mean()
        
        # Goodreads average for ALL books you've read
        results['avg_goodreads_rating_all'] = read_books['Average Rating'].mean()
        
        # Rating differences
        results['rating_difference'] = results['avg_my_rating'] - results['avg_goodreads_rating_for_rated']
    
    return results, read_books, rated_books

def plot_updated_rating_comparison(results, save_path='output/images/rating_comparison_updated.png'):
    """Plot improved comparison between user ratings and Goodreads average ratings"""
    if 'avg_my_rating' in results and 'avg_goodreads_rating_for_rated' in results:
        # Create figure with more space
        plt.figure(figsize=(12, 8))
        
        # Data for bars
        labels = [
            'Your Average\n(rated books only)', 
            'Goodreads Average\n(same books)', 
            'Goodreads Average\n(all books)'
        ]
        values = [
            results['avg_my_rating'], 
            results['avg_goodreads_rating_for_rated'],
            results['avg_goodreads_rating_all']
        ]
        
        # Create bars with different colors
        bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Add rating values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=12)
        
        plt.title('Your Ratings vs. Goodreads Average (Null Values Excluded)', fontsize=16)
        plt.ylabel('Rating (out of 5)', fontsize=14)
        plt.ylim(0, 5.5)  # Set y-axis limit to 5.5 to make room for text
        
        # Add explanatory note
        plt.figtext(0.5, 0.01, 
                  "Note: 'Your Average' only includes books you've explicitly rated. Null/zero ratings are excluded.", 
                  ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(pad=3)  # Add padding for the text
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved updated rating comparison chart to {save_path}")
        return save_path
    return None

# Main function
def main():
    print("===== Rating Analysis Update =====\n")
    
    # Load data
    file_path = "data/goodreads_library_export.csv"
    df = load_goodreads_data(file_path)
    
    # Analyze ratings with proper null handling
    results, read_books, rated_books = analyze_ratings(df)
    
    # Generate updated visualization
    plot_updated_rating_comparison(results)
    
    # Generate a small summary report
    summary = "\n===== Rating Analysis Summary =====\n\n"
    
    if 'avg_my_rating' in results:
        summary += f"Books you've read: {len(read_books)}\n"
        summary += f"Books you've rated: {len(rated_books)} ({(len(rated_books)/len(read_books)*100):.1f}% of read books)\n\n"
        
        summary += f"Your average rating (excluding null ratings): {results['avg_my_rating']:.2f} stars\n"
        summary += f"Goodreads average (for the same books): {results['avg_goodreads_rating_for_rated']:.2f} stars\n"
        summary += f"Goodreads average (for all your read books): {results['avg_goodreads_rating_all']:.2f} stars\n\n"
        
        if results['rating_difference'] > 0:
            summary += f"You rate books {abs(results['rating_difference']):.2f} stars HIGHER than the Goodreads average.\n"
        elif results['rating_difference'] < 0:
            summary += f"You rate books {abs(results['rating_difference']):.2f} stars LOWER than the Goodreads average.\n"
        else:
            summary += "Your ratings align perfectly with the Goodreads average.\n"
    
    print(summary)
    
    # Save summary
    with open('output/rating_analysis_summary.txt', 'w') as f:
        f.write(summary)
    
    print("Updated rating comparison chart saved to output/images/rating_comparison_updated.png")
    print("Summary saved to output/rating_analysis_summary.txt")

if __name__ == "__main__":
    main()
