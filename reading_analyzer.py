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

# Function to load and clean Goodreads data
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
    
    # Calculate days to read (for read books only)
    df_read = df[df['Read Status'] == 'read'].copy()
    if not df_read.empty and not df_read['Date Read'].isna().all() and not df_read['Date Added'].isna().all():
        df_read['Days to Read'] = (df_read['Date Read'] - df_read['Date Added']).dt.days
        # Merge back
        df = pd.merge(df, df_read[['Book Id', 'Days to Read']], on='Book Id', how='left')
    
    print(f"Data loaded successfully. Total records: {len(df)}")
    print(f"Read books: {len(df[df['Read Status'] == 'read'])}")
    print(f"To-read books: {len(df[df['Read Status'] == 'to-read'])}")
    
    return df

# Function to analyze reading behavior
def analyze_reading_behavior(df):
    """Analyze reading behavior and return insights"""
    print("Analyzing reading behavior...")
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
    if not read_books.empty and not read_books['My Rating'].isna().all():
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
        # Calculate days on TBR without using .dt on the result
        current_date = pd.Timestamp(datetime.now().date())
        tbr_books['Days on TBR'] = tbr_books['Date Added'].apply(lambda x: (current_date - x).days if pd.notnull(x) else None)
        results['avg_days_on_tbr'] = tbr_books['Days on TBR'].mean()
        results['books_on_tbr'] = len(tbr_books)
    
    return results, read_books, tbr_books

# Function to analyze authors and genres
def analyze_authors_genres(df):
    """Analyze author and genre patterns"""
    print("Analyzing author and genre patterns...")
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
    
    # Genre/bookshelf analysis - careful handling of bookshelves
    if not read_books.empty and 'Bookshelves' in read_books.columns:
        bookshelves = read_books['Bookshelves'].dropna()
        if not bookshelves.empty:
            shelf_list = []
            for shelves in bookshelves:
                if isinstance(shelves, str) and shelves:
                    shelf_list.extend([s.strip() for s in shelves.split(',') if s.strip()])
            
            if shelf_list:
                shelf_counts = pd.Series(shelf_list).value_counts().head(10).reset_index()
                shelf_counts.columns = ['Shelf', 'Count']
                results['top_shelves'] = shelf_counts
    
    return results

# Function to generate recommendations
def generate_recommendations(df, n_recommendations=10):
    """Generate book recommendations for TBR list"""
    print("Generating reading recommendations...")
    
    # Get read and to-read books
    read_books = df[df['Read Status'] == 'read'].copy()
    tbr_books = df[df['Read Status'] == 'to-read'].copy()
    
    if read_books.empty or tbr_books.empty:
        print("Not enough data to generate recommendations.")
        return pd.DataFrame()
    
    # Sort TBR books by Goodreads average rating
    recommended_books = tbr_books.sort_values('Average Rating', ascending=False).head(n_recommendations)
    
    return recommended_books[['Title', 'Author', 'Average Rating', 'Number of Pages']]

# Visualization functions
def plot_reading_frequency(results, save_path='output/images/reading_frequency.png'):
    """Plot reading frequency by year"""
    if 'books_by_year' in results and not results['books_by_year'].empty:
        data = results['books_by_year']
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Year', y='count', data=data)
        plt.title('Books Read by Year', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Books', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved reading frequency chart to {save_path}")
        return save_path
    return None

def plot_rating_comparison(results, save_path='output/images/rating_comparison.png'):
    """Plot comparison between user ratings and Goodreads average ratings"""
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
        
        plt.title('Your Ratings vs. Goodreads Average', fontsize=16)
        plt.ylabel('Rating (out of 5)', fontsize=14)
        plt.ylim(0, 5.5)  # Set y-axis limit to 5.5 to make room for text
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved rating comparison chart to {save_path}")
        return save_path
    return None

def plot_page_count_distribution(read_books, save_path='output/images/page_distribution.png'):
    """Plot distribution of book lengths"""
    if read_books is not None and 'Number of Pages' in read_books.columns and not read_books['Number of Pages'].isna().all():
        plt.figure(figsize=(10, 6))
        sns.histplot(read_books['Number of Pages'].dropna(), bins=20)
        plt.title('Distribution of Book Lengths', fontsize=16)
        plt.xlabel('Number of Pages', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        
        # Add vertical line for average
        avg_pages = read_books['Number of Pages'].mean()
        plt.axvline(x=avg_pages, color='r', linestyle='--', label=f'Average: {avg_pages:.0f} pages')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved page count distribution chart to {save_path}")
        return save_path
    return None

def plot_top_authors(author_results, save_path='output/images/top_authors.png'):
    """Plot top authors by book count"""
    if 'top_authors' in author_results and not author_results['top_authors'].empty:
        data = author_results['top_authors'].sort_values('Count')
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(data['Author'], data['Count'])
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.0f}', va='center')
        
        plt.title('Top Authors in Your Library', fontsize=16)
        plt.xlabel('Number of Books', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved top authors chart to {save_path}")
        return save_path
    return None

# Generate a simple text report for analysis findings
def generate_text_report(results, read_books, tbr_books, author_results, recommendations):
    """Generate a text report with key findings"""
    print("Generating text report...")
    
    report = """# Smart Reading Tracker Report

## Reading Behavior Analysis

"""
    
    # Basic stats
    total_read = len(read_books) if read_books is not None else 0
    tbr_count = len(tbr_books) if tbr_books is not None else 0
    
    report += f"Total books read: {total_read}\n"
    report += f"Books on to-read list: {tbr_count}\n\n"
    
    # Reading patterns
    if 'avg_my_rating' in results:
        report += f"Your average rating: {results['avg_my_rating']:.2f} stars\n"
    if 'avg_goodreads_rating' in results:
        report += f"Average Goodreads rating: {results['avg_goodreads_rating']:.2f} stars\n"
    if 'rating_difference' in results:
        if results['rating_difference'] > 0:
            report += f"You rate books {abs(results['rating_difference']):.2f} stars higher than the Goodreads average.\n"
        elif results['rating_difference'] < 0:
            report += f"You rate books {abs(results['rating_difference']):.2f} stars lower than the Goodreads average.\n"
        else:
            report += "Your ratings align perfectly with the Goodreads average.\n"
    
    # Page counts
    if 'avg_pages' in results:
        report += f"\nAverage book length: {results['avg_pages']:.0f} pages\n"
    if 'min_pages' in results and 'max_pages' in results:
        report += f"Shortest book: {results['min_pages']:.0f} pages\n"
        report += f"Longest book: {results['max_pages']:.0f} pages\n"
    
    # Reading time
    if 'avg_days_to_read' in results:
        report += f"\nAverage time to read a book: {results['avg_days_to_read']:.1f} days\n"
    if 'avg_days_on_tbr' in results:
        report += f"Average time books spend on your to-read list: {results['avg_days_on_tbr']:.1f} days\n"
    
    # Author analysis
    report += "\n## Author Analysis\n\n"
    
    if 'author_diversity' in author_results:
        report += f"You've read books from {author_results['author_diversity']} different authors.\n"
    if 'author_diversity_ratio' in author_results:
        diversity_ratio = author_results['author_diversity_ratio'] * 100
        report += f"Author diversity ratio: {diversity_ratio:.1f}%\n"
    
    # Top authors
    if 'top_authors' in author_results and not author_results['top_authors'].empty:
        report += "\nYour most-read authors:\n"
        for _, row in author_results['top_authors'].head(5).iterrows():
            report += f"- {row['Author']}: {row['Count']} books\n"
    
    # Book recommendations
    report += "\n## Book Recommendations\n\n"
    report += "Based on your reading history, here are the top books from your to-read list to consider next:\n\n"
    
    if recommendations is not None and not recommendations.empty:
        for i, (_, row) in enumerate(recommendations.head(10).iterrows()):
            pages = int(row['Number of Pages']) if not pd.isna(row['Number of Pages']) else "Unknown"
            report += f"{i+1}. \"{row['Title']}\" by {row['Author']} (Rating: {row['Average Rating']:.2f}, Pages: {pages})\n"
    else:
        report += "Not enough data to generate recommendations.\n"
    
    # Closing
    report += "\n## Analysis Summary\n\n"
    report += "This report was generated by analyzing your Goodreads export data without requiring personal notes or reviews.\n"
    report += "Visualizations of your reading patterns are available in the output/images directory.\n"
    
    # Write to file
    report_path = 'output/reading_analysis.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Text report generated successfully: {report_path}")
    return report_path

# Main function
def main():
    print("===== Smart Reading Tracker =====\n")
    
    # Load data
    file_path = "data/goodreads_library_export.csv"
    df = load_goodreads_data(file_path)
    
    # Analyze reading behavior
    results, read_books, tbr_books = analyze_reading_behavior(df)
    
    # Analyze authors and genres
    author_results = analyze_authors_genres(df)
    
    # Generate recommendations
    recommendations = generate_recommendations(df, n_recommendations=10)
    
    # Generate visualizations
    plot_reading_frequency(results)
    plot_rating_comparison(results)
    plot_page_count_distribution(read_books)
    plot_top_authors(author_results)
    
    # Generate text report
    report_path = generate_text_report(results, read_books, tbr_books, author_results, recommendations)
    
    print("\n===== Analysis Complete =====")
    print(f"Report available at: {report_path}")
    print("Visualizations available in: output/images/")

if __name__ == "__main__":
    main()
