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
    if not df_read.empty:
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
        tbr_books['Days on TBR'] = (datetime.now().date() - tbr_books['Date Added'].dt.date).dt.days
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
    
    # Genre/bookshelf analysis
    if not read_books.empty and 'Bookshelves' in read_books.columns and not read_books['Bookshelves'].isna().all():
        # Expand bookshelves into rows
        bookshelves = read_books['Bookshelves'].dropna().str.split(', ').explode()
        if not bookshelves.empty:
            results['top_shelves'] = bookshelves.value_counts().head(10).reset_index()
            results['top_shelves'].columns = ['Shelf', 'Count']
    
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
    
    # Method 1: Based on Goodreads average rating
    recommended_by_rating = tbr_books.sort_values('Average Rating', ascending=False).head(n_recommendations)
    
    # Method 2: Based on similarity to highly rated books (if ratings available)
    recommended_by_pages = pd.DataFrame()
    if not read_books['My Rating'].isna().all():
        # Get average page count of highly rated books
        highly_rated = read_books[read_books['My Rating'] >= 4]
        if not highly_rated.empty and not highly_rated['Number of Pages'].isna().all():
            avg_pages_preferred = highly_rated['Number of Pages'].mean()
            
            # Find books on TBR with similar page count
            tbr_books['page_diff'] = abs(tbr_books['Number of Pages'] - avg_pages_preferred)
            recommended_by_pages = tbr_books.sort_values('page_diff').head(n_recommendations)
    
    # Final recommendations (combine methods if both available)
    if not recommended_by_pages.empty:
        # Combine results by ranking
        all_recommendations = pd.concat([
            recommended_by_rating.assign(rating_rank=range(len(recommended_by_rating))),
            recommended_by_pages.assign(page_rank=range(len(recommended_by_pages)))
        ])
        
        # Group by Book Id and sum ranks (lower is better)
        ranked = all_recommendations.groupby('Book Id').apply(
            lambda x: pd.Series({
                'combined_rank': x['rating_rank'].min() if 'rating_rank' in x else 999 
                               + x['page_rank'].min() if 'page_rank' in x else 999,
                'Title': x['Title'].iloc[0],
                'Author': x['Author'].iloc[0],
                'Average Rating': x['Average Rating'].iloc[0],
                'Number of Pages': x['Number of Pages'].iloc[0]
            })
        ).sort_values('combined_rank').head(n_recommendations)
        
        return ranked[['Title', 'Author', 'Average Rating', 'Number of Pages']]
    else:
        return recommended_by_rating[['Title', 'Author', 'Average Rating', 'Number of Pages']]

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

# Function to generate a report
def generate_report(results, read_books, tbr_books, author_results, recommendations):
    """Generate an HTML report with insights and recommendations"""
    print("Generating report...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Reading Tracker Report</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                margin: 0; 
                padding: 20px; 
                color: #333; 
                max-width: 1200px; 
                margin: 0 auto; 
            }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .section {{ 
                margin-bottom: 30px; 
                border-bottom: 1px solid #eee; 
                padding-bottom: 20px; 
            }}
            .metrics {{ 
                display: flex; 
                justify-content: space-between; 
                flex-wrap: wrap; 
                margin-bottom: 20px; 
            }}
            .metric {{ 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 5px; 
                width: 30%; 
                margin-bottom: 15px; 
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            }}
            .metric h3 {{ margin-top: 0; color: #7b8a8b; }}
            .metric p {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0; 
            }}
            th, td {{ 
                padding: 12px 15px; 
                border-bottom: 1px solid #ddd; 
                text-align: left; 
            }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .image-container {{ margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
            .date {{ color: #7b8a8b; font-style: italic; }}
            .recommendations {{ 
                background: #f0f7ff; 
                padding: 20px; 
                border-radius: 5px; 
                margin-top: 30px; 
            }}
        </style>
    </head>
    <body>
        <h1>Smart Reading Tracker Report</h1>
        <p class="date">Generated on {datetime.now().strftime('%B %d, %Y')}</p>
        
        <div class="section">
            <h2>Reading Overview</h2>
            
            <div class="metrics">
    """
    
    # Add metrics
    total_read = len(read_books) if read_books is not None else 0
    tbr_count = len(tbr_books) if tbr_books is not None else 0
    
    html_content += f"""
                <div class="metric">
                    <h3>Books Read</h3>
                    <p>{total_read}</p>
                </div>
    """
    
    if 'avg_my_rating' in results:
        html_content += f"""
                <div class="metric">
                    <h3>Average Rating</h3>
                    <p>{results['avg_my_rating']:.2f} ★</p>
                </div>
        """
    
    html_content += f"""
                <div class="metric">
                    <h3>Books To Read</h3>
                    <p>{tbr_count}</p>
                </div>
    """
    
    if 'avg_pages' in results:
        html_content += f"""
                <div class="metric">
                    <h3>Average Book Length</h3>
                    <p>{results['avg_pages']:.0f} pages</p>
                </div>
        """
    
    if 'author_diversity' in author_results:
        html_content += f"""
                <div class="metric">
                    <h3>Unique Authors</h3>
                    <p>{author_results['author_diversity']}</p>
                </div>
        """
    
    if 'avg_days_to_read' in results:
        html_content += f"""
                <div class="metric">
                    <h3>Avg. Time to Read</h3>
                    <p>{results['avg_days_to_read']:.1f} days</p>
                </div>
        """
    
    html_content += """
            </div><!-- end metrics -->
    """
    
    # Add reading frequency chart if available
    reading_freq_path = plot_reading_frequency(results)
    if reading_freq_path:
        html_content += f"""
            <h3>Reading Frequency</h3>
            <div class="image-container">
                <img src="{reading_freq_path}" alt="Reading Frequency Chart">
            </div>
        """
    
    html_content += """
        </div><!-- end section -->
        
        <div class="section">
            <h2>Reading Patterns</h2>
    """
    
    # Add rating comparison chart if available
    rating_chart_path = plot_rating_comparison(results)
    if rating_chart_path:
        html_content += f"""
            <h3>Your Ratings vs. Goodreads Average</h3>
            <div class="image-container">
                <img src="{rating_chart_path}" alt="Rating Comparison Chart">
            </div>
        """
    
    # Add page distribution chart if available
    page_dist_path = plot_page_count_distribution(read_books)
    if page_dist_path:
        html_content += f"""
            <h3>Book Length Distribution</h3>
            <div class="image-container">
                <img src="{page_dist_path}" alt="Page Count Distribution Chart">
            </div>
        """
    
    html_content += """
        </div><!-- end section -->
        
        <div class="section">
            <h2>Author Analysis</h2>
    """
    
    # Add top authors chart if available
    authors_chart_path = plot_top_authors(author_results)
    if authors_chart_path:
        html_content += f"""
            <h3>Most Read Authors</h3>
            <div class="image-container">
                <img src="{authors_chart_path}" alt="Top Authors Chart">
            </div>
        """
    
    # Add author diversity stats if available
    if 'author_diversity' in author_results and 'author_diversity_ratio' in author_results:
        diversity_ratio = author_results['author_diversity_ratio'] * 100
        html_content += f"""
            <h3>Author Diversity</h3>
            <p>You've read books from {author_results['author_diversity']} different authors, 
            representing {diversity_ratio:.1f}% of your total library.</p>
        """
    
    html_content += """
        </div><!-- end section -->
    """
    
    # Add recommendations section if available
    if recommendations is not None and not recommendations.empty:
        html_content += """
        <div class="section recommendations">
            <h2>TBR Recommendations</h2>
            <p>Based on your reading history, these books from your To-Read list are recommended as your next reads:</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Title</th>
                        <th>Author</th>
                        <th>Avg Rating</th>
                        <th>Pages</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for _, row in recommendations.iterrows():
            html_content += f"""
                    <tr>
                        <td>{row['Title']}</td>
                        <td>{row['Author']}</td>
                        <td>{row['Average Rating']:.2f}</td>
                        <td>{int(row['Number of Pages']) if not pd.isna(row['Number of Pages']) else '-'}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
            
            <h3>How These Recommendations Were Generated</h3>
            <p>These recommendations are based on analyzing your reading patterns including:</p>
            <ul>
                <li>Books with high Goodreads ratings on your to-read list</li>
                <li>Books with similar page counts to your highly-rated reads</li>
                <li>Books that match your typical reading preferences</li>
            </ul>
        </div><!-- end recommendations section -->
        """
    
    # Close HTML
    html_content += """
        <div class="section">
            <h2>About This Report</h2>
            <p>This report was generated by the Smart Reading Tracker, analyzing your Goodreads library export data.</p>
            <p>The analysis is based on metadata from your reading history and does not require personal notes or reviews.</p>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    report_path = 'output/reading_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report generated successfully: {report_path}")
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
    
    # Generate report
    report_path = generate_report(results, read_books, tbr_books, author_results, recommendations)
    
    print("\n===== Analysis Complete =====")
    print(f"Report available at: {report_path}")
    print("Open this HTML file in your browser to view your personalized reading insights.")

if __name__ == "__main__":
    main()
