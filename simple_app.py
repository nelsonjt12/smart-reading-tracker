import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set styling for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Page configuration
st.set_page_config(
    page_title="Smart Reading Tracker",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Smart Reading Tracker 📚")
st.markdown(
    """This app analyzes your Goodreads reading data to provide insights and recommendations. 
    Upload your Goodreads export CSV to get started!"""
)

# Helper functions for data processing and visualization
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
    if not df_read.empty:
        df_read['Days to Read'] = (df_read['Date Read'] - df_read['Date Added']).dt.days
        # Merge back
        df = pd.merge(df, df_read[['Book Id', 'Days to Read']], on='Book Id', how='left')
    
    return df

def analyze_reading_behavior(df):
    """Analyze reading behavior and return insights"""
    results = {}
    
    # Filter read books
    read_books = df[df['Read Status'] == 'read'].copy()
    
    # Reading frequency by year
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
    if not read_books.empty and 'Bookshelves' in read_books.columns and not read_books['Bookshelves'].isna().all():
        # Expand bookshelves into rows
        bookshelves = read_books['Bookshelves'].dropna().str.split(', ').explode()
        results['top_shelves'] = bookshelves.value_counts().head(10).reset_index()
        results['top_shelves'].columns = ['Shelf', 'Count']
    
    return results

def generate_simple_recommendations(df, n_recommendations=5):
    """Generate simple book recommendations based on average ratings"""
    # Get read and to-read books
    read_books = df[df['Read Status'] == 'read'].copy()
    tbr_books = df[df['Read Status'] == 'to-read'].copy()
    
    if read_books.empty or tbr_books.empty:
        return pd.DataFrame()
    
    # Sort TBR books by Goodreads average rating
    recommended_books = tbr_books.sort_values('Average Rating', ascending=False).head(n_recommendations)
    
    return recommended_books

# Visualization functions
def plot_reading_frequency(results, period='year'):
    """Plot reading frequency by year or month"""
    if period == 'year' and 'books_by_year' in results:
        data = results['books_by_year']
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Year', y='count', data=data)
        plt.title('Books Read by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Books')
        return plt.gcf()

def plot_rating_comparison(results):
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
        
        plt.title('Your Ratings vs. Goodreads Average')
        plt.ylabel('Rating (out of 5)')
        plt.ylim(0, 5.5)  # Set y-axis limit to 5.5 to make room for text
        
        return plt.gcf()

def plot_page_count_distribution(read_books):
    """Plot distribution of book lengths"""
    if 'Number of Pages' in read_books.columns and not read_books['Number of Pages'].isna().all():
        plt.figure(figsize=(10, 6))
        sns.histplot(read_books['Number of Pages'].dropna(), bins=20)
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
    if 'top_authors' in author_results and not author_results['top_authors'].empty:
        data = author_results['top_authors'].sort_values('Count')
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(data['Author'], data['Count'])
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.0f}', va='center')
        
        plt.title('Top Authors in Your Library')
        plt.xlabel('Number of Books')
        plt.tight_layout()
        
        return plt.gcf()

# Sidebar for data loading
st.sidebar.header("Data Source")

# Input options
data_source = st.sidebar.radio(
    "Select data source:",
    ["Use sample data", "Upload Goodreads export"]
)

# Initialize session state for data persistence
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.read_books = None
    st.session_state.tbr_books = None
    st.session_state.results = None
    st.session_state.author_results = None
    st.session_state.recommendations = None

# Function to load and process data
def process_data(file_path):
    # Load and preprocess the data
    df = load_goodreads_data(file_path)
    
    # Analyze reading behavior
    results, read_books, tbr_books = analyze_reading_behavior(df)
    
    # Analyze authors and genres
    author_results = analyze_authors_genres(df)
    
    # Generate recommendations
    recommendations = generate_simple_recommendations(df, n_recommendations=10)
    
    # Store in session state
    st.session_state.df = df
    st.session_state.read_books = read_books
    st.session_state.tbr_books = tbr_books
    st.session_state.results = results
    st.session_state.author_results = author_results
    st.session_state.recommendations = recommendations
    st.session_state.data_loaded = True
    
    return df, results, read_books, tbr_books, author_results, recommendations

# Handle data source selection
file_path = None

if data_source == "Use sample data":
    file_path = "data/goodreads_library_export.csv"
    if os.path.exists(file_path):
        with st.spinner('Loading and analyzing sample data...'):
            df, results, read_books, tbr_books, author_results, recommendations = process_data(file_path)
        st.success('Sample data loaded!')
    else:
        st.error("Sample data file not found. Please upload your own data.")

elif data_source == "Upload Goodreads export":
    uploaded_file = st.sidebar.file_uploader("Upload your Goodreads export CSV", type="csv")
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with st.spinner('Loading and analyzing your data...'):
            temp_path = os.path.join("data", "temp_upload.csv")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the data
            df, results, read_books, tbr_books, author_results, recommendations = process_data(temp_path)
        st.success('Your data has been loaded and analyzed!')

# Main app content (only show if data is loaded)
if st.session_state.data_loaded:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Reading Overview",
        "Reading Patterns",
        "Author Analysis",
        "TBR Recommendations"
    ])
    
    # Tab 1: Reading Overview
    with tab1:
        st.header("Reading Overview")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Books read
            total_read = len(st.session_state.read_books) if st.session_state.read_books is not None else 0
            st.metric("Books Read", total_read)
            
            # Average rating if available
            if 'avg_my_rating' in st.session_state.results:
                avg_rating = st.session_state.results['avg_my_rating']
                st.metric("Avg. Rating", f"{avg_rating:.2f} ⭐")
        
        with col2:
            # Books to read
            tbr_count = len(st.session_state.tbr_books) if st.session_state.tbr_books is not None else 0
            st.metric("Books to Read", tbr_count)
            
            # Avg pages if available
            if 'avg_pages' in st.session_state.results:
                avg_pages = st.session_state.results['avg_pages']
                st.metric("Avg. Book Length", f"{avg_pages:.0f} pages")
        
        with col3:
            # Author diversity if available
            if 'author_diversity' in st.session_state.author_results:
                st.metric("Unique Authors", st.session_state.author_results['author_diversity'])
        
        # Reading frequency over time
        st.subheader("Reading Frequency")
        if 'books_by_year' in st.session_state.results and not st.session_state.results['books_by_year'].empty:
            year_fig = plot_reading_frequency(st.session_state.results, 'year')
            st.pyplot(year_fig)
        else:
            st.info("Not enough data to display reading frequency.")
    
    # Tab 2: Reading Patterns
    with tab2:
        st.header("Reading Patterns")
        
        # Rating comparison if ratings available
        st.subheader("Your Ratings vs. Goodreads Average")
        if 'avg_my_rating' in st.session_state.results and 'avg_goodreads_rating' in st.session_state.results:
            rating_fig = plot_rating_comparison(st.session_state.results)
            st.pyplot(rating_fig)
        else:
            st.info("Not enough rating data available.")
        
        # Page distribution if page data available
        st.subheader("Book Length Distribution")
        if st.session_state.read_books is not None and 'Number of Pages' in st.session_state.read_books.columns:
            pages_fig = plot_page_count_distribution(st.session_state.read_books)
            st.pyplot(pages_fig)
        else:
            st.info("Page count data not available.")
    
    # Tab 3: Author Analysis
    with tab3:
        st.header("Author Analysis")
        
        # Top authors
        st.subheader("Most Read Authors")
        if 'top_authors' in st.session_state.author_results and not st.session_state.author_results['top_authors'].empty:
            authors_fig = plot_top_authors(st.session_state.author_results)
            st.pyplot(authors_fig)
        else:
            st.info("Not enough data to display top authors.")
        
        # Author diversity stats
        st.subheader("Author Diversity")
        if 'author_diversity' in st.session_state.author_results and 'author_diversity_ratio' in st.session_state.author_results:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Authors", st.session_state.author_results['author_diversity'])
            with col2:
                diversity_ratio = st.session_state.author_results['author_diversity_ratio'] * 100
                st.metric("Author Diversity Ratio", f"{diversity_ratio:.1f}%")
                st.caption("Percentage of books by unique authors")
        else:
            st.info("Author diversity data not available.")
    
    # Tab 4: TBR Recommendations
    with tab4:
        st.header("TBR Recommendations")
        st.markdown("Books from your To-Read list that you might want to prioritize:")
        
        if st.session_state.recommendations is not None and not st.session_state.recommendations.empty:
            # Show detailed recommendations table
            st.subheader("Top Books to Read Next")
            rec_table = st.session_state.recommendations[['Title', 'Author', 'Average Rating', 'Number of Pages']].copy()
            rec_table.columns = ['Title', 'Author', 'Avg Rating', 'Pages']
            st.dataframe(rec_table, use_container_width=True)
            
            # Explain the recommendation methodology
            with st.expander("How are these recommendations generated?"):
                st.write("""
                These recommendations are based on the average Goodreads ratings of books in your to-read list.
                Books with higher community ratings are suggested as priority reads.
                """)
        else:
            st.info("Not enough data to generate recommendations. Make sure you have both read books and books on your to-read list.")

else:
    # Display instructions if no data is loaded
    st.info(
        """### Getting Started
        
        1. Export your Goodreads library data:
           - Go to Goodreads > My Books > Import/Export > Export Library
        2. Upload the CSV file using the sidebar
        3. Explore your reading insights and recommendations!
        
        **Note:** You can also use the provided sample data to see how the app works.
        """
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Smart Reading Tracker v1.0")
st.sidebar.caption("Created using Streamlit and Python")
