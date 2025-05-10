import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Import custom modules
from scripts.goodreads_processor import (
    load_goodreads_data,
    analyze_reading_behavior,
    analyze_authors_genres,
    generate_recommendations
)
from scripts.visualizations import (
    plot_reading_frequency,
    plot_rating_comparison,
    plot_page_count_distribution,
    plot_top_authors,
    plot_genre_distribution,
    plot_reading_time,
    plot_tbr_recommendations
)

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

# Sidebar for data loading
st.sidebar.header("Data Source")

# Input options
data_source = st.sidebar.radio(
    "Select data source:",
    ["Use sample data", "Upload Goodreads export", "Use existing data"]
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
    recommendations = generate_recommendations(df, n_recommendations=10)
    
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
        df, results, read_books, tbr_books, author_results, recommendations = process_data(file_path)
    else:
        st.error("Sample data file not found. Please upload your own data.")
        data_source = "Upload Goodreads export"

elif data_source == "Upload Goodreads export":
    uploaded_file = st.sidebar.file_uploader("Upload your Goodreads export CSV", type="csv")
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_path = os.path.join("data", "temp_upload.csv")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the data
        df, results, read_books, tbr_books, author_results, recommendations = process_data(temp_path)

elif data_source == "Use existing data" and st.session_state.data_loaded:
    # Use data already loaded in session
    df = st.session_state.df
    results = st.session_state.results
    read_books = st.session_state.read_books
    tbr_books = st.session_state.tbr_books
    author_results = st.session_state.author_results
    recommendations = st.session_state.recommendations

# Main app content (only show if data is loaded)
if st.session_state.data_loaded:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Reading Overview",
        "Reading Patterns",
        "Author & Genre Analysis",
        "TBR Recommendations"
    ])
    
    # Tab 1: Reading Overview
    with tab1:
        st.header("Reading Overview")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Books read
            total_read = len(read_books) if read_books is not None else 0
            st.metric("Books Read", total_read)
            
            # Average rating
            if 'avg_my_rating' in results:
                avg_rating = results['avg_my_rating']
                st.metric("Avg. Rating", f"{avg_rating:.2f} ⭐")
        
        with col2:
            # Books to read
            tbr_count = len(tbr_books) if tbr_books is not None else 0
            st.metric("Books to Read", tbr_count)
            
            # Avg pages
            if 'avg_pages' in results:
                avg_pages = results['avg_pages']
                st.metric("Avg. Book Length", f"{avg_pages:.0f} pages")
        
        with col3:
            # Author diversity
            if 'author_diversity' in author_results:
                st.metric("Unique Authors", author_results['author_diversity'])
            
            # Reading speed
            if 'avg_days_to_read' in results:
                avg_days = results['avg_days_to_read']
                st.metric("Avg. Time to Read", f"{avg_days:.1f} days")
        
        # Reading frequency over time
        st.subheader("Reading Frequency")
        if 'books_by_year' in results and not results['books_by_year'].empty:
            year_fig = plot_reading_frequency(results, 'year')
            st.pyplot(year_fig)
        else:
            st.info("Not enough data to display reading frequency.")
    
    # Tab 2: Reading Patterns
    with tab2:
        st.header("Reading Patterns")
        
        # Rating comparison
        st.subheader("Your Ratings vs. Goodreads Average")
        if 'avg_my_rating' in results and 'avg_goodreads_rating' in results:
            rating_fig = plot_rating_comparison(results)
            st.pyplot(rating_fig)
        else:
            st.info("Not enough rating data available.")
        
        # Page distribution
        st.subheader("Book Length Distribution")
        if read_books is not None and 'Number of Pages' in read_books.columns:
            pages_fig = plot_page_count_distribution(read_books)
            st.pyplot(pages_fig)
        else:
            st.info("Page count data not available.")
        
        # Reading time
        st.subheader("Time to Read Books")
        if read_books is not None and 'Days to Read' in read_books.columns:
            time_fig = plot_reading_time(read_books)
            st.pyplot(time_fig)
        else:
            st.info("Reading time data not available.")
        
        # Monthly patterns
        st.subheader("Monthly Reading Patterns")
        if 'books_by_month' in results and not results['books_by_month'].empty:
            month_fig = plot_reading_frequency(results, 'month')
            st.pyplot(month_fig)
        else:
            st.info("Not enough data to display monthly reading patterns.")
    
    # Tab 3: Author & Genre Analysis
    with tab3:
        st.header("Author & Genre Analysis")
        
        # Top authors
        st.subheader("Most Read Authors")
        if 'top_authors' in author_results and not author_results['top_authors'].empty:
            authors_fig = plot_top_authors(author_results)
            st.pyplot(authors_fig)
        else:
            st.info("Not enough data to display top authors.")
        
        # Genre distribution
        st.subheader("Genre Distribution")
        if 'top_shelves' in author_results and not author_results['top_shelves'].empty:
            genres_fig = plot_genre_distribution(author_results)
            st.pyplot(genres_fig)
        else:
            st.info("Bookshelf/genre data not available.")
        
        # Author diversity stats
        st.subheader("Author Diversity")
        if 'author_diversity' in author_results and 'author_diversity_ratio' in author_results:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Authors", author_results['author_diversity'])
            with col2:
                diversity_ratio = author_results['author_diversity_ratio'] * 100
                st.metric("Author Diversity Ratio", f"{diversity_ratio:.1f}%")
                st.caption("Percentage of books by unique authors")
        else:
            st.info("Author diversity data not available.")
    
    # Tab 4: TBR Recommendations
    with tab4:
        st.header("TBR Recommendations")
        st.markdown("Books from your To-Read list that match your reading preferences:")
        
        if recommendations is not None and not recommendations.empty:
            # Show recommendation plot
            rec_fig = plot_tbr_recommendations(recommendations)
            st.pyplot(rec_fig)
            
            # Show detailed recommendations table
            st.subheader("Detailed Recommendations")
            rec_table = recommendations[['Title', 'Author', 'Average Rating', 'Number of Pages', 'similarity_score']].copy()
            rec_table.columns = ['Title', 'Author', 'Avg Rating', 'Pages', 'Match Score']
            rec_table['Match Score'] = rec_table['Match Score'].apply(lambda x: f"{x:.2f}")
            st.dataframe(rec_table, use_container_width=True)
            
            # Explain the recommendation methodology
            with st.expander("How are these recommendations generated?"):
                st.write("""
                The recommendations are based on similarity between your highly-rated books and your to-read list. 
                The system analyzes patterns in your reading history, considering factors like:
                
                - Book metadata (title, author, publisher)
                - Bookshelf tags and categories
                - Average ratings and page counts
                
                Books that align most closely with your preferred reading patterns receive higher match scores.                
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
