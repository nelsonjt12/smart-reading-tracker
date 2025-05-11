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

# Apply custom scholarly theme
def load_custom_css():
    with open("static/css/scholarly_theme.css", "r") as f:
        return f.read()
        
# Apply the custom CSS
st.markdown(f'<style>{load_custom_css()}</style>', unsafe_allow_html=True)

# Add custom icon font for decorative elements
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)

# App title and description
st.markdown("<h1><i class='fas fa-book-reader'></i> Smart Reading Tracker</h1>", unsafe_allow_html=True)
st.markdown(
    """<p class='app-description'>This scholarly application analyzes your Goodreads reading data to provide 
    insightful analytics and personalized book recommendations. Upload your Goodreads export CSV 
    to begin your literary journey.</p>""", unsafe_allow_html=True
)

# Sidebar for data loading
st.sidebar.markdown("<h2><i class='fas fa-database'></i> Data Source</h2>", unsafe_allow_html=True)

# Add a decorative separator in the sidebar
st.sidebar.markdown("<div class='sidebar-separator'><span>✦</span><span>✦</span><span>✦</span></div>", unsafe_allow_html=True)

# Input options with more elegant labels
data_source = st.sidebar.radio(
    "Select your literary data source:",
    ["Browse sample library", "Import personal collection", "Use current library"]
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

if data_source == "Browse sample library":
    file_path = "data/goodreads_library_export.csv"
    if os.path.exists(file_path):
        df, results, read_books, tbr_books, author_results, recommendations = process_data(file_path)
    else:
        st.error("Sample library not found. Please upload your personal collection.")
        data_source = "Import personal collection"

elif data_source == "Import personal collection":
    st.sidebar.markdown("<p class='import-instruction'>Upload your Goodreads library export to analyze your personal reading history.</p>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Select CSV file", type="csv")
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_path = os.path.join("data", "temp_upload.csv")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the data
        df, results, read_books, tbr_books, author_results, recommendations = process_data(temp_path)

elif data_source == "Use current library" and st.session_state.data_loaded:
    # Use data already loaded in session
    df = st.session_state.df
    results = st.session_state.results
    read_books = st.session_state.read_books
    tbr_books = st.session_state.tbr_books
    author_results = st.session_state.author_results
    recommendations = st.session_state.recommendations
    
    # Display a confirmation message
    st.sidebar.success("Using your previously loaded library.")

# Main app content (only show if data is loaded)
if st.session_state.data_loaded:
    # Create tabs for different sections with icons
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Reading Overview",
        "📊 Reading Patterns",
        "✍️ Author & Genre Analysis",
        "📚 TBR Recommendations"
    ])
    
    # Tab 1: Reading Overview
    with tab1:
        st.markdown("<h2><i class='fas fa-chart-line'></i> Reading Overview</h2>", unsafe_allow_html=True)
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Books read with elegant styling
            if read_books is not None:
                total_books = len(read_books)
                st.markdown("<h4 class='metric-header'><i class='fas fa-book'></i> Volumes Completed</h4>", unsafe_allow_html=True)
                st.metric("Volumes in Collection", total_books)
            
            # Average rating
            if results is not None and 'avg_my_rating' in results:
                avg_rating = results['avg_my_rating']
                st.markdown("<h4 class='metric-header'><i class='fas fa-star'></i> Critical Assessment</h4>", unsafe_allow_html=True)
                st.metric("Your Average Rating", f"{avg_rating:.2f} ⭐")
        
        with col2:
            # Books to read
            tbr_count = len(tbr_books) if tbr_books is not None else 0
            st.metric("Books to Read", tbr_count)
            
            # Pages read with elegant styling
            if 'Number of Pages' in read_books.columns:
                total_pages = read_books['Number of Pages'].sum()
                st.markdown("<h4 class='metric-header'><i class='fas fa-scroll'></i> Pages Traversed</h4>", unsafe_allow_html=True)
                st.metric("Literary Journey", f"{total_pages:,} pages")
        
        with col3:
            # Author diversity
            if 'author_diversity' in author_results:
                st.metric("Unique Authors", author_results['author_diversity'])
            
            # Average time to read with elegant styling
            if 'avg_days_to_read' in results:
                avg_days = results['avg_days_to_read']
                st.markdown("<h4 class='metric-header'><i class='fas fa-hourglass-half'></i> Reading Pace</h4>", unsafe_allow_html=True)
                st.metric("Average Immersion", f"{avg_days:.1f} days per volume")
        
        # Reading frequency over time
        st.subheader("Reading Frequency")
        if 'books_by_year' in results and not results['books_by_year'].empty:
            year_fig = plot_reading_frequency(results, 'year')
            st.pyplot(year_fig)
        else:
            st.info("Not enough data to display reading frequency.")
    
    # Tab 2: Reading Patterns
    with tab2:
        st.markdown("<h2><i class='fas fa-chart-bar'></i> Reading Patterns</h2>", unsafe_allow_html=True)
        
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
        st.markdown("<h2><i class='fas fa-users'></i> Author & Genre Analysis</h2>", unsafe_allow_html=True)
        
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
        st.markdown("<h2><i class='fas fa-book'></i> TBR Recommendations</h2>", unsafe_allow_html=True)
        st.markdown("<p class='recommendation-intro'>Curated selections from your To-Read list that align with your reading preferences:</p>", unsafe_allow_html=True)
        
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

# Footer with enhanced scholarly styling
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='footer'>
    <div class='footer-ornament'>✦ ✦ ✦</div>
    <div class='footer-title'><i class='fas fa-feather-alt'></i> Smart Reading Tracker</div>
    <div class='footer-edition'>Scholarly Edition v1.0</div>
    <div class='footer-subtitle'>Crafted with care for the discerning reader</div>
    <div class='footer-ornament'>✦</div>
    <div class='footer-quote'>"A reader lives a thousand lives before he dies... The man who never reads lives only one." <br>― George R.R. Martin</div>
    <div class='footer-quote'>"Reading is an act of civilization; it's one of the greatest acts of civilization because it takes the free raw material of the mind and builds castles of possibilities." <br>― Ben Okri</div>
    <div class='footer-ornament'>✦ ✦ ✦</div>
</div>
""", unsafe_allow_html=True)
