import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

def set_style():
    """Set the visualization style for a scholarly, elegant dark theme"""
    # Set the style to dark background
    plt.style.use('dark_background')
    
    # Configure serif fonts for scholarly look
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Libre Baskerville', 'Playfair Display', 'Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    
    # Set figure size and text properties
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    
    # Set elegant colors
    plt.rcParams['axes.facecolor'] = '#1e1e1e'  # Dark background
    plt.rcParams['figure.facecolor'] = '#121212'  # Even darker background
    plt.rcParams['text.color'] = '#f0f0f0'  # Light text
    plt.rcParams['axes.labelcolor'] = '#d4af37'  # Gold for labels
    plt.rcParams['axes.edgecolor'] = '#3a3a3a'  # Subtle edge color
    plt.rcParams['xtick.color'] = '#c5c5c5'  # Light gray for ticks
    plt.rcParams['ytick.color'] = '#c5c5c5'
    plt.rcParams['grid.color'] = '#3a3a3a'  # Subtle grid
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.5
    
    # Set spine visibility and color
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    
    # Add a light grid for readability
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def plot_reading_frequency(results, period='year'):
    """Plot reading frequency by year or month"""
    set_style()
    
    if period == 'year' and 'books_by_year' in results:
        data = results['books_by_year']
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use a gold-toned color palette for the bars
        bars = ax.bar(data['Year'], data['count'], 
                   color='#d4af37', alpha=0.8, 
                   edgecolor='#3a3a3a', linewidth=1.5)
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.0f}', ha='center', va='bottom',
                   color='#f0f0f0', fontweight='bold')
            
        # Add decorative elements and styling
        ax.set_title('Literary Journey by Year', fontweight='bold', color='#d4af37')
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Volumes Read', fontweight='bold')
        
        # Add a subtle horizontal line at y=0
        ax.axhline(y=0, color='#3a3a3a', linewidth=1.5, alpha=0.6)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#c5c5c5')
        
        # Adjust layout
        plt.tight_layout()
        return fig
    
    elif period == 'month' and 'books_by_month' in results:
        # Convert month numbers to names
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        data = results['books_by_month'].copy()
        data['Month_Name'] = data['Month'].map(month_names)
        
        # Group by year and plot with different colors for each year
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Define an elegant color palette with gold and muted tones
        colors = ['#d4af37', '#800020', '#4B0082', '#006400', '#4682B4', '#800080']
        
        # Plot each year's reading pattern
        for i, (year, group) in enumerate(data.groupby('Year')):
            # Cycle through the colors
            color = colors[i % len(colors)]
            
            # Sort by month to ensure correct order
            group = group.sort_values('Month')
            
            # Add the line plot
            ax.plot(group['Month_Name'], group['count'], marker='o', color=color, 
                   linewidth=2.5, markersize=8, label=str(year))
            
            # Add markers with a black edge for more elegant look
            ax.plot(group['Month_Name'], group['count'], 'o', color=color, 
                   markeredgecolor='#1e1e1e', markeredgewidth=1.5, markersize=8)
        
        # Add styling
        ax.set_title('Seasonal Reading Patterns', fontweight='bold', color='#d4af37')
        ax.set_xlabel('Month of the Year', fontweight='bold')
        ax.set_ylabel('Volumes Read', fontweight='bold')
        
        # Customize legend
        legend = ax.legend(title='Year', frameon=True, fontsize=10)
        legend.get_frame().set_facecolor('#2a2a2a')
        legend.get_frame().set_edgecolor('#3a3a3a')
        legend.get_title().set_color('#d4af37')
        for text in legend.get_texts():
            text.set_color('#f0f0f0')
        
        # Add grid for readability of values
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Adjust layout
        plt.tight_layout()
        return fig

def plot_rating_comparison(results):
    """Plot comparison between user ratings and Goodreads average ratings"""
    set_style()
    
    if 'avg_my_rating' in results and 'avg_goodreads_rating' in results:
        labels = ['Your Critique', 'Goodreads Consensus']
        values = [results['avg_my_rating'], results['avg_goodreads_rating']]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Use elegant colors - gold for user, burgundy for Goodreads
        bar_colors = ['#d4af37', '#800020']
        bars = ax.bar(labels, values, color=bar_colors, width=0.5,
                    edgecolor='#3a3a3a', linewidth=1.5)
        
        # Add decorative elements - star rating symbols
        for i, bar in enumerate(bars):
            height = bar.get_height()
            
            # Add star symbol and rating value
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'★ {height:.2f}', ha='center', va='bottom',
                   color='#f0f0f0', fontweight='bold', fontsize=14)
            
            # Add subtle text description below each bar
            if i == 0:
                ax.text(bar.get_x() + bar.get_width()/2., -0.25,
                       'Your discerning taste', ha='center', va='top',
                       color='#a0a0a0', fontsize=10, fontstyle='italic')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., -0.25,
                       'The collective opinion', ha='center', va='top',
                       color='#a0a0a0', fontsize=10, fontstyle='italic')
        
        # Enhance with styling
        ax.set_title('Critical Assessment Comparison', fontweight='bold', color='#d4af37')
        ax.set_ylabel('Rating (★ out of 5)', fontweight='bold')
        ax.set_ylim(0, 5.5)  # Set y-axis limit to make room for text
        
        # Add subtle grid lines for readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add a horizontal line at the perfect score of 5
        ax.axhline(y=5, color='#c5c5c5', linestyle=':', linewidth=1, alpha=0.5)
        ax.text(1.5, 5.05, 'Perfect Score', ha='center', va='bottom',
               color='#c5c5c5', fontsize=9, fontstyle='italic', alpha=0.8)
        
        # Remove x-axis line but keep ticks
        ax.spines['bottom'].set_color('#3a3a3a')
        ax.tick_params(axis='x', colors='#c5c5c5', direction='out', length=6)
        
        # Adjust layout
        plt.tight_layout()
        return fig

def plot_page_count_distribution(read_books):
    """Plot distribution of book lengths with scholarly styling"""
    set_style()
    
    if 'Number of Pages' in read_books.columns and not read_books['Number of Pages'].isna().all():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the data for plotting
        page_data = read_books['Number of Pages'].dropna()
        
        # Create custom bins for better visual distribution
        bins = np.linspace(0, max(page_data) + 100, 25)
        
        # Plot histogram with custom styling
        n, bins, patches = ax.hist(page_data, bins=bins, color='#d4af37', alpha=0.7,
                                  edgecolor='#3a3a3a', linewidth=1.5)
        
        # Add a kde curve for elegant visualization
        kde_x = np.linspace(0, max(page_data) + 100, 300)
        kde = sns.kdeplot(page_data, ax=ax, color='#800020', linewidth=2.5, label='Density')
        
        # Add vertical line for average with annotation
        avg_pages = page_data.mean()
        ax.axvline(x=avg_pages, color='#f0f0f0', linestyle='--', linewidth=2, 
                 label=f'Average: {avg_pages:.0f} pages')
        
        # Add text annotation for the average
        ax.text(avg_pages + 20, ax.get_ylim()[1] * 0.9, 
               f'Average Volume: {avg_pages:.0f} pages', 
               color='#f0f0f0', fontsize=10, fontstyle='italic')
        
        # Add styling
        ax.set_title('Literary Volume Analysis', fontweight='bold', color='#d4af37')
        ax.set_xlabel('Number of Pages', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        
        # Customize legend
        legend = ax.legend(frameon=True, loc='upper right')
        legend.get_frame().set_facecolor('#2a2a2a')
        legend.get_frame().set_edgecolor('#3a3a3a')
        for text in legend.get_texts():
            text.set_color('#f0f0f0')
            
        # Add a text annotation with interesting insights
        quartiles = np.percentile(page_data, [25, 50, 75])
        annotation_text = f'''Quartiles:
25%: {quartiles[0]:.0f} pages
50% (median): {quartiles[1]:.0f} pages
75%: {quartiles[2]:.0f} pages'''
        
        # Place annotation box in upper left
        props = dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', alpha=0.7, edgecolor='#3a3a3a')
        ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props, color='#c5c5c5')
        
        # Adjust layout
        plt.tight_layout()
        return fig

def plot_top_authors(author_results):
    """Plot top authors by book count with scholarly styling"""
    set_style()
    
    if 'top_authors' in author_results and not author_results['top_authors'].empty:
        # Sort data by count descending for proper display
        data = author_results['top_authors'].sort_values('Count', ascending=True)
        
        # Limit to top 10 if there are more
        if len(data) > 10:
            data = data.tail(10)
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a gold gradient palette
        gold_palette = []
        for i in range(len(data)):
            # Gradient from dark gold to light gold
            gold_shade = f'#{min(255, 180 + i*5):02x}{min(255, 150 + i*7):02x}{min(255, 40 + i*5):02x}'
            gold_palette.append(gold_shade)
            
        # Plot horizontal bars with custom styling
        bars = ax.barh(data['Author'], data['Count'], color=gold_palette,
                      edgecolor='#3a3a3a', linewidth=1, alpha=0.9)
        
        # Add count labels with elegant styling
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                   f'{width:.0f}', va='center', color='#f0f0f0',
                   fontweight='bold', fontsize=10)
            
        # Add a decorative quotation mark to each author name
        for i, author in enumerate(data['Author']):
            ax.text(-0.5, i, '“', color='#d4af37', fontsize=16,
                   va='center', ha='right', fontweight='bold')
            
        # Add styling
        ax.set_title('Distinguished Authors in Your Collection', fontweight='bold', color='#d4af37')
        ax.set_xlabel('Number of Works', fontweight='bold')
        
        # Remove y-axis label as it's redundant
        ax.set_ylabel('')
        
        # Customize tick parameters
        ax.tick_params(axis='y', which='major', pad=20)  # Add padding for author names
        
        # Add subtle grid lines only for x-axis
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add a text annotation about the collection
        total_books = data['Count'].sum()
        unique_authors = len(data)
        
        annotation_text = f'''Your Literary Companions:
{total_books} works by these {unique_authors} distinguished authors'''
        
        # Place annotation box
        props = dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', alpha=0.7, edgecolor='#3a3a3a')
        ax.text(0.05, 0.05, annotation_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', bbox=props, color='#c5c5c5')
        
        # Adjust layout
        plt.tight_layout()
        return fig

def plot_genre_distribution(author_results):
    """Plot genre/shelf distribution with scholarly styling"""
    set_style()
    
    if 'top_shelves' in author_results and not author_results['top_shelves'].empty:
        # Sort and limit to top genres if too many
        data = author_results['top_shelves'].sort_values('Count', ascending=False)
        if len(data) > 8:
            data = data.head(8)
        
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw=dict(aspect='equal'))
        
        # Create a custom gold and burgundy color palette
        gold_base = np.array([212, 175, 55]) / 255.0  # Gold base color
        burgundy_base = np.array([128, 0, 32]) / 255.0  # Burgundy base color
        
        colors = []
        for i in range(len(data)):
            # Alternate between gold and burgundy variants
            if i % 2 == 0:
                # Gold variant
                color = gold_base * (0.6 + 0.4 * (i / len(data)))
                colors.append(tuple(np.clip(color, 0, 1)))
            else:
                # Burgundy variant
                color = burgundy_base * (0.6 + 0.4 * (i / len(data)))
                colors.append(tuple(np.clip(color, 0, 1)))
        
        # Create wedges with custom styling
        wedges, texts, autotexts = ax.pie(
            data['Count'], 
            labels=None,  # We'll add custom labels outside
            autopct='%1.1f%%', 
            startangle=90, 
            shadow=False,
            colors=colors,
            wedgeprops=dict(width=0.5, edgecolor='#1e1e1e', linewidth=1.5),
            textprops=dict(color='#f0f0f0', fontweight='bold')
        )
        
        # Customize percentage text
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        # Make the circle dark in the center
        centre_circle = plt.Circle((0, 0), 0.35, fc='#121212', ec='#3a3a3a', linewidth=1.5)
        ax.add_patch(centre_circle)
        
        # Add an elegant title in the center
        ax.text(0, 0, 'Literary\nTastes', ha='center', va='center', fontsize=16, 
              fontweight='bold', color='#d4af37', fontfamily='serif')
        
        # Create custom legend with shelf names and counts
        legend_elements = []
        for i, (shelf, count) in enumerate(zip(data['Shelf'], data['Count'])):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'{shelf} ({count})',
                                            markerfacecolor=colors[i], markersize=10))
        
        # Place legend with custom styling
        legend = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                          frameon=True, title='Genres & Categories')
        legend.get_frame().set_facecolor('#2a2a2a')
        legend.get_frame().set_edgecolor('#3a3a3a')
        legend.get_title().set_color('#d4af37')
        legend.get_title().set_fontweight('bold')
        for text in legend.get_texts():
            text.set_color('#f0f0f0')
        
        # Set title for the entire chart
        ax.set_title('Literary Genre Exploration', fontweight='bold', color='#d4af37', y=1.05)
        
        # Adjust layout
        plt.tight_layout()
        return fig

def plot_reading_time(read_books):
    """Plot time taken to read books with scholarly styling"""
    set_style()
    
    if 'Days to Read' in read_books.columns and not read_books['Days to Read'].isna().all():
        # Filter out extreme outliers
        q1 = read_books['Days to Read'].quantile(0.25)
        q3 = read_books['Days to Read'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        filtered_data = read_books[read_books['Days to Read'] <= upper_bound]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create custom styled box plot
        boxplot = ax.boxplot(filtered_data['Days to Read'].dropna(), vert=True, patch_artist=True, 
                         widths=0.6, showmeans=True)
        
        # Customize boxplot colors for scholarly look
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(boxplot[element], color='#d4af37')
            
        # Fill box with gold color
        for patch in boxplot['boxes']:
            patch.set(facecolor='#d4af37', alpha=0.3, edgecolor='#d4af37', linewidth=2)
            
        # Style the median line
        for median in boxplot['medians']:
            median.set(color='#800020', linewidth=2.5)
            
        # Style the mean marker
        for mean in boxplot['means']:
            mean.set(marker='D', markeredgecolor='#f0f0f0', markerfacecolor='#800020',
                   markersize=8, markeredgewidth=1.5)
        
        # Add styling
        ax.set_title('Literary Immersion Duration', fontweight='bold', color='#d4af37')
        ax.set_ylabel('Days to Complete', fontweight='bold')
        
        # Remove x-tick labels (not needed for single boxplot)
        ax.set_xticks([])
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add statistical annotations
        stats = {
            'Minimum': filtered_data['Days to Read'].min(),
            'Q1 (25%)': q1,
            'Median': filtered_data['Days to Read'].median(),
            'Mean': filtered_data['Days to Read'].mean(),
            'Q3 (75%)': q3,
            'Maximum': filtered_data['Days to Read'].max()
        }
        
        # Create annotation text
        stats_text = '\n'.join([f'{k}: {v:.1f} days' for k, v in stats.items()])
        
        # Add annotation with statistical details
        props = dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', alpha=0.7, edgecolor='#3a3a3a')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props, color='#c5c5c5')
        
        # Add a literary quote about reading time
        quote = '"The time for reading, like the time for loving,\ndilates the day and expands the universe."\n— Harold Bloom'
        
        # Add elegant quote in bottom right
        quote_props = dict(boxstyle='round,pad=0.5', facecolor='#1e1e1e', alpha=0.7, edgecolor='#d4af37')
        ax.text(0.95, 0.05, quote, transform=ax.transAxes, fontsize=9,
               horizontalalignment='right', verticalalignment='bottom', 
               bbox=quote_props, color='#d4af37', fontstyle='italic')
        
        # Adjust layout
        plt.tight_layout()
        return fig

def plot_tbr_recommendations(recommendations):
    """Plot recommended books from TBR list with scholarly styling"""
    set_style()
    
    if not recommendations.empty and 'similarity_score' in recommendations.columns:
        # Sort by similarity score
        data = recommendations.sort_values('similarity_score', ascending=True).copy()
        
        # Limit to top 8 recommendations if there are more
        if len(data) > 8:
            data = data.tail(8)
            
        # Truncate long titles and add author surnames for more context
        data['Display_Title'] = data.apply(
            lambda x: (x['Title'][:35] + '...') if len(x['Title']) > 35 
            else x['Title'], axis=1
        )
        
        # Add author surname in parentheses for context
        data['Display_Title'] = data.apply(
            lambda x: f"{x['Display_Title']} ({x['Author'].split()[-1]})" 
            if pd.notna(x['Author']) and len(x['Author'].split()) > 0 
            else x['Display_Title'], axis=1
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a gradient color palette from burgundy to gold
        cmap = plt.cm.ScalarMappable(cmap=plt.cm.YlOrBr_r)
        cmap.set_array([])
        colors = cmap.to_rgba(np.linspace(0, 1, len(data)))
        
        # Plot horizontal bars with elegant styling
        bars = ax.barh(data['Display_Title'], data['similarity_score'], 
                      color=colors, edgecolor='#3a3a3a', linewidth=1,
                      height=0.7, alpha=0.9)
        
        # Add score labels with elegant styling
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', va='center', color='#f0f0f0',
                   fontweight='bold', fontsize=10)
        
        # Add decorative book icon to each recommendation
        for i, _ in enumerate(data['Display_Title']):
            ax.text(-0.05, i, '\uf02d', color='#d4af37', fontsize=14,
                   va='center', ha='center', family='FontAwesome')
            
        # Add styling
        ax.set_title('Recommended Literary Ventures', fontweight='bold', color='#d4af37')
        ax.set_xlabel('Affinity with Your Reading Preferences', fontweight='bold')
        
        # Remove y-axis label as it's redundant
        ax.set_ylabel('')
        
        # Set x-axis limit to make room for text
        ax.set_xlim(0, 1.1)
        
        # Add subtle grid lines for readability
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add an elegant description
        desc = "These selections from your 'To Be Read' collection align most\nclosely with your demonstrated literary preferences and tastes."
        
        # Place annotation box with description
        props = dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', alpha=0.7, edgecolor='#3a3a3a')
        ax.text(0.5, -0.1, desc, transform=ax.transAxes, fontsize=9,
               ha='center', bbox=props, color='#c5c5c5', fontstyle='italic')
        
        # Add a scoring guide
        score_guide = "Affinity Score: 1.0 (perfect match) → 0.0 (minimal similarity)"
        ax.text(0.5, 1.05, score_guide, transform=ax.transAxes, fontsize=8,
              ha='center', color='#a0a0a0')
        
        # Adjust layout
        fig.tight_layout()
        return fig
