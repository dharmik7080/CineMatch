"""
CineMatch Project - Phase 4: Statistical Analytics & Visualizations
Sub-Phase 4.2: Data Visualization Panel (Seaborn, Plotly, NetworkX)

Syllabus Reference:
- Unit 1.4: Statistical Analytics
- Unit 2.1: Data Visualization (Seaborn Heatmaps)
- Unit 2.2: Interactive Visuals (Plotly Express)
- Unit 2.4: Graph Theory & Topology (NetworkX Relationship Graphs)
"""

import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive background to prevent rendering errors on server
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
from django.conf import settings

def get_credits_df():
    """
    Helper to safely load the credits metadata csv dataset.
    """
    try:
        csv_path = os.path.join(settings.BASE_DIR, '..', 'credits.csv')
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print("[Analytics Engine] Error loading credits.csv:", e)
        return None

def generate_seaborn_heatmap():
    """
    Syllabus Reference: Unit 2.1 Data Visualization with Seaborn
    Generates a correlation matrix heatmap using sns.heatmap().
    Shows statistical relationships between movie attributes (popularity, rating, budget, etc.).
    """
    df = get_credits_df()
    if df is None:
        return ""
        
    # Select numerical variables for statistical correlation (Unit 1.4)
    num_cols = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count']
    df_numeric = df[num_cols].dropna()
    
    # Pearson Correlation Coefficient Calculation
    corr_matrix = df_numeric.corr()
    
    # Establish Seaborn figure
    plt.figure(figsize=(8, 6))
    
    # Dark mode theme styling (Wow Aesthetics)
    sns.set_theme(style="dark", rc={
        "axes.facecolor": "#07070b",
        "figure.facecolor": "#07070b",
        "text.color": "#ffffff",
        "axes.labelcolor": "#ffffff",
        "xtick.color": "#9ca3af",
        "ytick.color": "#9ca3af",
    })
    
    # Create premium diverging purple-blue heatmap palette
    cmap = sns.diverging_palette(275, 150, s=80, l=55, n=9, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap=cmap, 
        fmt=".2f", 
        linewidths=0.5, 
        cbar=True,
        annot_kws={"size": 10}
    )
    
    plt.title("Correlation Matrix Heatmap: Movie Performance Metas", fontsize=12, pad=15, color='#ffffff', weight='bold')
    plt.tight_layout()
    
    # Buffer serialization
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#07070b', dpi=120)
    buf.seek(0)
    encoded_png = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return encoded_png

def generate_plotly_scatter():
    """
    Syllabus Reference: Unit 2.2 Interactive Plotly Express charts
    Generates an interactive scatter chart mapping Budget vs Revenue,
    color-coded by ratings and sized by popularity, enabling zoom/hover inspect.
    """
    df = get_credits_df()
    if df is None:
        return ""
        
    # Clean records to remove zero inputs for logical plotting (Unit 1.4)
    df_filtered = df[(df['budget'] > 1000000) & (df['revenue'] > 1000000)].head(150)
    
    # Build dynamic Plotly Scatter Chart
    fig = px.scatter(
        df_filtered,
        x="budget",
        y="revenue",
        size="popularity",
        color="vote_average",
        hover_name="title",
        title="Interactive Ingress: Movie Budget vs Revenue (Color: Rating, Size: Popularity)",
        labels={'budget': 'Production Budget ($)', 'revenue': 'Global Revenue ($)', 'vote_average': 'Vote Average'},
        template="plotly_dark",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_colorbar=dict(title="Rating")
    )
    
    # Render figure div directly
    from plotly.offline import plot
    chart_div = plot(fig, output_type='div', include_plotlyjs=False)
    return chart_div

def generate_networkx_graph(watchlist_movie_ids):
    """
    Syllabus Reference: Unit 2.4 Graph Theory and Topology (NetworkX Relationship Graphing)
    Draws a relational network mapping connections between user-saved titles
    and their categorized genres, visually plotting topology.
    """
    G = nx.Graph()
    df = get_credits_df()
    
    if df is None or not watchlist_movie_ids:
        # Default mock relationship graph if user watchlist is empty
        mock_edges = [
            ("Avatar", "Sci-Fi"), ("Avatar", "Adventure"), ("Avatar", "Fantasy"),
            ("Interstellar", "Sci-Fi"), ("Interstellar", "Adventure"), ("Interstellar", "Drama"),
            ("Inception", "Sci-Fi"), ("Inception", "Action"), ("Inception", "Adventure"),
            ("The Dark Knight Rises", "Action"), ("The Dark Knight Rises", "Crime"), ("The Dark Knight Rises", "Thriller"),
            ("Sci-Fi", "Adventure"), ("Action", "Adventure")
        ]
        G.add_edges_from(mock_edges)
        
        # Annotate nodes
        for node in G.nodes():
            if node in ["Sci-Fi", "Adventure", "Fantasy", "Drama", "Action", "Crime", "Thriller"]:
                G.nodes[node]['type'] = 'genre'
            else:
                G.nodes[node]['type'] = 'movie'
    else:
        # Build live graph based on active user watchlist items
        import ast
        for movie_id in watchlist_movie_ids[:5]: # Cap at 5 nodes to maintain visibility
            row = df[df['id'] == movie_id]
            if not row.empty:
                title = row.iloc[0]['title']
                G.add_node(title, type='movie')
                
                try:
                    genres = ast.literal_eval(row.iloc[0]['genres'])
                    for g in genres[:3]: # Map top 3 genres
                        gname = g['name']
                        G.add_node(gname, type='genre')
                        G.add_edge(title, gname)
                except Exception:
                    pass
                    
    # Establish plot
    plt.figure(figsize=(8, 6))
    
    # Layout topology using Spring Layout (Fruchterman-Reingold force-directed algorithm)
    pos = nx.spring_layout(G, seed=42)
    
    # Distinguish colors between movie nodes and genre nodes
    node_colors = []
    for node in G.nodes():
        if G.nodes[node].get('type') == 'genre':
            node_colors.append('#3b82f6') # Neon Blue for Genres
        else:
            node_colors.append('#a855f7') # Neon Purple for Movies
            
    # Draw graph elements
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color=(1.0, 1.0, 1.0, 0.15))
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='#ffffff', font_family='sans-serif', font_weight='bold')
    
    plt.title("Watchlist Network Topology: Title-Genre Mapping", fontsize=12, color='#ffffff', weight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Buffer serialization
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#07070b', dpi=120)
    buf.seek(0)
    encoded_png = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return encoded_png
