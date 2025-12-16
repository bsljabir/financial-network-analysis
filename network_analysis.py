"""
Network Analysis Script for Multi-Asset Financial Markets
Run this after collecting data with collect_financial_data.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from datetime import datetime

print("=" * 70)
print("MULTI-ASSET FINANCIAL NETWORK ANALYSIS")
print("=" * 70)

# Load the returns data
print("\n📊 Loading data...")
try:
    returns = pd.read_csv('multi_asset_returns_2021_2024.csv', index_col=0, parse_dates=True)
    print(f"✅ Data loaded: {returns.shape[0]} days, {returns.shape[1]} assets")
except FileNotFoundError:
    print("❌ Error: Data file not found. Please run collect_financial_data.py first!")
    exit()

# Define asset categories for coloring
asset_categories = {
    'Stocks': ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOG', 'META', 'NVDA', 'BRK-B', 
               'JPM', 'JNJ', 'XOM', 'PG', 'V', 'KO', 'WMT'],
    'ETFs': ['SPY', 'QQQ', 'IAU', 'VTI', 'EEM', 'AGG', 'SLV', 'XLE', 'REET', 'VIG'],
    'Commodities': ['GC=F', 'SI=F', 'CL=F', 'BZ=F', 'NG=F', 'HG=F', 'PL=F', 'ZC=F', 'ZW=F', 'ZS=F'],
    'Crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD', 
               'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'LTC-USD'],
    'Fixed_Income': ['TLT', 'IEF', 'BND', 'BIL', 'SHV', 'HYG', 'VCSH', 'TIP', 'MBB', 'ICVT']
}

# Create color map
category_colors = {
    'Stocks': '#1f77b4',      # Blue
    'ETFs': '#ff7f0e',         # Orange
    'Commodities': '#2ca02c',  # Green
    'Crypto': '#d62728',       # Red
    'Fixed_Income': '#9467bd'  # Purple
}

node_colors = {}
for category, assets in asset_categories.items():
    for asset in assets:
        if asset in returns.columns:
            node_colors[asset] = category_colors[category]

# ============================================================================
# STEP 1: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: CORRELATION ANALYSIS")
print("=" * 70)

correlation_matrix = returns.corr()
correlation_matrix.to_csv('correlation_matrix.csv')
print("✅ Correlation matrix saved to: correlation_matrix.csv")

# Visualize correlation heatmap
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Multi-Asset Returns (2021-2024)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Correlation heatmap saved to: correlation_heatmap.png")
plt.close()

# Summary statistics
print("\nCorrelation Summary:")
print(f"  Average correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}")
print(f"  Max correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max():.3f}")
print(f"  Min correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min():.3f}")

# ============================================================================
# STEP 2: VOLATILITY ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: VOLATILITY ANALYSIS")
print("=" * 70)

# Calculate annualized volatility
volatility = returns.std() * np.sqrt(252)
volatility_df = pd.DataFrame({'Volatility': volatility})

# Add categories
volatility_df['Category'] = volatility_df.index.map(
    lambda x: next((cat for cat, assets in asset_categories.items() if x in assets), 'Unknown')
)

volatility_df.to_csv('asset_volatility.csv')
print("✅ Volatility measures saved to: asset_volatility.csv")

# Plot volatility by category
plt.figure(figsize=(14, 8))
volatility_df_sorted = volatility_df.sort_values('Volatility', ascending=True)
colors_list = [node_colors.get(idx, 'gray') for idx in volatility_df_sorted.index]
plt.barh(range(len(volatility_df_sorted)), volatility_df_sorted['Volatility'], color=colors_list)
plt.yticks(range(len(volatility_df_sorted)), volatility_df_sorted.index, fontsize=8)
plt.xlabel('Annualized Volatility', fontsize=12)
plt.title('Asset Volatility (2021-2024)', fontsize=14)
plt.tight_layout()
plt.savefig('volatility_chart.png', dpi=300, bbox_inches='tight')
print("✅ Volatility chart saved to: volatility_chart.png")
plt.close()

print("\nTop 5 Most Volatile Assets:")
for asset, vol in volatility.nlargest(5).items():
    print(f"  {asset}: {vol:.2%}")

# ============================================================================
# STEP 3: NETWORK CONSTRUCTION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: NETWORK CONSTRUCTION")
print("=" * 70)

# Try multiple thresholds
thresholds = [0.3, 0.5, 0.7]
networks = {}

for threshold in thresholds:
    G = nx.Graph()
    G.add_nodes_from(correlation_matrix.columns)
    
    # Add edges where correlation exceeds threshold
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    weight=abs(corr)
                )
    
    networks[threshold] = G
    print(f"  Threshold {threshold}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Use threshold 0.3 for main analysis
G = networks[0.3]
print(f"\n✅ Using network with threshold 0.3 for detailed analysis")

# ============================================================================
# STEP 4: NETWORK METRICS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: NETWORK METRICS CALCULATION")
print("=" * 70)

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
clustering_coeff = nx.clustering(G)

# Combine into DataFrame
metrics_df = pd.DataFrame({
    'Degree_Centrality': degree_centrality,
    'Betweenness_Centrality': betweenness_centrality,
    'Closeness_Centrality': closeness_centrality,
    'Eigenvector_Centrality': eigenvector_centrality,
    'Clustering_Coefficient': clustering_coeff
})

# Add category
metrics_df['Category'] = metrics_df.index.map(
    lambda x: next((cat for cat, assets in asset_categories.items() if x in assets), 'Unknown')
)

metrics_df.to_csv('network_metrics.csv')
print("✅ Network metrics saved to: network_metrics.csv")

# Print top assets by centrality
print("\nTop 10 Assets by Degree Centrality (Most Connected):")
for asset, cent in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
    category = metrics_df.loc[asset, 'Category']
    print(f"  {asset:12} ({category:15}): {cent:.3f}")

print("\nTop 10 Assets by Betweenness Centrality (Key Bridges):")
for asset, cent in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
    category = metrics_df.loc[asset, 'Category']
    print(f"  {asset:12} ({category:15}): {cent:.3f}")

print("\nTop 10 Assets by Eigenvector Centrality (Most Influential):")
for asset, cent in sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
    category = metrics_df.loc[asset, 'Category']
    print(f"  {asset:12} ({category:15}): {cent:.3f}")

# ============================================================================
# STEP 5: NETWORK VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: NETWORK VISUALIZATION")
print("=" * 70)

# Main network visualization
plt.figure(figsize=(24, 20))

# Use spring layout for positioning
pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

# Node sizes based on degree centrality
node_sizes = [degree_centrality[node] * 3000 for node in G.nodes()]

# Node colors based on category
colors = [node_colors.get(node, 'gray') for node in G.nodes()]

# Edge widths based on correlation strength
edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]

# Draw network
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.2)
nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=category) 
                   for category, color in category_colors.items()]
plt.legend(handles=legend_elements, loc='upper left', fontsize=12)

plt.title('Multi-Asset Financial Network (2021-2024)\nCorrelation Threshold: 0.3', 
          fontsize=18, pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('financial_network_main.png', dpi=300, bbox_inches='tight')
print("✅ Main network visualization saved to: financial_network_main.png")
plt.close()

# Create a smaller, cleaner version with only high centrality nodes
print("\n📊 Creating simplified network visualization...")
top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:25]
top_node_names = [node for node, _ in top_nodes]
G_sub = G.subgraph(top_node_names)

plt.figure(figsize=(16, 14))
pos_sub = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)
node_sizes_sub = [degree_centrality[node] * 4000 for node in G_sub.nodes()]
colors_sub = [node_colors.get(node, 'gray') for node in G_sub.nodes()]
edge_widths_sub = [G_sub[u][v]['weight'] * 3 for u, v in G_sub.edges()]

nx.draw_networkx_nodes(G_sub, pos_sub, node_size=node_sizes_sub, node_color=colors_sub, alpha=0.8)
nx.draw_networkx_edges(G_sub, pos_sub, width=edge_widths_sub, alpha=0.3)
nx.draw_networkx_labels(G_sub, pos_sub, font_size=10, font_weight='bold')

plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
plt.title('Top 25 Most Connected Assets\n(Simplified Network View)', fontsize=16, pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('financial_network_simplified.png', dpi=300, bbox_inches='tight')
print("✅ Simplified network saved to: financial_network_simplified.png")
plt.close()

# ============================================================================
# STEP 6: COMMUNITY DETECTION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: COMMUNITY DETECTION")
print("=" * 70)

from networkx.algorithms import community

# Detect communities using Louvain method
communities = community.greedy_modularity_communities(G)
print(f"Found {len(communities)} communities")

# Assign community IDs
community_map = {}
for idx, comm in enumerate(communities):
    for node in comm:
        community_map[node] = idx

# Print communities
for idx, comm in enumerate(communities):
    print(f"\nCommunity {idx+1} ({len(comm)} assets):")
    for node in list(comm)[:10]:  # Show first 10
        category = metrics_df.loc[node, 'Category']
        print(f"  {node} ({category})")
    if len(comm) > 10:
        print(f"  ... and {len(comm)-10} more")

# Visualize communities
plt.figure(figsize=(20, 18))
pos_comm = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
community_colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))
node_colors_comm = [community_colors[community_map[node]] for node in G.nodes()]

nx.draw_networkx_nodes(G, pos_comm, node_size=500, node_color=node_colors_comm, alpha=0.8)
nx.draw_networkx_edges(G, pos_comm, alpha=0.1)
nx.draw_networkx_labels(G, pos_comm, font_size=6)

plt.title('Community Detection in Multi-Asset Network', fontsize=16, pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('network_communities.png', dpi=300, bbox_inches='tight')
print("\n✅ Community visualization saved to: network_communities.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("\nGenerated Files:")
print("  1. correlation_matrix.csv - Full correlation matrix")
print("  2. correlation_heatmap.png - Correlation visualization")
print("  3. asset_volatility.csv - Volatility measures")
print("  4. volatility_chart.png - Volatility comparison")
print("  5. network_metrics.csv - All centrality measures")
print("  6. financial_network_main.png - Full network visualization")
print("  7. financial_network_simplified.png - Top 25 assets network")
print("  8. network_communities.png - Community structure")
print("\nKey Findings:")
print(f"  • Network density: {nx.density(G):.3f}")
print(f"  • Average clustering coefficient: {nx.average_clustering(G):.3f}")
print(f"  • Number of connected components: {nx.number_connected_components(G)}")
print(f"  • Network diameter: {nx.diameter(G) if nx.is_connected(G) else 'Network is not connected'}")
print("=" * 70)
