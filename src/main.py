# Auto-generated from root_cause_analysis_kg_main.ipynb
# Colab-specific code (pip installs, google.colab, /content paths) has been removed or adapted.


# --- Cell ---
# Data manipulation and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Graph libraries
import networkx as nx
from pyvis.network import Network

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Statistical analysis
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# LLM Integration
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("All libraries imported successfully!")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

# --- Cell ---
# Display first few rows of events data
print("=" * 80)
print("EVENT DATA - First 5 rows")
print("=" * 80)
print(events_df.head())

print("\n" + "=" * 80)
print("EVENT DATA - Info")
print("=" * 80)
print(events_df.info())

print("\n" + "=" * 80)
print("EVENT DATA - Summary Statistics")
print("=" * 80)
print(events_df.describe())

# --- Cell ---
# Display first few rows of performance data
print("=" * 80)
print("PERFORMANCE DATA - First 5 rows")
print("=" * 80)
print(perf_df.head())

print("\n" + "=" * 80)
print("PERFORMANCE DATA - Info")
print("=" * 80)
print(perf_df.info())

print("\n" + "=" * 80)
print("PERFORMANCE DATA - Summary Statistics")
print("=" * 80)
print(perf_df.describe())

# --- Cell ---
# Function to clean and preprocess events data
def clean_events_data(df):
    """
    Clean and preprocess events data:
    - Handle missing values
    - Normalize timestamps
    - Extract key features
    - Encode categories
    """
    df_clean = df.copy()

    # 1. Convert timestamp to datetime
    df_clean['TimeGenerated'] = pd.to_datetime(df_clean['TimeGenerated [UTC]'], errors='coerce')

    # 2. Handle missing values in key columns
    df_clean['EventLevelName'].fillna('Unknown', inplace=True)
    df_clean['RenderedDescription'].fillna('No description', inplace=True)
    df_clean['Computer'].fillna('Unknown', inplace=True)

    # 3. Remove duplicates
    df_clean.drop_duplicates(subset=['TimeGenerated', 'EventID', 'Computer'], keep='first', inplace=True)

    # 4. Extract severity level (numeric encoding)
    severity_map = {'Critical': 5, 'Error': 4, 'Warning': 3, 'Information': 2, 'Verbose': 1, 'Unknown': 0}
    df_clean['Severity'] = df_clean['EventLevelName'].map(severity_map)

    # 5. Create event category from Source
    df_clean['EventCategory'] = df_clean['Source'].str.split('-').str[-1]

    # 6. Sort by timestamp
    df_clean.sort_values('TimeGenerated', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean

# Apply cleaning
events_clean = clean_events_data(events_df)
print("Events data cleaned!")
print(f"Original shape: {events_df.shape} → Cleaned shape: {events_clean.shape}")
print(f"Removed {len(events_df) - len(events_clean)} duplicate rows")

# --- Cell ---
# Function to clean and preprocess performance data
def clean_perf_data(df):
    """
    Clean and preprocess performance data:
    - Handle missing values
    - Normalize timestamps
    - Create metric identifiers
    """
    df_clean = df.copy()

    # 1. Convert timestamp to datetime
    df_clean['TimeGenerated'] = pd.to_datetime(df_clean['TimeGenerated [UTC]'], errors='coerce')

    # 2. Handle missing values
    df_clean['InstanceName'].fillna('_Total', inplace=True)
    df_clean['CounterValue'] = pd.to_numeric(df_clean['CounterValue'], errors='coerce')
    df_clean.dropna(subset=['CounterValue'], inplace=True)

    # 3. Create composite metric name
    df_clean['MetricName'] = df_clean['ObjectName'] + '_' + df_clean['CounterName'].str.replace(' ', '_')

    # 4. Remove duplicates
    df_clean.drop_duplicates(subset=['TimeGenerated', 'MetricName', 'Computer'], keep='first', inplace=True)

    # 5. Sort by timestamp
    df_clean.sort_values('TimeGenerated', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean

# Apply cleaning
perf_clean = clean_perf_data(perf_df)
print("Performance data cleaned!")
print(f"Original shape: {perf_df.shape} → Cleaned shape: {perf_clean.shape}")
print(f"Removed {len(perf_df) - len(perf_clean)} rows with missing/invalid values")

# --- Cell ---
# Feature engineering for events data
def engineer_event_features(df):
    """
    Create advanced features from events data:
    - Event frequency
    - Time between events
    - Event bursts
    """
    df_feat = df.copy()

    # 1. Calculate time difference between consecutive events
    df_feat['TimeDelta'] = df_feat['TimeGenerated'].diff().dt.total_seconds()

    # 2. Event frequency per minute
    df_feat['EventsPerMinute'] = df_feat.groupby(df_feat['TimeGenerated'].dt.floor('1min'))['EventID'].transform('count')

    # 3. Rolling event count (last 5 events)
    df_feat['RollingEventCount'] = df_feat['EventID'].rolling(window=5, min_periods=1).count()

    # 4. Identify event bursts (more than 10 events per minute)
    df_feat['EventBurst'] = (df_feat['EventsPerMinute'] > 10).astype(int)

    # 5. Mean time between events by EventID
    df_feat['MeanTimeBetweenEvents'] = df_feat.groupby('EventID')['TimeDelta'].transform('mean')

    return df_feat

events_featured = engineer_event_features(events_clean)
print("Event features engineered!")
print(f"New columns added: {set(events_featured.columns) - set(events_clean.columns)}")

# --- Cell ---
# Feature engineering for performance data
def engineer_perf_features(df):
    """
    Create advanced features from performance data:
    - Rolling averages
    - Deviations from baseline
    - Threshold alerts
    """
    df_feat = df.copy()

    # Create pivot table for easier feature engineering
    pivot_df = df_feat.pivot_table(
        index=['TimeGenerated', 'Computer'],
        columns='MetricName',
        values='CounterValue',
        aggfunc='mean'
    ).reset_index()

    # Calculate rolling statistics for key metrics
    for col in pivot_df.columns[2:]:  # Skip TimeGenerated and Computer
        if pivot_df[col].dtype in ['float64', 'int64']:
            # Rolling mean (5 periods)
            pivot_df[f'{col}_RollingMean'] = pivot_df[col].rolling(window=5, min_periods=1).mean()

            # Deviation from mean
            pivot_df[f'{col}_Deviation'] = pivot_df[col] - pivot_df[col].mean()

            # Z-score for anomaly detection
            pivot_df[f'{col}_ZScore'] = (pivot_df[col] - pivot_df[col].mean()) / (pivot_df[col].std() + 1e-10)

    return pivot_df

perf_featured = engineer_perf_features(perf_clean)
print("Performance features engineered!")
print(f"Total features created: {len(perf_featured.columns)}")
print(f"Sample features: {list(perf_featured.columns[:10])}")

# --- Cell ---
# Merge datasets on time windows (nearest timestamp match)
def merge_event_perf_data(events_df, perf_df, time_window='1min'):
    """
    Merge events and performance data based on time windows.
    Groups performance data by time window and matches with events.
    """
    # Round timestamps to nearest time window
    events_df['TimeWindow'] = events_df['TimeGenerated'].dt.floor(time_window)
    perf_df['TimeWindow'] = perf_df['TimeGenerated'].dt.floor(time_window)

    # Aggregate events by time window
    events_agg = events_df.groupby(['TimeWindow', 'Computer']).agg({
        'EventID': 'count',
        'Severity': 'max',
        'EventBurst': 'max',
        'EventsPerMinute': 'mean'
    }).rename(columns={'EventID': 'EventCount'}).reset_index()

    # Merge on time window and computer
    merged_df = pd.merge(
        events_agg,
        perf_df,
        on=['TimeWindow', 'Computer'],
        how='outer'
    )

    # Fill missing values
    merged_df['EventCount'].fillna(0, inplace=True)
    merged_df['Severity'].fillna(0, inplace=True)
    merged_df['EventBurst'].fillna(0, inplace=True)

    return merged_df

unified_data = merge_event_perf_data(events_featured, perf_featured)
print("Data merged successfully!")
print(f"Unified dataset shape: {unified_data.shape}")
print(f"\nTime range: {unified_data['TimeWindow'].min()} to {unified_data['TimeWindow'].max()}")
print(unified_data.head())

# --- Cell ---
# Save cleaned and processed data
events_featured.to_csv('events_cleaned.csv', index=False)
perf_featured.to_csv('perf_cleaned.csv', index=False)
unified_data.to_csv('unified_data.csv', index=False)

print("  - Cleaned data saved!")
print("  - events_cleaned.csv")
print("  - perf_cleaned.csv")
print("  - unified_data.csv")

# --- Cell ---
# Define entity types and relationships
ENTITY_TYPES = {
    'System': 'Computer systems',
    'Component': 'System components (CPU, Memory, Disk, Network)',
    'Event': 'System events and alerts',
    'Metric': 'Performance metrics',
    'Alert': 'High-severity events'
}

RELATIONSHIP_TYPES = {
    'CAUSES': 'Entity A causes Entity B',
    'CORRELATES_WITH': 'Entity A correlates with Entity B',
    'AFFECTS': 'Entity A affects Entity B',
    'OCCURS_IN': 'Event occurs in System',
    'PRECEDES': 'Event A happens before Event B'
}

print("Knowledge Graph Schema Defined")
print("\nEntity Types:")
for entity, desc in ENTITY_TYPES.items():
    print(f"  • {entity}: {desc}")

print("\nRelationship Types:")
for rel, desc in RELATIONSHIP_TYPES.items():
    print(f"  • {rel}: {desc}")

# --- Cell ---
# Initialize Knowledge Graph
KG = nx.DiGraph()

print("Building Knowledge Graph...")

# Add system nodes
systems = events_featured['Computer'].unique()
for system in systems:
    KG.add_node(system, node_type='System', label=system)

print(f"Added {len(systems)} System nodes")

# Add event nodes (sample top 50 most frequent events)
top_events = events_featured['EventID'].value_counts().head(50)
for event_id, count in top_events.items():
    event_name = f"Event_{event_id}"
    # Get event description
    event_desc = events_featured[events_featured['EventID'] == event_id]['RenderedDescription'].iloc[0]
    # Truncate description
    event_desc = event_desc[:100] + "..." if len(event_desc) > 100 else event_desc

    KG.add_node(
        event_name,
        node_type='Event',
        label=event_name,
        description=event_desc,
        frequency=int(count)
    )

print(f"Added {len(top_events)} Event nodes")

# Add metric nodes (from performance data)
metrics = perf_clean['MetricName'].unique()[:30]  # Top 30 metrics
for metric in metrics:
    KG.add_node(metric, node_type='Metric', label=metric)

print(f"Added {len(metrics)} Metric nodes")

# Add component nodes
components = perf_clean['ObjectName'].unique()
for component in components:
    KG.add_node(component, node_type='Component', label=component)

print(f"Added {len(components)} Component nodes")

print(f"\n Total nodes: {KG.number_of_nodes()}")

# --- Cell ---
# Create OCCURS_IN relationships (Events -> Systems)
for _, row in events_featured.iterrows():
    event_name = f"Event_{row['EventID']}"
    system = row['Computer']

    if KG.has_node(event_name) and KG.has_node(system):
        if not KG.has_edge(event_name, system):
            KG.add_edge(event_name, system, relationship='OCCURS_IN', weight=1)
        else:
            # Increment weight for repeated occurrences
            KG[event_name][system]['weight'] += 1

print(f"Created OCCURS_IN relationships")

# Create AFFECTS relationships (Components -> Metrics)
for _, row in perf_clean.iterrows():
    component = row['ObjectName']
    metric = row['MetricName']

    if KG.has_node(component) and KG.has_node(metric):
        if not KG.has_edge(component, metric):
            KG.add_edge(component, metric, relationship='AFFECTS', weight=1)

print(f"Created AFFECTS relationships")

# Create temporal PRECEDES relationships (Event -> Event)
events_sorted = events_featured.sort_values('TimeGenerated')
for i in range(len(events_sorted) - 1):
    event1 = f"Event_{events_sorted.iloc[i]['EventID']}"
    event2 = f"Event_{events_sorted.iloc[i+1]['EventID']}"

    time_diff = (events_sorted.iloc[i+1]['TimeGenerated'] - events_sorted.iloc[i]['TimeGenerated']).total_seconds()

    # Only connect events that occur within 60 seconds of each other
    if time_diff <= 60 and KG.has_node(event1) and KG.has_node(event2):
        if not KG.has_edge(event1, event2):
            KG.add_edge(event1, event2, relationship='PRECEDES', weight=1, time_diff=time_diff)
        else:
            KG[event1][event2]['weight'] += 1

print(f"Created PRECEDES relationships")

print(f"\n Total edges: {KG.number_of_edges()}")

# --- Cell ---
# Visualize using matplotlib (static view)
plt.figure(figsize=(16, 12))

# Use spring layout for better visualization
pos = nx.spring_layout(KG, k=0.5, iterations=50, seed=42)

# Color nodes by type
node_colors = []
for node in KG.nodes():
    node_type = KG.nodes[node].get('node_type', 'Unknown')
    if node_type == 'System':
        node_colors.append('#FF6B6B')  # Red
    elif node_type == 'Event':
        node_colors.append('#4ECDC4')  # Teal
    elif node_type == 'Metric':
        node_colors.append('#95E1D3')  # Light green
    elif node_type == 'Component':
        node_colors.append('#F38181')  # Pink
    else:
        node_colors.append('#CCCCCC')  # Gray

# Draw nodes
nx.draw_networkx_nodes(KG, pos, node_color=node_colors, node_size=300, alpha=0.8)

# Draw edges
nx.draw_networkx_edges(KG, pos, alpha=0.3, arrows=True, arrowsize=10)

# Draw labels (only for important nodes)
important_nodes = {node: KG.nodes[node].get('label', node)
                   for node in KG.nodes()
                   if KG.nodes[node].get('node_type') in ['System', 'Component']}
nx.draw_networkx_labels(KG, pos, important_nodes, font_size=8)

plt.title('Knowledge Graph - System Events & Performance', fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('knowledge_graph_static.png', dpi=300, bbox_inches='tight')
plt.show()

print("Static graph visualization saved as 'knowledge_graph_static.png'")

# --- Cell ---
# Create interactive visualization using PyVis
def create_interactive_graph(graph, output_file='knowledge_graph.html'):
    """
    Create an interactive HTML visualization of the knowledge graph.
    """
    net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white', directed=True)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)

    # Add nodes with colors based on type
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get('node_type', 'Unknown')
        label = node_data.get('label', node)

        # Set color based on node type
        color_map = {
            'System': '#FF6B6B',
            'Event': '#4ECDC4',
            'Metric': '#95E1D3',
            'Component': '#F38181',
            'Unknown': '#CCCCCC'
        }

        net.add_node(
            node,
            label=label,
            title=f"Type: {node_type}\nNode: {label}",
            color=color_map.get(node_type, '#CCCCCC')
        )

    # Add edges
    for edge in graph.edges(data=True):
        source, target, data = edge
        relationship = data.get('relationship', 'related')
        weight = data.get('weight', 1)

        net.add_edge(
            source,
            target,
            title=f"{relationship} (weight: {weight})",
            value=weight
        )

    # Save and display
    net.save_graph(output_file)
    print(f"Interactive graph saved as '{output_file}'")
    return net

interactive_graph = create_interactive_graph(KG)
print("\n You can open 'knowledge_graph.html' in your browser to explore the interactive graph.")

# --- Cell ---
# Save graph to GEXF format (compatible with Gephi and other tools)
nx.write_gexf(KG, 'knowledge_graph.gexf')
print("Knowledge Graph saved as 'knowledge_graph.gexf'")

# Also save as GraphML
nx.write_graphml(KG, 'knowledge_graph.graphml')
print("Knowledge Graph saved as 'knowledge_graph.graphml'")

# Print graph statistics
print("\n Knowledge Graph Statistics:")
print(f"  • Total Nodes: {KG.number_of_nodes()}")
print(f"  • Total Edges: {KG.number_of_edges()}")
print(f"  • Graph Density: {nx.density(KG):.4f}")
print(f"  • Is Directed: {KG.is_directed()}")

# --- Cell ---
# Create time-series data for correlation analysis
def prepare_correlation_data(events_df, perf_df, time_window='5min'):
    """
    Prepare data for correlation analysis by aggregating over time windows.
    """
    # Aggregate events
    events_df['TimeWindow'] = events_df['TimeGenerated'].dt.floor(time_window)
    event_counts = events_df.groupby(['TimeWindow', 'EventID']).size().reset_index(name='EventCount')
    event_pivot = event_counts.pivot(index='TimeWindow', columns='EventID', values='EventCount').fillna(0)
    event_pivot.columns = [f'Event_{col}' for col in event_pivot.columns]

    # Aggregate performance metrics
    perf_df['TimeWindow'] = perf_df['TimeGenerated'].dt.floor(time_window)
    perf_agg = perf_df.pivot_table(
        index='TimeWindow',
        columns='MetricName',
        values='CounterValue',
        aggfunc='mean'
    ).fillna(method='ffill').fillna(0)

    # Merge
    combined = event_pivot.join(perf_agg, how='outer').fillna(0)

    return combined

correlation_data = prepare_correlation_data(events_featured, perf_clean)
print(f"Correlation data prepared with shape: {correlation_data.shape}")

# --- Cell ---
# Compute correlation matrix
print("Computing correlations...")
correlation_matrix = correlation_data.corr()

# Extract event-metric correlations
event_cols = [col for col in correlation_matrix.columns if col.startswith('Event_')]
metric_cols = [col for col in correlation_matrix.columns if not col.startswith('Event_')]

# Get cross-correlations (events vs metrics)
cross_corr = correlation_matrix.loc[event_cols, metric_cols]

print(f"Correlation matrix computed: {cross_corr.shape}")

# Visualize top correlations
plt.figure(figsize=(14, 10))
top_corr = cross_corr.abs().stack().nlargest(70)
top_corr_df = pd.DataFrame({
    'Event-Metric Pair': [f"{idx[0]} ↔ {idx[1]}" for idx in top_corr.index],
    'Correlation': [cross_corr.loc[idx] for idx in top_corr.index]
})

sns.barplot(data=top_corr_df, y='Event-Metric Pair', x='Correlation', palette='coolwarm')
plt.title('Top 30 Event-Metric Correlations', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('top_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

print("Correlation analysis complete!")

# --- Cell ---
# ---- PREP 1: build correlation_data ----
def prepare_correlation_data(events_df, perf_df, time_window='5min'):
    events_df = events_df.copy()
    perf_df = perf_df.copy()

    # Aggregate events
    events_df['TimeWindow'] = events_df['TimeGenerated'].dt.floor(time_window)
    event_counts = (
        events_df.groupby(['TimeWindow', 'EventID'])
                 .size()
                 .reset_index(name='EventCount')
    )
    event_pivot = event_counts.pivot(index='TimeWindow',
                                     columns='EventID',
                                     values='EventCount').fillna(0)
    event_pivot.columns = [f'Event_{col}' for col in event_pivot.columns]

    # Aggregate performance metrics
    perf_df['TimeWindow'] = perf_df['TimeGenerated'].dt.floor(time_window)
    perf_agg = perf_df.pivot_table(
        index='TimeWindow',
        columns='MetricName',
        values='CounterValue',
        aggfunc='mean'
    ).fillna(method='ffill').fillna(0)

    # Merge
    combined = event_pivot.join(perf_agg, how='outer').fillna(0)
    combined.sort_index(inplace=True)
    return combined

correlation_data = prepare_correlation_data(events_featured, perf_clean)

# ---- PREP 2: correlations (events vs metrics) ----
corr = correlation_data.corr()
event_cols  = [c for c in corr.columns if c.startswith('Event_')]
metric_cols = [c for c in corr.columns if not c.startswith('Event_')]

cross_corr = corr.loc[event_cols, metric_cols]

# Long-form pairs
pairs = cross_corr.stack().reset_index()
pairs.columns = ['Event', 'Metric', 'Correlation']
pairs['abs_corr'] = pairs['Correlation'].abs()

# --- Cell ---
# Pick the top 3–4 most variable metrics
metric_variance = correlation_data[metric_cols].var().sort_values(ascending=False)
top_metrics = metric_variance.head(4).index.tolist()

plt.figure(figsize=(12, 5))
for m in top_metrics:
    series = correlation_data[m]
    series_norm = (series - series.min()) / (series.max() - series.min() + 1e-9)
    plt.plot(series_norm.index, series_norm, label=m)

plt.title("Figure 1 – Key Metrics Over Time (normalized)", fontsize=14, fontweight="bold")
plt.xlabel("Time")
plt.ylabel("Normalized Value (0–1)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Cell ---
# Total events per time window
event_counts_ts = correlation_data[event_cols].sum(axis=1)

plt.figure(figsize=(12, 4))
plt.plot(event_counts_ts.index, event_counts_ts.values)
plt.title("Figure 2 – Total Event Activity Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Time")
plt.ylabel("Number of Events (per 5 min)")
plt.tight_layout()
plt.show()

# --- Cell ---
# ---- PREP 3: build event bundles based on co-occurrence ----
event_only = correlation_data[event_cols]
event_corr = event_only.corr()

threshold = 0.99   # treat events with corr ≥ 0.99 as a bundle
visited = set()
bundles = []

for ev in event_corr.index:
    if ev in visited:
        continue
    group = event_corr.index[event_corr.loc[ev] >= threshold].tolist()
    visited.update(group)
    bundles.append(group)

# Map each event to a bundle name
bundle_map = {}
for i, group in enumerate(bundles, start=1):
    name = f"EventBundle_{i}"
    for ev in group:
        bundle_map[ev] = name

print("Event bundles:")
for i, g in enumerate(bundles, start=1):
    print(f"  EventBundle_{i}: {g}")

# Build bundle time series (sum counts in each bundle)
bundle_df = pd.DataFrame(index=correlation_data.index)
for ev, name in bundle_map.items():
    if name not in bundle_df:
        bundle_df[name] = 0
    bundle_df[name] += correlation_data[ev]

metric_only = correlation_data[metric_cols]
correlation_data_bundles = pd.concat([bundle_df, metric_only], axis=1)

# correlations with bundles
corr_b = correlation_data_bundles.corr()
bundle_cols = [c for c in corr_b.columns if c.startswith("EventBundle_")]
metric_cols_b = [c for c in corr_b.columns if c not in bundle_cols]

cross_corr_b = corr_b.loc[bundle_cols, metric_cols_b]
pairs_b = cross_corr_b.stack().reset_index()
pairs_b.columns = ['Bundle', 'Metric', 'Correlation']
pairs_b['abs_corr'] = pairs_b['Correlation'].abs()

# --- Cell ---
# Strong correlations only
strong_b = pairs_b[pairs_b['abs_corr'] >= 0.6]

top_bundles = (
    strong_b.groupby('Bundle')['abs_corr']
            .max()
            .sort_values(ascending=False)
            .head(6)
            .index
)

top_metrics = (
    strong_b.groupby('Metric')['abs_corr']
            .max()
            .sort_values(ascending=False)
            .head(6)
            .index
)

heat_b = strong_b[strong_b['Bundle'].isin(top_bundles) &
                  strong_b['Metric'].isin(top_metrics)]

heat_pivot_b = heat_b.pivot(index='Bundle', columns='Metric', values='Correlation')

plt.figure(figsize=(10, 6))
sns.heatmap(heat_pivot_b, annot=True, fmt=".2f", center=0, cmap='coolwarm')
plt.title("Figure 3 – Strong Correlations: Event Bundles vs Metrics (|r| ≥ 0.6)",
          fontsize=14, fontweight="bold")
plt.xlabel("Metric")
plt.ylabel("Event Bundle")
plt.tight_layout()
plt.show()

# --- Cell ---
def plot_bundle_metric_timeseries(data_bundles, bundle_col, metric_col):
    ts = data_bundles[[bundle_col, metric_col]].copy().sort_index()

    # Normalize to 0–1 so they share an axis
    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-9)

    plt.figure(figsize=(12, 4))
    plt.plot(ts_norm.index, ts_norm[bundle_col], label=bundle_col, linewidth=2)
    plt.plot(ts_norm.index, ts_norm[metric_col], label=metric_col,
             linestyle="--", linewidth=2)
    plt.title(f"Figure 4 – Time Series: {bundle_col} vs {metric_col}",
              fontsize=14, fontweight="bold")
    plt.xlabel("Time")
    plt.ylabel("Normalized Value (0–1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example (use actual names that appear in Figure 3):
plot_bundle_metric_timeseries(
    correlation_data_bundles,
    bundle_col="EventBundle_1",
    metric_col="LogicalDisk_%_Idle_Time"
)

# --- Cell ---
# Add causal relationships based on high correlations
CORRELATION_THRESHOLD = 0.5  # Threshold for significant correlation

causal_edges_added = 0

for event in event_cols:
    for metric in metric_cols:
        corr_value = cross_corr.loc[event, metric]

        # Add causal edge if correlation is strong
        if abs(corr_value) > CORRELATION_THRESHOLD:
            if KG.has_node(event) and KG.has_node(metric):
                KG.add_edge(
                    event,
                    metric,
                    relationship='CORRELATES_WITH' if corr_value > 0 else 'INVERSELY_CORRELATED',
                    weight=abs(corr_value),
                    correlation=corr_value
                )
                causal_edges_added += 1

print(f"Added {causal_edges_added} causal edges based on correlations")
print(f"Updated graph: {KG.number_of_nodes()} nodes, {KG.number_of_edges()} edges")

# --- Cell ---
# Calculate PageRank to identify most influential nodes
print("Running PageRank algorithm...")
pagerank = nx.pagerank(KG, weight='weight')

# Sort by PageRank score
pagerank_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

print("\n Top 20 Most Influential Nodes (PageRank):")
print("=" * 80)
for i, (node, score) in enumerate(pagerank_sorted[:20], 1):
    node_type = KG.nodes[node].get('node_type', 'Unknown')
    label = KG.nodes[node].get('label', node)
    print(f"{i:2d}. {label:40s} | Type: {node_type:10s} | Score: {score:.6f}")

# --- Cell ---
# Calculate Degree Centrality
print("\n Calculating Degree Centrality...")
in_degree = dict(KG.in_degree())
out_degree = dict(KG.out_degree())

# Sort by in-degree (nodes that are affected by many others)
in_degree_sorted = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)

print("\n Top 15 Nodes by In-Degree (Most Affected):")
print("=" * 80)
for i, (node, degree) in enumerate(in_degree_sorted[:15], 1):
    node_type = KG.nodes[node].get('node_type', 'Unknown')
    label = KG.nodes[node].get('label', node)
    print(f"{i:2d}. {label:40s} | Type: {node_type:10s} | In-Degree: {degree}")

# Sort by out-degree (nodes that affect many others - potential root causes)
out_degree_sorted = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)

print("\n Top 15 Nodes by Out-Degree (Potential Root Causes):")
print("=" * 80)
for i, (node, degree) in enumerate(out_degree_sorted[:15], 1):
    node_type = KG.nodes[node].get('node_type', 'Unknown')
    label = KG.nodes[node].get('label', node)
    print(f"{i:2d}. {label:40s} | Type: {node_type:10s} | Out-Degree: {degree}")

# --- Cell ---
# Calculate Betweenness Centrality (nodes that connect different parts of the graph)
print("\n Calculating Betweenness Centrality...")
betweenness = nx.betweenness_centrality(KG, weight='weight')
betweenness_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

print("\n Top 15 Nodes by Betweenness Centrality (Critical Bridges):")
print("=" * 80)
for i, (node, score) in enumerate(betweenness_sorted[:15], 1):
    node_type = KG.nodes[node].get('node_type', 'Unknown')
    label = KG.nodes[node].get('label', node)
    print(f"{i:2d}. {label:40s} | Type: {node_type:10s} | Score: {score:.6f}")

# --- Cell ---
# Combine multiple metrics to identify root causes
def identify_root_causes(graph, pagerank, out_degree, betweenness, top_n=10):
    """
    Combine multiple centrality metrics to identify likely root causes.
    Focuses on Event nodes with high out-degree and PageRank.
    """
    # Normalize scores
    max_pr = max(pagerank.values())
    max_od = max(out_degree.values())
    max_bc = max(betweenness.values()) if max(betweenness.values()) > 0 else 1

    root_cause_scores = {}

    for node in graph.nodes():
        node_type = graph.nodes[node].get('node_type', 'Unknown')

        # Focus on Event nodes as potential root causes
        if node_type == 'Event':
            # Composite score: weighted combination of metrics
            pr_score = pagerank.get(node, 0) / max_pr
            od_score = out_degree.get(node, 0) / max_od
            bc_score = betweenness.get(node, 0) / max_bc

            # Weighted combination (out-degree is most important for root causes)
            composite_score = (0.4 * od_score) + (0.3 * pr_score) + (0.3 * bc_score)

            root_cause_scores[node] = {
                'composite_score': composite_score,
                'pagerank': pr_score,
                'out_degree': out_degree.get(node, 0),
                'betweenness': bc_score,
                'label': graph.nodes[node].get('label', node),
                'description': graph.nodes[node].get('description', 'N/A')
            }

    # Sort by composite score
    sorted_causes = sorted(root_cause_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)

    return sorted_causes[:top_n]

top_root_causes = identify_root_causes(KG, pagerank, out_degree, betweenness, top_n=10)

print("\n" + "=" * 100)
print("TOP 10 LIKELY ROOT CAUSES")
print("=" * 100)

for i, (node, metrics) in enumerate(top_root_causes, 1):
    print(f"\n{i}. {metrics['label']}")
    print(f"   Description: {metrics['description']}")
    print(f"   Composite Score: {metrics['composite_score']:.4f}")
    print(f"   Out-Degree: {metrics['out_degree']} (affects {metrics['out_degree']} other entities)")
    print(f"   PageRank: {metrics['pagerank']:.4f}")
    print(f"   Betweenness: {metrics['betweenness']:.4f}")
    print(f"   -" * 50)

# --- Cell ---
# Trace paths from root causes to impacted metrics
def trace_impact_paths(graph, root_cause, target_type='Metric', max_depth=3):
    """
    Find all paths from a root cause to nodes of a specific type.
    """
    target_nodes = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == target_type]

    paths = []
    for target in target_nodes:
        try:
            # Find shortest path
            path = nx.shortest_path(graph, source=root_cause, target=target)
            if len(path) <= max_depth + 1:
                paths.append(path)
        except nx.NetworkXNoPath:
            continue

    return paths

# Trace paths for top root cause
if top_root_causes:
    top_cause_node = top_root_causes[0][0]
    top_cause_label = top_root_causes[0][1]['label']

    impact_paths = trace_impact_paths(KG, top_cause_node, target_type='Metric', max_depth=3)

    print(f"\n Impact Paths from '{top_cause_label}':")
    print("=" * 80)

    for i, path in enumerate(impact_paths[:10], 1):  # Show first 10 paths
        path_str = " → ".join([KG.nodes[n].get('label', n) for n in path])
        print(f"{i}. {path_str}")

    if len(impact_paths) > 10:
        print(f"\n... and {len(impact_paths) - 10} more paths")

# --- Cell ---
from typing import List, Dict, Any

# 4.1 Shared system prompt and prompt builder

SYSTEM_PROMPT = """
You are a root cause analysis assistant.
You work over a knowledge graph of events, components, and performance metrics.
Use only the facts given in the Knowledge section.
If the graph does not contain enough information, say that clearly.
Be concise and clear.
""".strip()

def build_user_prompt(query: str, relevant_triples) -> str:
    """
    Build the user-side content (knowledge + question) that will be sent to any LLM.
    """
    bullets = "\n".join(f"- {t['text']}" for t in relevant_triples)
    return (
        "You are given facts from an IT knowledge graph.\n"
        "Use them to answer the question as clearly as possible.\n\n"
        f"Knowledge:\n{bullets}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

## 4.2 Convert Graph to Natural Language Triples

def extract_knowledge_triples(graph, max_triples: int = 500) -> List[Dict[str, Any]]:
    """
    Convert graph relationships into simple natural language triples.

    Each triple is a dictionary with:
      - subject
      - relationship
      - object
      - text   (natural language form)
      - weight (edge weight if present, else 1)
    """
    # Initialize an empty list to store all triples
    triples: List[Dict[str, Any]] = []

    # Iterate over all edges of the graph, including edge data
    for source, target, data in graph.edges(data=True):
        # Get a human readable label for the source node
        # If there is no label, fall back to the node id
        source_label = graph.nodes[source].get("label", source)

        # Same idea for the target node
        target_label = graph.nodes[target].get("label", target)

        # Determine the relationship type for this edge
        # If missing, default to a generic phrase
        relationship = data.get("relationship", "related to")

        # Build a natural language sentence for this triple
        # Example: "CPU spike causes High latency"
        triple_text = f"{source_label} {relationship.lower().replace('_', ' ')} {target_label}"

        # Append the structured triple record
        triples.append(
            {
                "subject": source_label,
                "relationship": relationship,
                "object": target_label,
                "text": triple_text,
                "weight": data.get("weight", 1),
            }
        )

        # If we hit the maximum triple count, stop
        if len(triples) >= max_triples:
            break

    # Return the full list of triples
    return triples

# Extract triples from your knowledge graph KG
knowledge_triples = extract_knowledge_triples(KG, max_triples=500)

# Quick sanity check printout
print(f"Extracted {len(knowledge_triples)} knowledge triples\n")
print("Sample triples:")
for triple in knowledge_triples[:10]:
    print(f"  • {triple['text']}")

# 4. Simple keyword based triple retrieval
def query_knowledge_graph(
    query: str,
    graph,
    triples: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant triples for a natural language query.

    Retrieval steps:
      - Lowercase the query.
      - Split the query into individual words.
      - Keep any triple whose text contains at least one of these words.
      - Sort the matched triples by weight in descending order.
      - Return only the top_k triples.
    """
    # Lowercase the query once for reuse
    query_lower = query.lower()

    # Prepare a list to collect matching triples
    relevant_triples: List[Dict[str, Any]] = []

    # Loop through all triples
    for triple in triples:
        # Lowercase the triple text for case insensitive match
        triple_text = triple["text"].lower()

        # If any query word appears inside the triple text, mark as relevant
        if any(word in triple_text for word in query_lower.split()):
            relevant_triples.append(triple)

    # Sort relevant triples by edge weight, highest first
    relevant_triples.sort(key=lambda x: x["weight"], reverse=True)

    # Return only the top_k most relevant triples
    return relevant_triples[:top_k]

# --- Cell ---
# Load a lightweight text generation model from Hugging Face
# Using Google's FLAN-T5 - a free, instruction-following model
print("Loading FLAN-T5 model (this may take a few minutes)...")

model_name = "google/flan-t5-base"  # Lightweight model that runs on Colab

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create pipeline for easier use
llm_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=False
)

print("LLM loaded successfully!")
print(f"Model: {model_name}")
print(f"Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# --- Cell ---
# Function to generate natural language answer using LLM
def generate_answer_google(query, relevant_triples, llm, system_prompt: str = SYSTEM_PROMPT):
    """
    Generate a natural language answer using the FLAN-T5 pipeline.
    """
    # Build the model input using shared helpers
    user_prompt = build_user_prompt(query, relevant_triples)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    result = llm(full_prompt, max_length=200, do_sample=False)
    answer = result[0]["generated_text"]
    return answer


# Main query function
def query_graph_llm_google(query_text, graph=KG, triples=knowledge_triples, llm=llm_pipeline):
    """
    Complete query pipeline: retrieve relevant info and generate answer.
    """
    print(f"\n Query: {query_text}")
    print("=" * 80)

    # Retrieve relevant triples
    relevant = query_knowledge_graph(query_text, graph, triples, top_k=10)

    if not relevant:
        print(" No relevant information found in the knowledge graph.")
        return None

    print(f"\n Found {len(relevant)} relevant facts:")
    for i, triple in enumerate(relevant[:5], 1):
        print(f"  {i}. {triple['text']}")

    # Generate answer using LLM
    print("\n Generating answer...")
    answer = generate_answer_google(query_text, relevant, llm)

    print("\n Answer:")
    print(answer)
    print("=" * 80)

    return answer

print("Query functions ready!")

# --- Cell ---
# Function to generate a comprehensive root cause report
def generate_root_cause_report(top_causes, graph, llm=llm_pipeline):
    """
    Generate a human-readable root cause analysis report.
    """
    report = []
    report.append("=" * 100)
    report.append("ROOT CAUSE ANALYSIS REPORT")
    report.append("=" * 100)
    report.append("\n")

    report.append("## Executive Summary\n")
    report.append(f"Based on analysis of {graph.number_of_nodes()} entities and {graph.number_of_edges()} relationships, ")
    report.append(f"we have identified {len(top_causes)} primary root causes affecting system performance.\n\n")

    report.append("## Top Root Causes\n")

    for i, (node, metrics) in enumerate(top_causes[:5], 1):
        report.append(f"\n### {i}. {metrics['label']}\n")
        report.append(f"**Description:** {metrics['description']}\n\n")
        report.append(f"**Impact Score:** {metrics['composite_score']:.4f}\n")
        report.append(f"**Affects:** {metrics['out_degree']} downstream entities\n\n")

        # Get connected nodes
        connected = list(graph.neighbors(node))[:5]
        if connected:
            report.append("**Directly Impacts:**\n")
            for conn in connected:
                conn_label = graph.nodes[conn].get('label', conn)
                conn_type = graph.nodes[conn].get('node_type', 'Unknown')
                report.append(f"  - {conn_label} ({conn_type})\n")
        report.append("\n")

    report.append("\n## Recommendations\n\n")
    report.append("1. **Monitor high-priority events:** Focus on events with high out-degree centrality.\n")
    report.append("2. **Implement early warning systems:** Set up alerts for root cause events.\n")
    report.append("3. **Review system architecture:** Consider isolating components with high impact.\n")
    report.append("4. **Conduct deeper analysis:** Investigate temporal patterns in root cause events.\n")

    report.append("\n" + "=" * 100)

    full_report = "".join(report)
    return full_report

# Generate and display the report
root_cause_report = generate_root_cause_report(top_root_causes, KG)
print(root_cause_report)

# Save report to file
with open('root_cause_report.txt', 'w') as f:
    f.write(root_cause_report)

print("\n Report saved as 'root_cause_report.txt'")

# --- Cell ---
# Example 1: Query about specific event
query_graph_llm_google("How many events precedes with another event?")

# --- Cell ---
# Create comprehensive summary dashboard
print("\n" + "="*100)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*100)

print("\n Phase 1: Data Preprocessing")
print(f"  ✓ Loaded {len(events_df)} events and {len(perf_df)} performance records")
print(f"  ✓ Cleaned data: {len(events_clean)} events, {len(perf_clean)} metrics")
print(f"  ✓ Created {len(unified_data)} unified time-series records")

print("\n Phase 2: Knowledge Graph")
print(f"  ✓ Built graph with {KG.number_of_nodes()} nodes and {KG.number_of_edges()} edges")
print(f"  ✓ Entity types: System, Component, Event, Metric")
print(f"  ✓ Relationship types: OCCURS_IN, AFFECTS, CORRELATES_WITH, PRECEDES")

print("\n Phase 3: Root Cause Analysis")
print(f"  ✓ Computed {cross_corr.size} event-metric correlations")
print(f"  ✓ Added {causal_edges_added} causal relationships")
print(f"  ✓ Identified {len(top_root_causes)} primary root causes")

print("\n Phase 4: LLM Integration")
print(f"  ✓ Loaded {model_name} model")
print(f"  ✓ Extracted {len(knowledge_triples)} knowledge triples")
print(f"  ✓ Natural language query system ready")

print("\n Generated Files:")
print("  • events_cleaned.csv")
print("  • perf_cleaned.csv")
print("  • unified_data.csv")
print("  • knowledge_graph.gexf")
print("  • knowledge_graph.graphml")
print("  • knowledge_graph.html (interactive)")
print("  • knowledge_graph_static.png")
print("  • top_correlations.png")
print("  • root_cause_report.txt")

print("\n Analysis complete! All deliverables generated.")
print("="*100)

# --- Cell ---

# Imports and OpenAI client setup

from openai import OpenAI                # New style OpenAI client
from typing import List, Dict, Any       # For type hints in our functions

# Create a single shared client instance
client = OpenAI()  # Reads OPENAI_API_KEY from environment

print("Using OpenAI model: gpt-5-mini")
print("OpenAI client ready. Make sure OPENAI_API_KEY is set in the environment.")

# Use OpenAI gpt-5-mini to generate answers
def generate_answer_openai(
    query: str,
    relevant_triples,
    model_name: str = "gpt-5-mini",
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    Generate a natural language answer using OpenAI gpt-5-mini.
    """
    # Build user content using the shared function
    user_content = build_user_prompt(query, relevant_triples)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    answer_text = completion.choices[0].message.content
    return answer_text

# Full query pipeline that prints everything nicely
def query_graph_llm_openai(
    query_text: str,
    graph=KG,
    triples: List[Dict[str, Any]] = knowledge_triples,
    model_name: str = "gpt-5-mini",
) -> str:
    """
    End to end query function.

    It will:
      - print hello at starting
      - Print the query.
      - Retrieve relevant triples.
      - Print the retrieved facts.
      - Call OpenAI gpt-5-mini to generate an answer.
      - Print and return the final answer.
    """
    # Show the incoming query
    print(f"\n Query: {query_text}")
    print("=" * 80)

    # Retrieve relevant triples using the basic keyword matcher
    relevant = query_knowledge_graph(query_text, graph, triples, top_k=10)

    # If no triples match, report and exit early
    if not relevant:
        print("No relevant information found in the knowledge graph.")
        return ""

    # Print how many relevant facts we got and show the first few
    print(f"\n Found {len(relevant)} relevant facts:")
    for i, triple in enumerate(relevant[:5], 1):
        print(f"  {i}. {triple['text']}")

    # Ask the OpenAI model for an answer
    print("\n Generating answer with OpenAI gpt-5-mini...")
    answer = generate_answer_openai(query_text, relevant, model_name=model_name)

    # Print the answer in a clear block
    print("\n Answer:")
    print(answer)
    print("=" * 80)

    # Return the answer string so you can reuse it if needed
    return answer

print("Query functions wired to OpenAI gpt-5-mini.")

# --- Cell ---


# Root cause report generator
#    - Reuses your top_root_causes and KG.
def generate_root_cause_report(
    top_causes: List[Any],
    graph,
) -> str:
    """
    Generate a human readable root cause analysis report.

    Expects:
      - top_causes: list of (node_id, metrics_dict)
        where metrics_dict has keys:
          - label
          - description
          - composite_score
          - out_degree
      - graph: NetworkX graph with node attributes.
    """
    # Start with an empty list for report lines
    report_lines: List[str] = []

    # Header section
    report_lines.append("=" * 100 + "\n")
    report_lines.append("ROOT CAUSE ANALYSIS REPORT\n")
    report_lines.append("=" * 100 + "\n\n")

    # Executive summary
    report_lines.append("## Executive Summary\n\n")
    report_lines.append(
        f"Based on analysis of {graph.number_of_nodes()} entities and "
        f"{graph.number_of_edges()} relationships, "
        f"we have identified {len(top_causes)} primary root causes affecting system performance.\n\n"
    )

    # Top causes section
    report_lines.append("## Top Root Causes\n")

    # Loop over the first five most important causes
    for i, (node, metrics) in enumerate(top_causes[:5], 1):
        report_lines.append(f"\n### {i}. {metrics['label']}\n\n")
        report_lines.append(f"**Description:** {metrics['description']}\n\n")
        report_lines.append(f"**Impact Score:** {metrics['composite_score']:.4f}\n\n")
        report_lines.append(f"**Affects:** {metrics['out_degree']} downstream entities\n\n")

        # Show a few directly connected nodes to highlight impact area
        connected_nodes = list(graph.neighbors(node))[:5]
        if connected_nodes:
            report_lines.append("**Directly Impacts:**\n")
            for conn in connected_nodes:
                conn_label = graph.nodes[conn].get("label", conn)
                conn_type = graph.nodes[conn].get("node_type", "Unknown")
                report_lines.append(f"  - {conn_label} ({conn_type})\n")
        report_lines.append("\n")

    # Recommendation section
    report_lines.append("\n## Recommendations\n\n")
    report_lines.append("1. **Monitor high priority events:** Focus on events with high out degree centrality.\n")
    report_lines.append("2. **Implement early warning systems:** Set up alerts for critical root cause events.\n")
    report_lines.append("3. **Review system architecture:** Consider isolating components with very high impact.\n")
    report_lines.append("4. **Conduct deeper analysis:** Investigate temporal patterns of recurring root cause events.\n")

    # Footer line
    report_lines.append("\n" + "=" * 100 + "\n")

    # Combine everything into a single string
    full_report = "".join(report_lines)

    return full_report

# Build and save the root cause report
root_cause_report = generate_root_cause_report(top_root_causes, KG)

# Print the report to the notebook
print(root_cause_report)

# Save it as a text file inside the Colab environment
with open("root_cause_report.txt", "w") as f:
    f.write(root_cause_report)

print("\n Report saved as 'root_cause_report.txt'")

# --- Cell ---
# Example 1: Query about specific event
query_graph_llm_openai("How many events precedes with another event?")

# --- Cell ---
# Example 1: Query about specific event flat T5
query_graph_llm_google("How many events precedes with another event?")
# Example 1: Query about specific event with chat-gpt-openai
query_graph_llm_openai("How many events precedes with another event?")
# Example 1: Query about specific event with chat-llama
query_graph_with_llm_llama("How many events precedes with another event?")

# --- Cell ---
# Example 2: Query about specific event flat T5
query_graph_llm_google("What metrics are affected by events?")
# Example 2: Query about specific event with chat-gpt-openai
query_graph_llm_openai("What metrics are affected by events?")
# Example 2: Query about specific event with chat-llama
query_graph_with_llm_llama("What metrics are affected by events?")

# --- Cell ---
# Example 3: Query about specific event flat T5
query_graph_llm_google("What are the main root causes of system issues?")
# Example 3: Query about specific event with chat-gpt-openai
query_graph_llm_openai("What are the main root causes of system issues?")
# Example 3: Query about specific event with chat-llama
query_graph_with_llm_llama("What are the main root causes of system issues?")
