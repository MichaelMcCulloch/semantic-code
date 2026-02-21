import json
import numpy as np
import pandas as pd
import plotly.express as px

def main():
    print("Loading data from clustered_index.json...")
    with open('clustered_index.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for file_record in data:
        file_path = file_record.get('file', 'Unknown')
        lang = file_record.get('language', 'Unknown')
        for unit in file_record.get('units', []):
            if 'umap_coords' in unit and 'cluster_id' in unit:
                coords = unit['umap_coords']
                records.append({
                    'file': file_path,
                    'language': lang,
                    'type': unit.get('type', ''),
                    'cluster_id': str(unit['cluster_id']),
                    'text': unit.get('text', ''),
                    'x': coords[0],
                    'y': coords[1],
                    'z': coords[2] if len(coords) > 2 else 0.0,
                })

    if not records:
        print("No pre-computed coordinates found in the clustered index.")
        return

    df = pd.DataFrame(records)
    
    # Process text for hovering: replace newlines with HTML <br>
    def format_hover(text):
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return text.replace("\n", "<br>")

    df['hover_text'] = df['text'].apply(format_hover)
    df = df.drop(columns=['text'])

    df['Cluster Name'] = df['cluster_id'].apply(lambda x: "Noise (-1)" if x == "-1" else f"Cluster {x}")
    df = df.sort_values(by='Cluster Name')

    print(f"Building 3D Interactive Plotly visualizer for {len(df)} points...")
    fig = px.scatter_3d(
        df,
        x='x', y='y', z='z',
        color='Cluster Name',
        hover_name='file',
        custom_data=['type', 'hover_text'],
        opacity=0.75,
        title='Semantic Code Constellation (SFR-Embedding-Code-400M_R)'
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>"
            "<i>Unit Type:</i> %{customdata[0]}<br><br>"
            "<b>Original Code:</b><br>"
            "<span style='font-family: monospace;'>%{customdata[1]}</span>"
            "<extra></extra>"
        ),
        marker=dict(size=4)
    )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title='UMAP X',
            yaxis_title='UMAP Y',
            zaxis_title='UMAP Z',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    output_html = "visualize.html"
    fig.write_html(output_html)
    print(f"\nSuccess! Open {output_html} in your browser.")

if __name__ == '__main__':
    main()
