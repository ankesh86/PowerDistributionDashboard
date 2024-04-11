import streamlit as st
import plotly.graph_objs as go
import networkx as nx
import pandas as pd
import numpy as np
from multipage_streamlit import State
from streamlit_echarts import st_echarts

# Function to load data from CSV files
def load_data(selected_network):
    if selected_network == 'IEEE 16 bus':
        bus_data_df = pd.read_csv('PowerDashPages/Data/output16busBusdata.csv')
        line_data_df = pd.read_csv('PowerDashPages/Data/output16busLinedata.csv')
        BasePowerOutputdata = pd.read_csv('PowerDashPages/Data/output16bus.csv')
    elif selected_network == 'IEEE 33 bus':
        bus_data_df = pd.read_csv('PowerDashPages/Data/output33busBusdata.csv')
        line_data_df = pd.read_csv('PowerDashPages/Data/output33busLinedata.csv')
        BasePowerOutputdata = pd.read_csv('PowerDashPages/Data/output33bus.csv')
    elif selected_network == 'IEEE 69 bus':
        bus_data_df = pd.read_csv('PowerDashPages/Data/output69busBusdata.csv')
        line_data_df = pd.read_csv('PowerDashPages/Data/output69busLinedata.csv')
        BasePowerOutputdata = pd.read_csv('PowerDashPages/Data/output69bus.csv')

    return bus_data_df, line_data_df, BasePowerOutputdata

def create_network_graph(bus_data_df, line_data_df):
    # Initialize the graph
    G = nx.Graph()

    # Add nodes to the graph with their respective values from bus_data_df
    for index, row in bus_data_df.iterrows():
        G.add_node(row['bus no'], value=row['Apparent_Power'])

    # Add edges to the graph with their respective values from line_data_df
    for index, row in line_data_df.iterrows():
        G.add_edge(row['from bus'], row['To bus'], value=row['Impedance'])

    # Generate positions for the nodes using a layout
    pos = nx.spring_layout(G)

    # Create edge traces with hover information
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=2, color='white'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Add midpoint invisible markers for edges to display hover information
    edge_info = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_info.append(go.Scatter(
            x=[(x0 + x1) / 2],
            y=[(y0 + y1) / 2],
            text=[f'Impedance: {G.edges[edge]["value"]}'],
            mode='markers',
            hoverinfo='text',
            marker=dict(size=10, color='rgba(0,0,0,0)'),  # Invisible marker
            showlegend=False
        ))

    # Create node traces
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=15,
            color=[],  # Will be filled in the loop below
            line=dict(width=2))
    )

    # Assign color to nodes
    for node in G.nodes():
        x, y = pos[node]
        node_color = 'red' if node == 1 else 'LightSkyBlue'
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += tuple([node_color])
        node_trace['text'] += tuple([f'Node {node}: Apparent Power {G.nodes[node]["value"]}'])
        

    # Create a figure
    fig = go.Figure(data=[edge_trace, node_trace] + edge_info,
                    layout=go.Layout(
                        title='Network Representation Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def create_vdi_chart(line_data_df, BasePowerOutputdata):
    entity_series = line_data_df.iloc[:, 0]  # Assuming 'from bus' is in the first column
    vdi_series = BasePowerOutputdata.iloc[:, 3]  # Assuming VDI values are in the fourth column
    
    df = pd.DataFrame({
        'Entity': entity_series.values,  # Use .values to ensure compatibility
        'VDI': vdi_series.values
    })
    
    max_vdi_value = df['VDI'].max()
    min_vdi_value = df['VDI'].min()
    max_vdi_entity = df[df['VDI'] == max_vdi_value]['Entity'].values[0]
    min_vdi_entity = df[df['VDI'] == min_vdi_value]['Entity'].values[0]
    
    fig = go.Figure(data=[
        go.Bar(x=df['Entity'], y=df['VDI'], marker_color='indigo')
    ])
    
    fig.update_layout(
        title_text='Voltage Deviation Index (VDI) by Entity',
        xaxis=dict(
            title='Branch',
            tickmode='array',
            tickvals=df['Entity'].tolist(),
            tickangle=-45
        ),
        yaxis=dict(
            title='VDI',
            range=[min(df['VDI']) * 0.9, max(df['VDI']) * 1.1]
        ),
        height=400,  # Set the height of the figure (in pixels)
        width=1000    # Set the width of the figure (in pixels)
        
    )
    
    fig.add_annotation(x=max_vdi_entity, y=max_vdi_value,
                       text=f"Highest: {max_vdi_value}", showarrow=True, arrowhead=1,
                       ax=0, ay=-40)
    
    fig.add_annotation(x=min_vdi_entity, y=min_vdi_value,
                       text=f"Lowest: {min_vdi_value}", showarrow=True, arrowhead=1,
                       ax=0, ay=-40)
    
    return fig

def run():
    state = State(__name__)
    # the above line is required if you want to save states across page switches.
    # you can provide your own namespace prefix to make keys unique across pages.
    # here we use __name__ for convenience.
    
    # Access and print the selected network option stored in the session state
    selected_network = st.session_state.get('selected_network', 'No network selected')
    st.write(f'Selected Network: {selected_network}')
    
    # Load data
    bus_data_df, line_data_df, BasePowerOutputdata = load_data(selected_network)
    
    st.header(":blue[Know your Network]")

    col = st.columns((6,3), gap='medium')

    with col[0]:
        # Replace hardcoded node and edge values with loaded data
        network_fig = create_network_graph(bus_data_df, line_data_df)
        st.plotly_chart(network_fig, use_container_width=True, width=700, height=200)

    with col[1]:
        st.markdown('## :blue[Overall Power Loss]')
        Apparent_power = BasePowerOutputdata.iloc[:, 0].sum()
        Active_power = BasePowerOutputdata.iloc[:, 1].sum()
        Reactive_power = BasePowerOutputdata.iloc[:, 2].sum()

        def format_value_to_kilo(value, unit="VA"):
            kilo_value = value / 1000  # Convert to kilo
            formatted_string = f"{kilo_value:.2f} k{unit}"  # Format to two decimal places
            return formatted_string

        formatted_Apparent_power = format_value_to_kilo(Apparent_power, 'VA')
        formatted_Active_power = format_value_to_kilo(Active_power, 'W')
        formatted_Reactive_power = format_value_to_kilo(Reactive_power, 'VAR')

        st.write(f'''
        - :orange[**Apparent Power Loss:**] {formatted_Apparent_power}
        - :orange[**Active Power Loss:**] {formatted_Active_power}
        - :orange[**Reactive Power Loss:**] {formatted_Reactive_power}
        - :red[#Red Node] : Represents Generator bus 
        - :blue[#Blue Nodes] : Represents the Distribution buses in the network
        - Data and Conpept utilised : [Paper on Load Flow and data Losses collection](https://www.researchgate.net/publication/333904328_Optimal_location_and_sizing_of_Photovoltaic_DG_system_using_Direct_Load_Flow_method).
        ''')


    col1 = st.columns((5.5,3.5), gap='medium')
    with col1[0]:
        st.markdown('## :blue[Voltage Deviation]')
        #st.header("Voltage Deviation Index (VDI) Chart")
        vdi_chart = create_vdi_chart(line_data_df, BasePowerOutputdata)
        st.plotly_chart(vdi_chart, use_container_width=True, width=700, height=300)


    with col1[1]:
        st.markdown('## :blue[Voltage Stability]')
        max_vsi_value = BasePowerOutputdata['VSI'].max()  # Adjusted for a mock column name 'VSI'

        # Assuming max_range_value is defined based on your data
        max_range_value = 1.1

        # Creating the gauge chart for VSI with the corrected max_vsi_value
        fig_vsi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=max_vsi_value,
            title={'text': "Voltage Stability Index (VSI)"},
            gauge={
                'axis': {'range': [None, max_range_value], 'tickwidth': 1, 'tickcolor': "green"},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [-1, 1], 'color': 'lightgray'},
                    {'range': [1, max_range_value], 'color': 'tomato'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1.05
                }
            }
        ))

        # Displaying the gauge chart in Streamlit
        st.plotly_chart(fig_vsi, use_container_width=True, width=500, height=300)



    state.save()  # MUST CALL THIS TO SAVE THE STATE!