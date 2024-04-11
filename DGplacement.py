import streamlit as st
from multipage_streamlit import State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt

# Function to load data from CSV files
def load_data(selected_network):
    # Load data without header
    if selected_network == 'IEEE 16 bus':
        Sloss_data_df = pd.read_csv('PowerDashPages/Data2/output16bus_SLoss.csv', header=None)
        Ploss_data_df = pd.read_csv('PowerDashPages/Data2/output16bus_PLoss.csv', header=None)
        Qloss_data_df = pd.read_csv('PowerDashPages/Data2/output16bus_QLoss.csv', header=None)
        VDI_data_df = pd.read_csv('PowerDashPages/Data2/output16bus_VtgDev.csv', header=None)
        VSI_data_df = pd.read_csv('PowerDashPages/Data2/output16bus_VSI.csv', header=None)
        max_DG_size = 28700000;
        no_of_bus = 15;
    
    elif selected_network == 'IEEE 33 bus':
        Sloss_data_df = pd.read_csv('PowerDashPages/Data2/output33bus_SLoss.csv', header=None)
        Ploss_data_df = pd.read_csv('PowerDashPages/Data2/output33bus_PLoss.csv', header=None)
        Qloss_data_df = pd.read_csv('PowerDashPages/Data2/output33bus_QLoss.csv', header=None)
        VDI_data_df = pd.read_csv('PowerDashPages/Data2/output33bus_VtgDev.csv', header=None)
        VSI_data_df = pd.read_csv('PowerDashPages/Data2/output33bus_VSI.csv', header=None)
        max_DG_size = 3715000;
        no_of_bus = 33;
    
    elif selected_network == 'IEEE 69 bus':
        Sloss_data_df = pd.read_csv('PowerDashPages/Data2/output69bus_SLoss.csv', header=None)
        Ploss_data_df = pd.read_csv('PowerDashPages/Data2/output69bus_PLoss.csv', header=None)
        Qloss_data_df = pd.read_csv('PowerDashPages/Data2/output69bus_QLoss.csv', header=None)
        VDI_data_df = pd.read_csv('PowerDashPages/Data2/output69bus_VtgDev.csv', header=None)
        VSI_data_df = pd.read_csv('PowerDashPages/Data2/output69bus_VSI.csv', header=None)
        max_DG_size = 3801890;
        no_of_bus = 69;

    # Assign column names programmatically
    num_columns = 100  # Assuming each CSV has 100 columns; adjust if different
    column_names = [f'{i}' for i in range(1, num_columns + 1)]
    Sloss_data_df.columns = column_names
    Ploss_data_df.columns = column_names
    Qloss_data_df.columns = column_names
    VDI_data_df.columns = column_names
    VSI_data_df.columns = column_names
    
    return Sloss_data_df, Ploss_data_df, Qloss_data_df, VDI_data_df, VSI_data_df, max_DG_size, no_of_bus

def create_3d_mesh_plotly(Sloss_data_df, x_value=None, y_value=None):
    bus_numbers = np.arange(1, Sloss_data_df.shape[0] + 1)
    sizes = np.arange(1, Sloss_data_df.shape[1] + 1)
    B, S = np.meshgrid(bus_numbers, sizes)
    Z = Sloss_data_df.values.T

    fig = go.Figure(data=[go.Surface(z=Z, x=B, y=S, opacity=0.5)])
    if x_value is not None and y_value is not None:
        z_value = Sloss_data_df.iloc[x_value-1, y_value-1]
        fig.add_trace(go.Scatter3d(
            x=[x_value],
            y=[y_value],
            z=[z_value],
            mode='markers',
            marker=dict(size=5, color='red', opacity=1.0)
        ))

    fig.update_layout(
        title='3D Mesh Grid of Sloss Values by Bus Number and Size',
        autosize=False,
        width=800,
        height=600,
        scene=dict(
            xaxis_title='Bus Number',
            yaxis_title='Size',
            zaxis_title='Sloss (kVA)'
        ),
        scene_aspectmode='auto'
    )
    return fig

# Function for calculating percentage difference
def calculate_percentage_difference(base_value, new_value):
    if base_value != 0:
        return ((new_value - base_value) / base_value) * 100
    else:
        return 0

# Function to calculate difference and format numbers
def format_difference(base, new):
    difference = new - base
    percent_change = (difference / base) * 100 if base != 0 else 0
    formatted_difference = format_value_to_kilo(difference, '')
    # Use green color for decrease (improvement) and red for increase (deterioration)
    color = 'green' if difference < 0 else 'red'
    if difference >= 0:
        return f'<span style="color: {color};">+{formatted_difference} ({percent_change:.2f}%)</span>'
    else:
        return f'<span style="color: {color};">{formatted_difference} ({percent_change:.2f}%)</span>'

# Function for formatting value to kilo
def format_value_to_kilo(value, unit="VA"):
    kilo_value = value / 1000
    formatted_string = f"{kilo_value:.2f} k{unit}"
    return formatted_string

# Function to create donut charts using Altair
def make_donut(input_response, input_text, minimize=True):
    chart_color = ['#E74C3C', '#781F16'] if input_response >= 0 else ['#27AE60', '#12783D'] if minimize else ['#27AE60', '#12783D'] if input_response >= 0 else ['#E74C3C', '#781F16']
    source = pd.DataFrame({"Topic": ['', input_text], "% value": [100 - abs(input_response), abs(input_response)]})
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta=alt.Theta(field="% value", type="quantitative"),
        color=alt.Color("Topic:N", scale=alt.Scale(domain=[input_text, ''], range=chart_color), legend=None),
    ).properties(width=130, height=130)
    text = plot.mark_text(align='center', baseline='middle', dy=5, color="black", fontSize=12, fontWeight=700).encode(text=alt.value(f'{abs(input_response):.2f}%'))
    return plot + text

def run():
    state = State(__name__)
    
    st.header(":blue[Power System Metric Analysis]")
    # Access and print the selected network option stored in the session state
    selected_network = st.session_state.get('selected_network', 'No network selected')
    st.write(f'Selected Network: {selected_network}')
    Sloss_data_df, Ploss_data_df, Qloss_data_df, VDI_data_df, VSI_data_df, max_DG_size, no_of_bus = load_data(selected_network)
    st.header("Here :green[select size and bus number] where you want to install !")
    c1, c2 = st.columns((2,2), gap='medium')
    with c1:
        y_value = st.slider("Select DG size limits : ", min_value=0, max_value=100, key=state("x", 50))
        st.write('Your selected DG size:', (y_value*max_DG_size*0.01)/1000 ," kW")
    with c2:
        # Define a list of options for the dropdown
        options = list(range(1, no_of_bus))

        # Create the select box widget
        x_value = st.selectbox(
            'Choose an option',  # Dropdown label
            options,             # List of dropdown options
            index=0              # Default selected index (optional)
        )

        # Display the selected option
        st.write(f'You selected bus number : {x_value}')

    base_sloss = Sloss_data_df.iloc[0,0]
    base_ploss = Ploss_data_df.iloc[0,0]
    base_qloss = Qloss_data_df.iloc[0,0]
    base_vdi = VDI_data_df.iloc[0,0]
    base_vsi = VSI_data_df.iloc[0,0]

    new_sloss = Sloss_data_df.iloc[x_value-1,y_value-1]
    new_ploss = Ploss_data_df.iloc[x_value-1,y_value-1]
    new_qloss = Qloss_data_df.iloc[x_value-1,y_value-1]
    new_vdi = VDI_data_df.iloc[x_value-1,y_value-1]
    new_vsi = VSI_data_df.iloc[x_value-1,y_value-1]

    formatted_Sloss = format_value_to_kilo(new_sloss, 'VA')
    formatted_Ploss = format_value_to_kilo(new_ploss, 'W')
    formatted_Qloss = format_value_to_kilo(new_qloss, 'VAR')
    vdi_diff = calculate_percentage_difference(base_vdi, new_vdi)
    vsi_diff = calculate_percentage_difference(base_vsi, new_vsi)

    col1, col2, col3 = st.columns((2,2,4), gap='medium')

    with col1:
        st.markdown('### :blue[Power Loss Metrics]')
        st.markdown(f"#### :orange[Apparent Power Loss:] {formatted_Sloss}", unsafe_allow_html=True)
        st.markdown(f"##### {format_difference(base_sloss, new_sloss)}", unsafe_allow_html=True)
        st.markdown(f"#### :orange[Active Power Loss:] {formatted_Ploss}", unsafe_allow_html=True)
        st.markdown(f"##### {format_difference(base_ploss, new_ploss)}", unsafe_allow_html=True)
        st.markdown(f"#### :orange[Reactive Power Loss:] {formatted_Qloss}", unsafe_allow_html=True)
        st.markdown(f"##### {format_difference(base_qloss, new_qloss)}", unsafe_allow_html=True)
  
    with col2:
        st.markdown('### :blue[Voltage Indices]')
        st.metric(label=":orange[VDI Change] :green[(Minimization)]", value=f"{new_vdi:.5f}")
        st.altair_chart(make_donut(vdi_diff, 'VDI', minimize=True))
        st.metric(label=":orange[VSI Change] :green[(Maximization)]", value=f"{new_vsi:.5f}")
        st.altair_chart(make_donut(vsi_diff, 'VSI', minimize=False))

    with col3:
        # Create the 3D mesh plot using Plotly
        fig = create_3d_mesh_plotly(Sloss_data_df, x_value=x_value, y_value=y_value)
        st.plotly_chart(fig, use_container_width=True)
    
    state.save()