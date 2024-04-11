import streamlit as st
from multipage_streamlit import State
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def load_data(selected_network):
    if selected_network == 'IEEE 16 bus':
        Ploss_data_df = pd.read_csv('PowerDashPages/Data3/output16bus_PLoss.csv', header=None)
        max_DG_size = 28700000;
        no_of_bus = 15;
    elif selected_network == 'IEEE 33 bus':
        Ploss_data_df = pd.read_csv('PowerDashPages/Data3/output33bus_PLoss.csv', header=None)
        max_DG_size = 3715000;
        no_of_bus = 33;
    elif selected_network == 'IEEE 69 bus':
        Ploss_data_df = pd.read_csv('PowerDashPages/Data3/output69bus_PLoss.csv', header=None)
        max_DG_size = 3801890;
        no_of_bus = 69;

    Ploss_data_df.columns = [str(i) for i in range(Ploss_data_df.shape[1])]
    Ploss_data_df.index = np.arange(1, Ploss_data_df.shape[0] + 1)
    return Ploss_data_df, max_DG_size, no_of_bus

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

def custom_colorscale(value, min_val=100, max_val=900):
    normalized = (value - min_val) / (max_val - min_val)
    if value < 500:
        return f'rgb(0, {255 * normalized * 2}, 0)'
    else:
        return f'rgb({255 * (normalized - 0.5) * 2}, 0, 0)'

def exhaustive_search_with_animation_data(Ploss_df, dg_range=(0, 99)):
    num_bus_numbers = Ploss_df.shape[0]
    animation_data = []
    min_ploss = np.inf
    best_bus = None
    best_dg_size = None
    for bus in range(1, num_bus_numbers + 1):
        for dg_size in range(*dg_range):
            current_ploss = Ploss_df.iloc[bus - 1, dg_size]
            animation_data.append({'positions': [[bus, dg_size]], 'scores': [current_ploss]})
            if current_ploss < min_ploss:
                min_ploss = current_ploss
                best_bus = bus
                best_dg_size = dg_size
    return animation_data, best_bus, best_dg_size, min_ploss

def pso_optimize_with_animation_data(Ploss_df, n_particles=30, max_iter=100, dg_range=(0, 99)):
    w = 0.9
    c1 = 2.05
    c2 = 2.05
    num_bus_numbers = Ploss_df.shape[0]
    
    positions = np.zeros((n_particles, 2), dtype=int)
    positions[:, 0] = np.random.randint(1, num_bus_numbers + 1, size=n_particles)
    positions[:, 1] = np.random.randint(dg_range[0], dg_range[1], size=n_particles)
    velocities = np.random.randn(n_particles, 2)
    
    personal_best_positions = positions.copy()
    personal_best_scores = np.full(n_particles, np.inf)
    global_best_score = np.inf
    global_best_position = positions[0].copy()
    
    animation_data = []
    for _ in range(max_iter):
        iteration_data = {'positions': positions.copy(), 'scores': []}
        
        for i in range(n_particles):
            bus_number, dg_size = positions[i]
            current_score = Ploss_df.iloc[bus_number - 1, dg_size]
            iteration_data['scores'].append(current_score)
            
            if current_score < personal_best_scores[i]:
                personal_best_scores[i] = current_score
                personal_best_positions[i] = positions[i].copy()
            
            if current_score < global_best_score:
                global_best_score = current_score
                global_best_position = positions[i].copy()
        
        animation_data.append(iteration_data)
        
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - positions[i]) + c2 * r2 * (global_best_position - positions[i])
            # Correct velocity application
            positions[i] += velocities[i].astype(int)
            positions[i, 0] = np.clip(positions[i, 0], 1, num_bus_numbers)
            positions[i, 1] = np.clip(positions[i, 1], dg_range[0], dg_range[1] - 1)
    
    return animation_data, global_best_position[0], global_best_position[1], global_best_score

def run():
    state = State(__name__)
    
    # Access and print the selected network option stored in the session state
    selected_network = st.session_state.get('selected_network', 'No network selected')
 
    Ploss_data_df, max_DG_size, no_of_bus = load_data(selected_network)

    st.write(f'Selected Network: {selected_network}')

    st.header(":blue[Here we would find the optimal location using different methods]")
    c1, c2, c3 = st.columns((2,2,2))

    with c1:
        # List of options for the dropdown
        method = st.selectbox("Select the method", ("Exhaustive Search", "PSO"))

    with c2:
        dg_start, dg_end = st.slider("Select the range of DG sizes", 0, 99, (20, 80), 1)
        
    with c3:
        # Display the selected option
        st.write(f':orange[You selected:] :green[{method}]')
        actual_dg_start = dg_start * max_DG_size // 100
        actual_dg_end = dg_end * max_DG_size // 100

        finalDGstart = format_value_to_kilo(actual_dg_start, 'W')
        finalDGend = format_value_to_kilo(actual_dg_end, 'W')

        st.write(f':orange[Your selected DG size range:] :green[{finalDGstart} to {finalDGend}]')

    if st.button(":green[Start Optimization]"):
        progress_bar = st.progress(0)

        dg_range = (dg_start, dg_end + 1)  # Include the end in the range
    
        if method == "Exhaustive Search":
            animation_data, best_bus, best_dg_size, min_ploss = exhaustive_search_with_animation_data(Ploss_data_df, dg_range)
        else:  # PSO
            animation_data, best_bus, best_dg_size, min_ploss = pso_optimize_with_animation_data(Ploss_data_df, dg_range=dg_range)

        # Simplified example, actual animation setup would be similar to previous examples
        # Setup animation frames
        frames = [
            go.Frame(
                data=[go.Scatter(
                    x=[pos[0] for pos in frame['positions']],
                    y=[pos[1] for pos in frame['positions']],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=[custom_colorscale(score) for score in frame['scores']],
                    )
                )],
                name=str(i)
            )
            for i, frame in enumerate(animation_data)
        ]

        # Before setting up the figure, define initial_colors using the scores from the first frame of animation_data
        initial_scores = animation_data[0]['scores']  # Get the scores from the first frame
        initial_colors = [custom_colorscale(score) for score in initial_scores]  # Apply colorscale to these scores

            # Now, initialize the figure with the corrected initial_colors
        fig = go.Figure(
            data=[go.Scatter(
                    x=[pos[0] for pos in animation_data[0]['positions']],  # X positions from the first frame
                    y=[pos[1] for pos in animation_data[0]['positions']],  # Y positions from the first frame
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=initial_colors,  # Use the defined initial_colors here
                    )
                )],
                layout=go.Layout(
                    xaxis=dict(range=[0, Ploss_data_df.shape[0]+1], title='Bus'),  # Adjusted range for better visualization
                    yaxis=dict(range=[0, 100], title='DG Size'),#max([pos[1] for pos in animation_data[0]['positions']])+1], title='DG Size'),  # Dynamically adjust range
                    title='Dynamic Ploss Variation Over Iterations with '+method,
                    updatemenus=[{
                        'buttons': [
                            {'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                            'label': 'Play', 'method': 'animate'},
                            {'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                            'label': 'Pause', 'method': 'animate'}
                        ],
                        'direction': 'left', 'pad': {'r': 10, 't': 87}, 'showactive': False, 'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'
                    }]
                ),
                frames=frames  # Assuming frames are defined as in the previous code block
            )
        progress_bar.progress(100)

        colnew1, colnew2 = st.columns((4,2))

        with colnew1:
        # Display the figure in the Streamlit app
                st.plotly_chart(fig, use_container_width=True)
            
        base_ploss = Ploss_data_df.iloc[0,0]
        formatted_Ploss = format_value_to_kilo(min_ploss, 'W')
            
        with colnew2:
            DGinW = best_dg_size * max_DG_size // 100
            formatted_DGinW = format_value_to_kilo(DGinW, 'VA')
            st.markdown(f"##### :red[Optimal Bus:] {best_bus}")
            st.markdown(f"##### :red[Optimal DG Size:] {formatted_DGinW}")
            st.markdown(f"##### :orange[Min Ploss:] {formatted_Ploss} ({best_dg_size}% load)")  
            st.markdown(f"##### :orange[Active Power Loss:] {formatted_Ploss}", unsafe_allow_html=True)
            st.markdown(f"##### {format_difference(base_ploss, min_ploss)}", unsafe_allow_html=True)

    state.save()  # MUST CALL THIS TO SAVE THE STATE!