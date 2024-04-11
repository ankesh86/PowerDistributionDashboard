#######################
# Import libraries
import streamlit as st
import altair as alt

#######################
# Page configuration
st.set_page_config(
    page_title="Optimal Distribution Generator Placement",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


#######################
# Sidebar
with st.sidebar:
    st.title('🔌 Disribution System Parameters Dashboard')
    
    #color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    #selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

    # Radio button in the sidebar
    option = st.radio(
    ':red[Choose Distribution network:]',
    ('IEEE 16 bus', 'IEEE 33 bus', 'IEEE 69 bus'))

    st.session_state['selected_network'] = option
    st.write(f'You selected: :green[{option}]')

    st.markdown('### :blue[Resources]')
    st.write(f'''
        - :orange[**Tools:**] Python(Libraries - Pandas, Numpy, Altair, Streamlit), MATLAB
        - :orange[**Language used :**] Python for visualization, MATLAB for Load flow calculations
        ''')
    st.markdown('### :blue[References]')
    st.write(f'''  
        - [Optimization of network reconfiguration by using Particle swarm optimization](https://ieeexplore.ieee.org/document/7853268).
        - [Optimization of network by using Exhaustive Search](https://ieeexplore.ieee.org/document/8980255).    
        ''')
    




#######################

# Dashboard Main Panel

st.header('🗼Optimal DGs(Distributed Generation Units) placement')

st.markdown('''DG placement is required for :blue[improving system reliability, reducing cost, integration of non-conventional energy 
sources, improving economic and environmental effects] of installing large DG into distribution system. :red[Randomly 
placing] the local energy generators from :red[micro-grids] i.e. PV plants, Wind farms, micro-turbines etc. proves to have 
:red[increased system loss]. So :green[optimal sizing and siting] is required to :green[improve the system losses] and it has been the 
challenge to the researchers and the DISCOM companies or industry to plan the system as per improved reliability, 
reduced losses and improved voltage profle.''')

st.markdown(''':blue[Dispersed generation] or small-scale generation in the network is referred to as DGs.''')
#col = st.columns((1.5, 4.5, 2), gap='medium')
st.write(f'Network selected: :blue[{option}]')

#######################
# Adding multipage
import multipage_streamlit as mt
from PowerDashPages import DGplacement, networkdetails,optimalloc
app = mt.MultiPage()

app.add("🔎 Visualise your system", networkdetails.run)
app.add("📊 Design Your Network", DGplacement.run)
app.add("💹 Optimal DG Placement", optimalloc.run)
app.run_expander()

##########################
##Footer
col = st.columns((3, 3, 3), gap='medium')
with col[0]:
    st.markdown('##### :grey[Final Project]')
    st.markdown('''##### :grey[EECE5642 - Data Visualization]''')
    st.markdown('### :red[Northeastern University]')
with col[1]:
    st.markdown('##### :grey[Presented to -]')
    st.markdown('##### :grey[Prof. Raymond Wu]')
with col[2]: 
    st.markdown('##### :grey[Prepared by -]')
    st.markdown('##### :grey[Ankesh Kumar]')
    st.markdown('##### :grey[NU ID - 002208893]')

