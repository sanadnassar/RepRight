import streamlit as st
import time
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RepRight | Pro Analysis", page_icon="🦾", layout="wide")

# --- CUSTOM CSS (Black & Cyan Theme) ---
st.markdown("""
    <style>
    /* Global Backgrounds */
    .stApp {
        background-color: #050505;
        color: #FFFFFF;
    }
            /* --- HIDE SCROLLBAR --- */
    
    ::-webkit-scrollbar {
        display: none;
    }
            
    html, body, .stApp {
        scrollbar-width: none; 
        -ms-overflow-style: none;
    }
    
    /* Neon Cyan Brand Colors */
    h1, h2, h3 { color: #00FFFF !important; font-family: 'Courier New', Courier, monospace; }
    
    /* Style the containers to look like the wireframe cards */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #121212;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #1E1E1E;
    }

    /* Process Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #00FFFF;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 15px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00CCCC;
        color: #000000;
        box-shadow: 0 0 15px #00FFFF;
    }
    
    /* Progress Bar Cyan Override */
    .stProgress > div > div > div > div {
        background-color: #00FFFF;
    }
    </style>
""", unsafe_allow_html=True)


# --- TOP NAVIGATION BAR ---
col_logo, col_nav, col_profile = st.columns([1, 3, 1])
with col_logo:
    st.markdown("## REPRIGHT.")
with col_nav:
    # Mimicking the top nav from the wireframe
    st.markdown("<div style='text-align: center; padding-top: 15px;'>Home &nbsp;&nbsp;&nbsp; <span style='color: #00FFFF; border-bottom: 2px solid #00FFFF; padding-bottom: 5px;'>Analysis</span> &nbsp;&nbsp;&nbsp; History &nbsp;&nbsp;&nbsp; Settings</div>", unsafe_allow_html=True)
with col_profile:
    st.markdown("<div style='text-align: right; padding-top: 15px;'>👤 <b>Sanad</b></div>", unsafe_allow_html=True)

st.divider()

# --- MAIN DASHBOARD SPLIT ---
# Left Col (Upload) is slightly smaller than Right Col (Results)
left_col, right_col = st.columns([1.2, 2])

# ==========================================
# LEFT COLUMN: UPLOAD & SETTINGS
# ==========================================
with left_col:
    st.markdown("### ANALYZE SQUAT FORM")
    
    # User Profile Card
    with st.container():
        st.markdown("""
            <div style='background-color: #1A1A1A; padding: 15px; border-radius: 8px; border-left: 4px solid #00FFFF; margin-bottom: 20px;'>
                <h4 style='margin:0; color:white;'>WELCOME BACK, SANAD!</h4>
                <p style='margin:0; color:gray; font-size: 14px;'>Last Squat Score: 88 (Great)</p>
            </div>
        """, unsafe_allow_html=True)

    # Squat Type Selection
    st.markdown("**SELECT SQUAT TYPE:**")
    squat_type = st.radio("Squat Type", ["BODYWEIGHT", "BARBELL", "FRONT SQUAT"], horizontal=True, label_visibility="collapsed")
    
    st.write("") # Spacing
    
    # File Uploader
    st.markdown("**UPLOAD VIDEO:**")
    uploaded_file = st.file_uploader("Drop File Here (Max 50MB, .mp4)", type=['mp4', 'mov'], label_visibility="collapsed")
    
    st.write("") # Spacing
    
    # Process Button & Progress Logic
    if st.button("PROCESS VIDEO"):
        if uploaded_file is None:
            st.error("Please upload a video first!")
        else:
            # The "Hollywood" Progress Bar Effect
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate ML processing time
            for percent_complete in range(100):
                time.sleep(0.03) # Adjust speed here
                progress_bar.progress(percent_complete + 1)
                
                # Update text to look like real processing
                if percent_complete < 30:
                    status_text.text("Extracting frames via OpenCV...")
                elif percent_complete < 70:
                    status_text.text("Running MediaPipe Skeletal Tracking...")
                else:
                    status_text.text("Calculating rep depth and angles...")
                    
            status_text.text("Analysis Complete!")
            time.sleep(0.5)
            status_text.empty() # Clear text after finishing
            progress_bar.empty() # Clear bar after finishing
            
            st.session_state['video_processed'] = True


# ==========================================
# RIGHT COLUMN: ANALYSIS RESULTS
# ==========================================
with right_col:
    st.markdown("### ANALYSIS RESULTS")
    
    # Only show results if processing is done
    if 'video_processed' in st.session_state and st.session_state['video_processed']:
        
        # 1. Top Metrics Row
        met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
        met_col1.metric("OVERALL SCORE", "94", "+6")
        met_col2.metric("DEPTH", "100%", "Perfect")
        met_col3.metric("BAR PATH", "98%", "Stable")
        met_col4.metric("KNEE ALIGN", "91%", "-2%")
        met_col5.metric("BACK ANGLE", "95%", "Good")
        
        # 2. Video Playback & Graph Row
        vid_col, graph_col = st.columns([1.5, 1])
        
        with vid_col:
            st.markdown("**VIDEO PLAYBACK WITH FEEDBACK:**")
            # For now, replay the uploaded video. Later, Aref's OpenCV output goes here!
            st.video(uploaded_file)
            
            # Timeline Annotations
            st.markdown("**ANNOTATED TIMELINE**")
            st.markdown("""
                <span style='background-color:#004d4d; color:#00FFFF; padding: 4px 8px; border-radius: 4px; font-size:12px;'>0:03 Low Depth</span>
                <span style='background-color:#4d4d00; color:#ffff00; padding: 4px 8px; border-radius: 4px; font-size:12px;'>0:12 Knee Cave</span>
                <span style='background-color:#004d4d; color:#00FFFF; padding: 4px 8px; border-radius: 4px; font-size:12px;'>0:18 Great Depth</span>
            """, unsafe_allow_html=True)
            
        with graph_col:
            st.markdown("**MY PROGRESS**")
            
            # Create Dummy Data for the Plotly Graph
            df = pd.DataFrame({
                "Date": ["Oct 7", "Oct 12", "Oct 19", "Oct 22", "Oct 26"],
                "Score": [25, 45, 95, 88, 94]
            })
            
            # Generate the professional Plotly Line Chart
            fig = px.line(df, x="Date", y="Score", markers=True)
            fig.update_traces(line_color='#00FFFF', marker=dict(size=8, color='#00FFFF'))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=200,
                xaxis=dict(showgrid=False, color='gray'),
                yaxis=dict(showgrid=True, gridcolor='#333333', color='gray')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent History List
            st.markdown("**RECENT HISTORY**")
            st.markdown("<div style='font-size: 14px; color: #00FFFF;'>OCT 26: 94 - Excellent (Barbell)</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 14px; color: gray;'>OCT 22: 88 - Great (Barbell)</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 14px; color: gray;'>OCT 19: 95 - Great (Bodyweight)</div>", unsafe_allow_html=True)

    else:
        # Placeholder state before a video is uploaded and processed
        st.info("Awaiting video upload and processing... The AI is resting.")
        st.markdown("""
            <div style='height: 400px; display: flex; align-items: center; justify-content: center; border: 2px dashed #333; border-radius: 10px; color: #555;'>
                Upload a video to see your skeletal analysis and RepRight metrics here.
            </div>
        """, unsafe_allow_html=True)