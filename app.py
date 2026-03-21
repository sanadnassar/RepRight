import streamlit as st
import time
import pandas as pd
import plotly.express as px
import tempfile
from core.video_processor import process_video

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RepRight", page_icon="🦾", layout="wide")

# --- CUSTOM CSS (Black & Cyan Theme) ---
st.markdown("""
    <style>
    /* Global Backgrounds */
    .stApp {
        background-color: #050505;
        color: #FFFFFF;
    }
            
    /* Scale the entire app down to 90% */
    html {
        zoom: 0.9; 
    }
            
    /* --- KILL THE FOOTER COMPLETELY --- */
    footer, 
    [data-testid="stFooter"], 
    [data-testid="stBottom"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
            

    /* Hide the anchor link icon next to headers */
    button.step-down, .element-container:has(h1) a, .element-container:has(h2) a, .element-container:has(h3) a {
        display: none;
    }

    /* More direct approach for modern Streamlit versions */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
            
    /* --- HIDE SCROLLBAR BUT KEEP SCROLLING --- */
    /* Target the main scrollable containers in Streamlit */
    [data-testid="stMainBlockContainer"], 
    .main, 
    .stApp {
        overflow-y: auto; /* THIS IS THE FIX: Allows vertical scrolling */
        overflow-x: hidden; /* Prevents accidental left/right scrolling */
        scrollbar-width: none; /* Firefox */
        -ms-overflow-style: none; /* IE/Edge */
    }

    /* Chrome, Safari, and Opera */
    [data-testid="stMainBlockContainer"]::-webkit-scrollbar, 
    .main::-webkit-scrollbar, 
    .stApp::-webkit-scrollbar {
        display: none;
    }
            


    
    /* Neon Cyan Brand Colors */
    h1, h2, h3 { color: #00FFFF !important; font-family: 'Courier New', Courier, monospace; }
    
    /* Style the containers to look like the wireframe cards */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="erticalBlock"] {
        background-color: #121212;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #1E1E1E;
    }

    /* stButton Styling */
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

    /* Constrain File Uploader & PREVENT HEIGHT JUMPING */
    [data-testid="stFileUploader"] {
        width: 390px !important;
        height: 145px !important; /* Locks the total height of the area */
        display: flex;
        flex-direction: column;
    }

    /* Shrink the drag-and-drop zone slightly so the uploaded file info has room to appear */
    [data-testid="stFileUploadDropzone"] {
        padding: 15px !important; 
        min-height: 80px !important;
    }
    
    /* Constrain the Button container to match */
    [data-testid="stButton"] {
        width: 390px !important;
    }
    
    /* Ensure the button itself fills that 406px container */
    div[data-testid="stButton"] > button {
        width: 100% !important;
    }
            


    /* ========== RADIO BUTTON STYLING ========== */
    
    /* Section Labels */
    .section-label {
        color: #888;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    /* Radio button container - make horizontal and equal width */
    [data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        gap: 10px !important;
        width: 406px !important;
    }
    
    /* Each radio option - equal width */
    [data-testid="stRadio"] > div > label {
        flex: 1 !important;
        background-color: #1A1A1A !important;
        padding: 12px 15px !important;
        border-radius: 8px !important;
        border: 1px solid #333 !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
            
    /* Selected radio option */
    [data-testid="stRadio"] > div > label:has(input:checked) {
        background-color: #002222 !important; /* A nice deep cyan-tinted black */
        border: 1px solid #00FFFF !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3) !important;
    }
    
    /* Selected radio text */
    [data-testid="stRadio"] > div > label:has(input:checked) p {
        color: #00FFFF !important;
        font-weight: 800 !important;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.5) !important;
    }
    
    /* Radio option hover effect */
    [data-testid="stRadio"] > div > label:hover {
        border-color: #00FFFF !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.3) !important;
    }
    
    /* Selected radio option */
    [data-testid="stRadio"] > div > label[data-checked="true"] {
        background-color: #0a2a2a !important;
        border: 1px solid #00FFFF !important;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.3) !important;
    }
    
    /* Radio button text color */
    [data-testid="stRadio"] > div > label p {
        color: #FFFFFF !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        margin-left: -5px;
    }
    
    /* Selected radio text */
    [data-testid="stRadio"] > div > label[data-checked="true"] p {
        color: #00FFFF !important;
        font-weight: bold !important;
    }
    
    /* Hide the actual radio circle */
    [data-testid="stRadio"] > div > label > div:first-child {
        display: none !important;
    }
            
    /* --- CONSTRAIN THE VIDEO PLAYER --- */
    [data-testid="stVideo"] {
        width: 100% !important;
        max-height: 400px !important; /* Prevents the video from getting too tall */
        border-radius: 10px;
        border: 1px solid #1E1E1E; /* Matches your right-column card borders */
        overflow: hidden;
        background-color: #000000; /* Keeps the background dark if the video is vertical */
    }

    [data-testid="stVideo"] video {
        width: 100% !important;
        max-height: 400px !important;
        object-fit: contain !important; /* Ensures the whole body is visible without cropping */
    }
    </style>
""", unsafe_allow_html=True)



st.markdown("""
<h1 style='text-align: center; font-size: 65px; padding-bottom: 80px; margin-top: -60px'>REPRIGHT.</h1>
""", unsafe_allow_html=True)


# --- MAIN DASHBOARD SPLIT ---
# Left Col (Upload) is slightly smaller than Right Col (Results)
left_col, right_col = st.columns([1, 2])


# ==========================================
# LEFT COLUMN: UPLOAD & SETTINGS
# ==========================================
with left_col:
    st.markdown("## ANALYZE TRAINING FORM")
    
    # User Profile Card
    with st.container():
        st.markdown("""
            <div style='background-color: #1A1A1A; padding: 15px; border-radius: 8px; border-left: 4px solid #00FFFF; margin-bottom: 20px; cursor: default; width: 390px;'>
                <h4 style='margin:0; color:white;'>WELCOME BACK, SANAD!</h4>
                <p style='margin:0; color:gray; font-size: 14px;'>PLEASE UPLOAD A VIDEO TO GET STARTED</p>
            </div>
        """, unsafe_allow_html=True)

    # Exercise Type Selection
    st.markdown("<p class='section-label'>SELECT EXERCISE TYPE</p>", unsafe_allow_html=True)
    squat_type = st.radio(
        "Exercise Type", 
        ["SQUATS", "PULL UP", "BICEP CURL"], 
        horizontal=True, 
        label_visibility="collapsed"
    )
    
    st.write("") # Spacing
    
    # File Uploader
    st.markdown("**UPLOAD VIDEO:**")
    uploaded_file = st.file_uploader("Drop File Here", type=['mp4'], label_visibility="collapsed")
    
    st.write("") # Spacing
    
    # Process Button & Progress Logic (Inside Left Column)
    if st.button("PROCESS VIDEO", use_container_width=True):
        if uploaded_file is None:
            st.error("Please upload a video first!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def ui_updater(percent, message):
                progress_bar.progress(percent)
                status_text.text(message)
                
            try:
                # ---> THE CONNECTION HAPPENS HERE <---
                # We call Aref's function and it returns the path to the annotated video + real stats
                final_video_path, real_stats = process_video(
                    video_file=uploaded_file, 
                    exercise=squat_type, 
                    progress_callback=ui_updater
                )
                
                # Save everything to session state so the Right Column can use it
                st.session_state['processed_video_path'] = final_video_path
                st.session_state['form_stats'] = real_stats
                st.session_state['video_processed'] = True
                
                status_text.text("Analysis Complete!")
                time.sleep(0.5)
                
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")
                
            finally:
                status_text.empty() 
                progress_bar.empty() 
            
    # Invisible layout spacer
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)


# ==========================================
# RIGHT COLUMN: ANALYSIS RESULTS
# ==========================================
# ==========================================
# RIGHT COLUMN: ANALYSIS RESULTS
# ==========================================
# ==========================================
# RIGHT COLUMN: ANALYSIS RESULTS
# ==========================================
# ==========================================
# RIGHT COLUMN: ANALYSIS RESULTS
# ==========================================
with right_col:
    st.markdown("""
    <h2 style='padding-top: 15px; padding-bottom: 20px; padding-left: 10px;'>
        ANALYSIS RESULTS
    </h2>
    """, unsafe_allow_html=True)
    
    # --- ONLY SHOW EVERYTHING BELOW IF PROCESSING IS DONE ---
    if 'video_processed' in st.session_state and st.session_state['video_processed']:
        stats = st.session_state['form_stats']
        
        # 1. Top Metrics Row
        met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
        met_col1.metric("AVG SCORE", f"{stats['score']}/100")
        met_col2.metric("REPS", stats["reps"])
        met_col3.metric("GOOD FORM", stats["good_pct"])
        met_col4.metric("DEPTH",        stats["depth"])
        met_col5.metric("CONSISTENCY",  stats["consistency"])
        
        vid_col, graph_col = st.columns([1.5, 1])
        with vid_col:            
            # READ AS BYTES: This is the most stable way to feed processed video to the browser
            try:
                with open(st.session_state['processed_video_path'], 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes)
            except Exception as e:
                st.error("Error loading video. Ensure the 'avc1' codec is used in the backend.")
            
            st.markdown("**SESSION NOTES**")
            st.markdown(f"<span style='color:gray; font-size:14px;'>Exercise: {squat_type}</span>", unsafe_allow_html=True)
            feedback = stats.get("feedback", "")
            if feedback:
                st.markdown(f"""
                    <div style='background:#1A1A1A; border-left: 4px solid #00FFFF; 
                                padding:12px; border-radius:6px; margin-top:10px;'>
                        <p style='color:#00FFFF; margin:0; font-size:13px; 
                                font-weight:bold;'>OVERALL VERDICT</p>
                        <p style='color:white; margin:4px 0 0; font-size:14px;'>
                            {feedback}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        with graph_col:
            st.markdown("**MY PROGRESS**")
            df = pd.DataFrame({
                "Date": ["Oct 7", "Oct 12", "Oct 19", "Oct 22", "Today"],
                "Score": [25, 45, 95, 88, stats['score']]
            })
            fig = px.line(df, x="Date", y="Score", markers=True)
            # ... (Your fig layout code)
            st.plotly_chart(fig, use_container_width=True)

    # --- SHOW PLACEHOLDER IF NOT DONE YET ---
    else:
        st.info("Awaiting video processing...")
        st.markdown("""
            <div style='height: 435px; display: flex; align-items: center; justify-content: center; border: 2px dashed #333; border-radius: 10px; color: #555;'>
                Processed video and metrics will appear here once analysis is complete.
            </div>
        """, unsafe_allow_html=True)