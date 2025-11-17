import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import time

# Import our enhanced modules
from spike_data_loader import NeuralPatternGenerator
from decoding_analysis import OptimizedMultiClassDecoder

# Configure page
st.set_page_config(
    page_title="Neural Code Research Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic Spatial Computing CSS
st.markdown("""
<style>
    /* Import SF Pro Display font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Futuristic animated grid background */
    @keyframes gridPulse {
        0%, 100% { opacity: 0.03; }
        50% { opacity: 0.06; }
    }

    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(10, 132, 255, 0.3), 0 0 40px rgba(10, 132, 255, 0.1); }
        50% { box-shadow: 0 0 30px rgba(10, 132, 255, 0.5), 0 0 60px rgba(10, 132, 255, 0.2); }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    @keyframes rotateDNA {
        from { transform: rotateY(0deg); }
        to { transform: rotateY(360deg); }
    }

    @keyframes rungPulse {
        0%, 100% {
            opacity: 0.4;
            box-shadow: 0 0 8px rgba(10, 132, 255, 0.4);
        }
        50% {
            opacity: 0.8;
            box-shadow: 0 0 16px rgba(10, 132, 255, 0.8);
        }
    }

    /* DNA Helix Container */
    .dna-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 3rem 0;
        perspective: 1200px;
        animation: fadeIn 1.5s ease-out;
        min-height: 300px;
    }

    .dna-helix {
        width: 120px;
        height: 280px;
        position: relative;
        animation: rotateDNA 12s linear infinite;
        transform-style: preserve-3d;
    }

    /* DNA Strand Spheres */
    .dna-sphere {
        position: absolute;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #64d2ff, #0a84ff);
        box-shadow:
            0 0 12px rgba(10, 132, 255, 0.8),
            0 0 24px rgba(10, 132, 255, 0.4),
            inset 0 0 8px rgba(100, 210, 255, 0.6);
        transform-style: preserve-3d;
    }

    /* DNA Connecting Rungs */
    .dna-rung {
        position: absolute;
        height: 2px;
        background: linear-gradient(90deg,
            rgba(10, 132, 255, 0.2),
            rgba(64, 210, 255, 0.8),
            rgba(10, 132, 255, 0.2));
        transform-origin: center;
        transform-style: preserve-3d;
        animation: rungPulse 2s ease-in-out infinite;
    }

    /* Global dark theme with subtle sci-fi grid */
    .stApp {
        background:
            linear-gradient(0deg, rgba(10, 132, 255, 0.01) 1px, transparent 1px),
            linear-gradient(90deg, rgba(10, 132, 255, 0.01) 1px, transparent 1px),
            #0a0a0a;
        background-size: 50px 50px, 50px 50px, 100% 100%;
        background-position: 0 0, 0 0, center;
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', system-ui, sans-serif;
    }

    /* Main header with holographic effect */
    .main-header {
        font-size: 4.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #0a84ff 0%, #64d2ff 50%, #0a84ff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s linear infinite, fadeInUp 1s ease-out;
        filter: drop-shadow(0 0 20px rgba(10, 132, 255, 0.5));
        position: relative;
        z-index: 2;
    }

    .subtitle {
        font-size: 1.3rem;
        color: #a8b5c7;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: -0.01em;
        text-shadow: 0 0 10px rgba(10, 132, 255, 0.3);
        position: relative;
        z-index: 2;
        animation: fadeInUp 1s ease-out 0.3s backwards;
    }

    /* Frosted glass panels with holographic borders */
    .step-box {
        background: rgba(15, 20, 30, 0.7);
        backdrop-filter: blur(40px);
        -webkit-backdrop-filter: blur(40px);
        border: 1px solid transparent;
        background-image:
            linear-gradient(rgba(15, 20, 30, 0.7), rgba(15, 20, 30, 0.7)),
            linear-gradient(135deg, rgba(10, 132, 255, 0.6), rgba(100, 210, 255, 0.4), rgba(10, 132, 255, 0.6));
        background-origin: border-box;
        background-clip: padding-box, border-box;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(100, 210, 255, 0.2),
            0 0 40px rgba(10, 132, 255, 0.15);
        color: #f5f5f7;
        position: relative;
        overflow: hidden;
        z-index: 2;
        transition: all 0.3s ease;
        animation: fadeInUp 1s ease-out 0.9s backwards;
    }

    .step-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent,
            rgba(10, 132, 255, 0.05),
            transparent
        );
        animation: shimmer 3s linear infinite;
    }

    .step-box:hover {
        box-shadow:
            0 12px 48px rgba(0, 0, 0, 0.8),
            inset 0 1px 0 rgba(100, 210, 255, 0.3),
            0 0 60px rgba(10, 132, 255, 0.3);
        transform: translateY(-4px);
    }

    .step-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #0a84ff, #64d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        filter: drop-shadow(0 0 10px rgba(10, 132, 255, 0.5));
        position: relative;
        z-index: 1;
    }

    /* Content boxes with animated glow */
    .metric-box {
        background: rgba(15, 25, 35, 0.6);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(10, 132, 255, 0.4);
        margin: 1.5rem 0;
        box-shadow:
            0 4px 24px rgba(10, 132, 255, 0.2),
            inset 0 1px 0 rgba(10, 132, 255, 0.1);
        color: #f5f5f7;
        position: relative;
        z-index: 2;
        animation: glow 3s ease-in-out infinite;
    }

    .warning-box {
        background: rgba(40, 25, 15, 0.6);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 159, 10, 0.5);
        margin: 1.5rem 0;
        color: #ff9f0a;
        box-shadow:
            0 4px 24px rgba(255, 159, 10, 0.15),
            inset 0 1px 0 rgba(255, 159, 10, 0.1),
            0 0 30px rgba(255, 159, 10, 0.1);
        position: relative;
        z-index: 2;
    }

    .success-box {
        background: rgba(15, 40, 25, 0.6);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(50, 215, 75, 0.5);
        margin: 1.5rem 0;
        color: #32d74b;
        box-shadow:
            0 4px 24px rgba(50, 215, 75, 0.15),
            inset 0 1px 0 rgba(50, 215, 75, 0.1),
            0 0 30px rgba(50, 215, 75, 0.1);
        position: relative;
        z-index: 2;
    }

    .info-box {
        background: rgba(15, 25, 35, 0.6);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(10, 132, 255, 0.4);
        margin: 1.5rem 0;
        color: #f5f5f7;
        box-shadow:
            0 4px 24px rgba(10, 132, 255, 0.15),
            inset 0 1px 0 rgba(10, 132, 255, 0.1),
            0 0 30px rgba(10, 132, 255, 0.1);
        position: relative;
        z-index: 2;
        animation: fadeInUp 1s ease-out 0.6s backwards;
    }

    /* Text styling */
    .stMarkdown p, .stMarkdown li {
        color: #e5e5e7 !important;
        font-weight: 400;
        position: relative;
        z-index: 2;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f5f5f7 !important;
        font-weight: 600;
        letter-spacing: -0.02em;
        text-shadow: 0 0 20px rgba(10, 132, 255, 0.3);
        position: relative;
        z-index: 2;
    }

    /* Button styling with holographic effect */
    .stButton > button {
        background: linear-gradient(135deg, rgba(10, 132, 255, 0.9), rgba(100, 210, 255, 0.7));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: #ffffff;
        border: 1px solid rgba(100, 210, 255, 0.5);
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow:
            0 4px 16px rgba(10, 132, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        letter-spacing: -0.01em;
        position: relative;
        overflow: hidden;
        z-index: 2;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(10, 132, 255, 1), rgba(100, 210, 255, 0.9));
        box-shadow:
            0 8px 32px rgba(10, 132, 255, 0.6),
            0 0 40px rgba(10, 132, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 20, 0.9);
        backdrop-filter: blur(40px);
        -webkit-backdrop-filter: blur(40px);
        border-right: 1px solid rgba(10, 132, 255, 0.2);
        box-shadow: 2px 0 20px rgba(10, 132, 255, 0.1);
    }

    /* Input fields with glow */
    .stSelectbox, .stSlider {
        color: #f5f5f7;
    }

    .stSelectbox > div > div {
        background: rgba(15, 25, 35, 0.6);
        border: 1px solid rgba(10, 132, 255, 0.3);
        box-shadow: 0 0 10px rgba(10, 132, 255, 0.1);
    }

    /* Tabs with holographic effect */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 20, 30, 0.6);
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid rgba(10, 132, 255, 0.2);
        position: relative;
        z-index: 2;
    }

    .stTabs [data-baseweb="tab"] {
        color: #a8b5c7;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(10, 132, 255, 0.4), rgba(100, 210, 255, 0.2));
        color: #64d2ff;
        box-shadow: 0 0 20px rgba(10, 132, 255, 0.3);
        border: 1px solid rgba(10, 132, 255, 0.5);
    }

    /* Expander with glow */
    .streamlit-expanderHeader {
        background: rgba(15, 25, 35, 0.6);
        border: 1px solid rgba(10, 132, 255, 0.3);
        border-radius: 12px;
        color: #a8b5c7;
        transition: all 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        border-color: rgba(10, 132, 255, 0.6);
        box-shadow: 0 0 20px rgba(10, 132, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'decoder' not in st.session_state:
        st.session_state.decoder = None
    if 'decoder_trained' not in st.session_state:
        st.session_state.decoder_trained = False
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'user_mode' not in st.session_state:
        st.session_state.user_mode = 'getting_started'

def show_header():
    """Show main header and description"""
    st.markdown('<div class="main-header">Neural Code Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Neural Spike Data Analysis Tool - that uses a Random Forest machine learning model to predict neurological conditions (Normal, Parkinsonian, Epileptic) from synthetic neural spike train data based on features like firing rates and spike timing patterns</div>', unsafe_allow_html=True)

    # Animated DNA Helix - Realistic smooth double helix
    dna_html = '<div class="dna-container"><div class="dna-helix">'

    # Create smooth helical structure with many points
    import math
    num_segments = 40  # Number of base pairs along the helix
    helix_radius = 40  # Radius of the helix
    helix_height = 280  # Total height
    turns = 2.5  # Number of complete turns

    for i in range(num_segments):
        # Calculate position along helix
        t = i / num_segments
        y = t * helix_height
        angle = t * turns * 2 * math.pi

        # Strand 1 position
        x1 = helix_radius * math.cos(angle)
        z1 = helix_radius * math.sin(angle)

        # Strand 2 position (180 degrees offset)
        x2 = helix_radius * math.cos(angle + math.pi)
        z2 = helix_radius * math.sin(angle + math.pi)

        # Create spheres for both strands
        dna_html += f'<div class="dna-sphere" style="left: {60 + x1}px; top: {y}px; transform: translateZ({z1}px);"></div>'
        dna_html += f'<div class="dna-sphere" style="left: {60 + x2}px; top: {y}px; transform: translateZ({z2}px);"></div>'

        # Add connecting rungs every few segments
        if i % 3 == 0:
            rung_width = abs(x2 - x1)
            rung_left = min(60 + x1, 60 + x2)
            # Calculate rotation for the rung in 3D space
            rung_angle = math.atan2(z2 - z1, x2 - x1) * 180 / math.pi
            delay = (i / num_segments) * 2  # Stagger the pulse animation
            dna_html += f'<div class="dna-rung" style="left: {rung_left}px; top: {y}px; width: {rung_width}px; transform: translateZ({(z1 + z2) / 2}px) rotateY({rung_angle}deg); animation-delay: {delay}s;"></div>'

    dna_html += '</div></div>'

    st.markdown(dna_html, unsafe_allow_html=True)

    # Quick explanation
    st.markdown("""
    <div class="info-box">
        <h3>Custom ML Pipeline & Algorithm Development</h3>
        <p><strong>Developed end-to-end machine learning pipeline</strong> featuring custom neural pattern generation algorithms and multi-class classification models to automatically detect:</p>
        <ul>
            <li><strong>Healthy brain patterns</strong> (normal neural activity)</li>
            <li><strong>Parkinsonian patterns</strong> (beta oscillations, irregular firing)</li>
            <li><strong>Epileptiform patterns</strong> (hypersynchrony, bursting activity)</li>
        </ul>
        <p><em>Built from scratch: synthetic data generation, feature engineering pipeline, optimized multi-class decoder, and real-time visualization system.</em></p>
    </div>
    """, unsafe_allow_html=True)

def show_disclaimer():
    """Show research disclaimer"""
    with st.expander("Important: Educational Use Only"):
        st.markdown("""
        **This tool is for research and educational purposes only.**
        - All neural data is computer-generated (synthetic)
        - Not for medical diagnosis or clinical decisions
        - Designed to teach pattern recognition concepts
        """)

def show_getting_started():
    """Show getting started guide"""

    st.markdown("## Getting Started")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="step-box">
            <div class="step-number">1</div>
            <h3>Train the AI Brain</h3>
            <p>First, we need to teach our AI how to recognize different neural patterns. This takes 2-3 minutes and only needs to be done once.</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.decoder_trained:
            if st.button("Train AI to Recognize Patterns", type="primary", use_container_width=True):
                train_classifier()
        else:
            st.markdown("""
            <div class="success-box">
                <strong>AI is trained and ready!</strong><br>
                You can now generate and analyze neural patterns.
            </div>
            """, unsafe_allow_html=True)

            if st.button("Retrain AI", use_container_width=True):
                retrain_classifier()
    
    with col2:
        st.markdown("""
        <div class="step-box">
            <div class="step-number">2</div>
            <h3>Generate & Analyze</h3>
            <p>Once the AI is trained, choose a brain pattern type and watch the AI instantly identify what kind of neural activity it is!</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.decoder_trained:
            if st.button("Go to Pattern Analysis", type="primary", use_container_width=True):
                st.session_state.user_mode = 'analysis'
                st.rerun()
        else:
            st.info("Train the AI first, then this button will become active!")
    
    # Show training results if available
    if st.session_state.training_results:
        show_training_summary()

def show_analysis_mode():
    """Show the main analysis interface"""

    # Mode switcher
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Back to Getting Started", use_container_width=True):
            st.session_state.user_mode = 'getting_started'
            st.rerun()

    st.markdown("## Neural Pattern Analysis")

    # Pattern selection (simplified)
    st.markdown("### 1. Choose a Brain Pattern Type")
    
    pattern_descriptions = {
        "Healthy Rate": {
            "description": "Normal healthy brain activity with rate-based information encoding",
            "expected": "Should be classified as 'Healthy'",
            "color": "#32d74b",
            "bg_color": "#32d74b10"
        },
        "Healthy Temporal": {
            "description": "Normal healthy brain activity with timing-based information encoding",
            "expected": "Should be classified as 'Healthy'",
            "color": "#32d74b",
            "bg_color": "#32d74b10"
        },
        "Parkinsonian": {
            "description": "Parkinson's disease-like patterns with beta oscillations and irregular firing",
            "expected": "Should be classified as 'Parkinsonian'",
            "color": "#ff9f0a",
            "bg_color": "#ff9f0a10"
        },
        "Epileptiform": {
            "description": "Epilepsy-like patterns with hypersynchronous bursting activity",
            "expected": "Should be classified as 'Epileptiform'",
            "color": "#ff453a",
            "bg_color": "#ff453a10"
        },
        "Mixed Pathology": {
            "description": "Complex patterns with multiple pathological features",
            "expected": "Should be classified as 'Mixed Pathology'",
            "color": "#bf5af2",
            "bg_color": "#bf5af210"
        }
    }
    
    # Custom selectbox styling
    pattern_type = st.selectbox(
        "Select Pattern Type:",
        options=list(pattern_descriptions.keys())
    )

    # Show detailed description with color coding
    selected_info = pattern_descriptions[pattern_type]
    st.markdown(f"""
    <div style="background: rgba(28, 28, 30, 0.6);
                backdrop-filter: blur(30px);
                -webkit-backdrop-filter: blur(30px);
                padding: 1.8rem;
                border-radius: 16px;
                border: 1px solid {selected_info['color']}40;
                margin: 1rem 0;
                color: #f5f5f7;
                box-shadow: 0 4px 24px {selected_info['color']}20;">
        <h4 style="margin: 0 0 0.5rem 0; color: {selected_info['color']}; font-weight: 600;">
            {pattern_type}
        </h4>
        <p style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #e5e5e7;">
            {selected_info['description']}
        </p>
        <p style="margin: 0; font-style: italic; color: #a8b5c7;">
            <strong>Expected Result:</strong> {selected_info['expected']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced options (collapsed by default)
    with st.expander("Advanced Options (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            n_neurons = st.slider("Number of Neurons", 15, 40, 20)
            trial_duration = st.slider("Recording Duration (seconds)", 1.0, 3.0, 2.0, 0.5)
        with col2:
            n_stimuli = st.slider("Number of Stimuli", 3, 8, 5)
            base_firing_rate = st.slider("Base Firing Rate (Hz)", 8.0, 20.0, 12.0)
    
    # Set default values if expander is not used
    if 'n_neurons' not in locals():
        n_neurons = 20
        trial_duration = 2.0
        n_stimuli = 5
        base_firing_rate = 12.0
    
    # Generate button
    st.markdown("### 2. Generate Pattern and See AI Analysis")

    if st.button("Generate Neural Pattern & Analyze", type="primary", use_container_width=True):
        generate_and_analyze_pattern_simple(pattern_type, n_neurons, trial_duration, n_stimuli, base_firing_rate)
    
    # Show results if available
    if st.session_state.current_data:
        show_results()

def generate_and_analyze_pattern_simple(pattern_type, n_neurons, trial_duration, n_stimuli, base_firing_rate):
    """Generate and analyze pattern with simplified interface"""
    
    # Preset configurations
    presets = {
       "Healthy Rate": {
        'coding_type': 'rate',
        'oscillatory_power': 0.0,
        'population_synchrony': 0.05,    # Much lower
        'spike_regularity': 0.95,        # Much higher
        'pathological_bursting': 0.0
    },
        
        "Healthy Temporal": {
            'coding_type': 'temporal',
            'oscillatory_power': 0.0,
            'population_synchrony': 0.15,
            'spike_regularity': 0.8,
            'pathological_bursting': 0.0
        },
        "Parkinsonian": {
        'coding_type': 'rate',
        'oscillatory_power': 0.9,        # Much higher
        'population_synchrony': 0.9,     # Much higher
        'spike_regularity': 0.1,         # Much lower
        'pathological_bursting': 0.4
         },
          "Epileptiform": {
        'coding_type': 'temporal',
        'oscillatory_power': 0.2,        # Low (different from Parkinson's)
        'population_synchrony': 1.0,     # Maximum
        'spike_regularity': 0.2,         # Low
        'pathological_bursting': 0.9     # Maximum
    },
        "Mixed Pathology": {
            'coding_type': 'mixed',
            'oscillatory_power': 0.6,
            'population_synchrony': 0.7,
            'spike_regularity': 0.4,
            'pathological_bursting': 0.4
        }
    }
    
    with st.spinner(f"Generating {pattern_type} neural pattern..."):
        # Generate pattern
        generator = NeuralPatternGenerator(n_neurons=n_neurons, trial_duration=trial_duration)
        
        preset = presets[pattern_type]
        spike_data = generator.generate_synthetic_spikes(
            n_stimuli=n_stimuli,
            n_trials_per_stimulus=1,
            base_firing_rate=base_firing_rate,
            **preset
        )
        
        st.session_state.current_data = spike_data

        # Analyze with AI
        if st.session_state.decoder_trained:
            prediction = st.session_state.decoder.predict(spike_data)
            st.session_state.prediction_results = prediction

    st.success("Pattern generated and analyzed!")
    st.rerun()

def show_results():
    """Show analysis results in a clear, user-friendly way with explanations"""

    st.markdown("### AI Analysis Results")
    
    if st.session_state.prediction_results:
        prediction = st.session_state.prediction_results
        
        # Main result - big and clear with color coding
        col1, col2 = st.columns(2)
        
        with col1:
            # Determine color based on prediction
            prediction_info = {
                'Healthy_Rate': {'color': '#32d74b', 'category': 'Healthy'},
                'Healthy_Temporal': {'color': '#32d74b', 'category': 'Healthy'},
                'Parkinsonian': {'color': '#ff9f0a', 'category': 'Pathological'},
                'Epileptiform': {'color': '#ff453a', 'category': 'Pathological'},
                'Mixed_Pathology': {'color': '#bf5af2', 'category': 'Complex'}
            }

            info = prediction_info.get(prediction['predicted_class'], {'color': '#a8b5c7', 'category': 'Unknown'})

            st.markdown(f"""
            <div style="background: rgba(28, 28, 30, 0.6);
                        backdrop-filter: blur(30px);
                        -webkit-backdrop-filter: blur(30px);
                        padding: 2rem;
                        border-radius: 16px;
                        border: 1px solid {info['color']}40;
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 8px 32px {info['color']}30;">
                <h3 style="margin: 0.5rem 0; color: #a8b5c7; font-weight: 400;">AI Detected:</h3>
                <h2 style="margin: 0; color: {info['color']}; font-size: 2.5rem; font-weight: 700; letter-spacing: -0.02em;">
                    {prediction['predicted_class'].replace('_', ' ')}
                </h2>
                <p style="margin: 0.5rem 0; color: #e5e5e7; font-size: 1.1rem;">
                    <strong>Category:</strong> {info['category']} Pattern
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = prediction['confidence']
            if confidence > 0.8:
                conf_color = "#32d74b"
                message = "Very Confident"
            elif confidence > 0.6:
                conf_color = "#ff9f0a"
                message = "Moderately Confident"
            else:
                conf_color = "#ff453a"
                message = "Uncertain"

            st.markdown(f"""
            <div style="background: rgba(28, 28, 30, 0.6);
                        backdrop-filter: blur(30px);
                        -webkit-backdrop-filter: blur(30px);
                        padding: 2rem;
                        border-radius: 16px;
                        border: 1px solid {conf_color}40;
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 8px 32px {conf_color}30;">
                <h3 style="margin: 0.5rem 0; color: #a8b5c7; font-weight: 400;">Confidence Level:</h3>
                <h2 style="margin: 0; color: {conf_color}; font-size: 2.5rem; font-weight: 700; letter-spacing: -0.02em;">
                    {confidence:.1%}
                </h2>
                <p style="margin: 0.5rem 0; color: #e5e5e7; font-size: 1.1rem;">
                    <strong>{message}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed explanation based on the result
        show_result_explanation(prediction)
        
        # Detailed probabilities with better visualization
        st.markdown("### AI Confidence Breakdown")
        st.markdown("*How confident is the AI about each possible pattern type?*")
        
        prob_df = pd.DataFrame(list(prediction['probabilities'].items()),
                              columns=['Pattern Type', 'Probability'])
        prob_df['Pattern Type'] = prob_df['Pattern Type'].str.replace('_', ' ')
        prob_df = prob_df.sort_values('Probability', ascending=False)

        # Create 2-column layout: graph on left, explanation on right
        col1, col2 = st.columns([1, 1])

        with col1:
            # Create Vision Pro style bar chart
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#0a0a0a')

            # Color scheme matching Vision Pro aesthetic
            color_map = {
                'Healthy Rate': '#32d74b',
                'Healthy Temporal': '#32d74b',
                'Parkinsonian': '#ff9f0a',
                'Epileptiform': '#ff453a',
                'Mixed Pathology': '#bf5af2'
            }

            colors = [color_map.get(pattern, '#8e8e93') for pattern in prob_df['Pattern Type']]

            # Create bars with glow effect
            bars = ax.bar(prob_df['Pattern Type'], prob_df['Probability'], color=colors,
                         alpha=0.9, edgecolor=colors, linewidth=2)

            # Add glow effect by drawing multiple transparent layers
            for bar, color in zip(bars, colors):
                bar.set_edgecolor(color)
                bar.set_linewidth(1.5)

            # Add percentage labels with Vision Pro styling
            for bar, prob, color in zip(bars, prob_df['Probability'], colors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.1%}', ha='center', va='bottom', fontweight='600',
                       fontsize=10, color='#f5f5f7', family='SF Pro Display')

            # Highlight the winner with brighter glow
            bars[0].set_linewidth(2.5)
            bars[0].set_edgecolor(colors[0])

            # Vision Pro dark styling
            ax.set_facecolor('#0a0a0a')
            ax.set_ylabel('Probability', fontsize=11, fontweight='600', color='#a8b5c7',
                         family='SF Pro Display')
            ax.set_title('Classification Confidence', fontsize=13, fontweight='600',
                        color='#f5f5f7', family='SF Pro Display', pad=15)
            ax.set_ylim(0, max(prob_df['Probability']) * 1.15)

            # Minimal grid with subtle glow
            ax.grid(True, alpha=0.08, axis='y', color='#0a84ff', linestyle='-', linewidth=0.5)

            # Style axes
            plt.xticks(rotation=35, ha='right', color='#a8b5c7', fontsize=9, family='SF Pro Display')
            plt.yticks(color='#a8b5c7', fontsize=9, family='SF Pro Display')

            # Transparent spines with subtle glow
            for spine in ax.spines.values():
                spine.set_edgecolor('#1c1c1e')
                spine.set_linewidth(1)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            # Explanation panel with glassmorphism
            st.markdown("""
            <div style="background: rgba(15, 25, 35, 0.5); padding: 1.5rem; border-radius: 12px;
                        border: 1px solid rgba(10, 132, 255, 0.2); backdrop-filter: blur(10px);
                        height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <h4 style="color: #0a84ff; margin-top: 0; font-size: 1.1rem; font-weight: 600;">
                    Understanding Classification Confidence
                </h4>
                <div style="color: #a8b5c7; font-size: 0.9rem; line-height: 1.6;">
                    <p style="margin-top: 0.8rem;">
                        <strong style="color: #f5f5f7;">What this shows:</strong> The probability distribution across all possible neural pattern types. The highest bar indicates the AI's prediction.
                    </p>
                    <p>
                        <strong style="color: #f5f5f7;">How to interpret:</strong>
                    </p>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li><strong style="color: #32d74b;">Healthy patterns</strong> - Normal neural activity with regular firing rates</li>
                        <li><strong style="color: #ff9f0a;">Parkinsonian</strong> - Synchronized bursting typical of Parkinson's disease</li>
                        <li><strong style="color: #ff453a;">Epileptiform</strong> - Hyperactive patterns seen in epilepsy</li>
                        <li><strong style="color: #bf5af2;">Mixed Pathology</strong> - Complex patterns with multiple abnormalities</li>
                    </ul>
                    <p style="margin-bottom: 0;">
                        <strong style="color: #f5f5f7;">Confidence level:</strong> Higher percentages indicate stronger certainty in the classification.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show visualizations
    show_visualizations_simple()

def show_result_explanation(prediction):
    """Provide detailed explanation based on the prediction result"""

    predicted_class = prediction['predicted_class']
    confidence = prediction['confidence']
    probabilities = prediction['probabilities']

    # Get the second highest probability for comparison
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    second_choice = sorted_probs[1] if len(sorted_probs) > 1 else None

    st.markdown("### What Does This Result Mean?")
    
    # Generate explanation based on the specific case
    explanation = ""

    if predicted_class in ['Healthy_Rate', 'Healthy_Temporal']:
        if confidence > 0.8:
            explanation = f"""
            **Excellent! Clear Healthy Pattern Detected**

            The AI is very confident this shows **normal, healthy brain activity**. This means:
            - Regular, organized neural firing patterns
            - Appropriate synchronization levels  
            - No signs of pathological activity
            - Typical of healthy brain function
            
            **Why so confident?** The neural features clearly match healthy patterns with minimal ambiguity.
            """
        else:
            explanation = f"""
            **Likely Healthy, But Some Mixed Features**
            
            The AI thinks this is probably healthy, but detected some **ambiguous characteristics**:
            - Mostly normal firing patterns
            - Some features that could suggest mild irregularities
            - Still within healthy range, but not textbook-perfect

            **This is realistic!** Even healthy brains show natural variation.
            """

    elif predicted_class == 'Parkinsonian':
        if confidence > 0.8:
            explanation = f"""
            **Clear Parkinson's-like Pattern Detected**
            
            The AI found strong evidence of **Parkinsonian neural signatures**:
            - Beta oscillations (13-30 Hz excess activity)
            - Irregular, disrupted firing patterns
            - Increased neural synchronization
            - Reduced movement-related activity modulation

            **High confidence** suggests textbook Parkinson's-like features.
            """
        else:
            second_prob = second_choice[1] if second_choice else 0
            explanation = f"""
            **Parkinson's-like, But With Mixed Features**
            
            The AI detected Parkinsonian characteristics, but also sees features of **{second_choice[0].replace('_', ' ') if second_choice else 'other patterns'}** ({second_prob:.1%} probability).
            
            **This suggests:**
            - Some beta oscillations present
            - Mixed pathological features
            - Could be early-stage or complex presentation
            - Realistic for real-world cases where symptoms overlap

            **Clinical relevance:** Many patients show mixed features, especially in early stages.
            """

    elif predicted_class == 'Epileptiform':
        if confidence > 0.8:
            explanation = f"""
            **Clear Epilepsy-like Pattern Detected**
            
            The AI identified strong **epileptiform characteristics**:
            - Hypersynchronous neural activity
            - Burst firing patterns
            - Sharp, spike-like transients
            - Abnormal population dynamics

            **High confidence** indicates classic seizure-like neural signatures.
            """
        else:
            second_prob = second_choice[1] if second_choice else 0
            explanation = f"""
            **Epilepsy-like With Some Uncertainty**
            
            Strong epileptiform features detected, but the AI also sees **{second_choice[0].replace('_', ' ') if second_choice else 'other patterns'}** characteristics ({second_prob:.1%}).
            
            **This could mean:**
            - Interictal activity (between seizures)
            - Mixed seizure types
            - Transition periods
            - Complex epilepsy presentation

            **Real-world relevance:** Epilepsy patterns can be highly variable.
            """

    elif predicted_class == 'Mixed_Pathology':
        # This is the most complex case - need detailed explanation
        top_other_patterns = [item for item in sorted_probs[:3] if item[0] != 'Mixed_Pathology']

        if confidence > 0.7:
            explanation = f"""
            **Complex Mixed Pathological Pattern**
            
            The AI detected a **combination of multiple pathological features** rather than one clear disease pattern.
            
            **What this means:**
            - Features of both Parkinson's AND epilepsy-like activity
            - Multiple neural systems affected
            - Complex, real-world pathological presentation
            - Could represent comorbid conditions or disease progression
            
            **Top other possibilities:** {', '.join([f"{p[0].replace('_', ' ')} ({p[1]:.1%})" for p in top_other_patterns[:2]])}

            **Clinical significance:** Many neurological patients have mixed presentations!
            """
        else:
            explanation = f"""
            **Borderline Mixed Pattern - AI is Uncertain**
            
            The AI is having trouble choosing between **Mixed Pathology** and **{second_choice[0].replace('_', ' ') if second_choice else 'other patterns'}** ({second_choice[1]:.1%} vs {confidence:.1%}).
            
            **Why this happens:**
            - Pattern sits on the boundary between categories
            - Subtle or early pathological changes
            - Natural variation in neural activity
            - Realistic complexity of real brain data
            
            **This is actually good!** It shows the AI recognizes when patterns don't fit neat categories.
            
            **Try this:** Generate the same pattern again - do you get consistent results?
            """

    # Display the explanation in a glassmorphism box
    st.markdown(f"""
    <div style="background: rgba(28, 28, 30, 0.6);
                backdrop-filter: blur(30px);
                -webkit-backdrop-filter: blur(30px);
                padding: 1.8rem;
                border-radius: 16px;
                border: 1px solid rgba(10, 132, 255, 0.3);
                margin: 1rem 0;
                color: #e5e5e7;
                box-shadow: 0 4px 24px rgba(10, 132, 255, 0.1);">
        {explanation}
    </div>
    """, unsafe_allow_html=True)
    
def show_actionable_suggestions(predicted_class, confidence):
    """Show actionable suggestions for exploring results further"""

    st.markdown("### What Should You Try Next?")

    suggestions = []

    if confidence < 0.7:
        suggestions.append("**Generate the same pattern again** - Is the result consistent?")
        suggestions.append("**Try a more extreme version** - Increase the pathological parameters")

    if predicted_class == 'Mixed_Pathology':
        suggestions.append("**Compare with pure patterns** - Try 'Parkinsonian' and 'Epileptiform' separately")
        suggestions.append("**Look at detailed features** - Check the probability breakdown above")
        suggestions.append("**Experiment with parameters** - Try reducing some pathological settings")

    elif predicted_class in ['Parkinsonian', 'Epileptiform']:
        if confidence > 0.8:
            suggestions.append("**Try the opposite** - Generate a 'Healthy' pattern to see the contrast")
            suggestions.append("**Reduce pathological parameters** - See at what point it becomes 'Mixed'")
        else:
            suggestions.append("**Increase pathological strength** - Try higher values for clearer patterns")

    elif predicted_class in ['Healthy_Rate', 'Healthy_Temporal']:
        suggestions.append("**Try pathological patterns** - Compare with 'Parkinsonian' or 'Epileptiform'")
        suggestions.append("**Experiment with mixed patterns** - See how AI detects subtle changes")

    # Always suggest comparison
    suggestions.append("**Compare different preset types** - Build intuition about pattern differences")
    suggestions.append("**Educational insight** - This shows how AI assists medical diagnosis!")

    # Display suggestions in a nice format
    suggestion_text = "\n".join([f"- {suggestion}" for suggestion in suggestions])

    st.markdown(f"""
    <div style="background: rgba(28, 28, 30, 0.6);
                backdrop-filter: blur(30px);
                -webkit-backdrop-filter: blur(30px);
                padding: 1.8rem;
                border-radius: 16px;
                border: 1px solid rgba(191, 90, 242, 0.3);
                margin: 1rem 0;
                color: #e5e5e7;
                box-shadow: 0 4px 24px rgba(191, 90, 242, 0.1);">
        {suggestion_text}
    </div>
    """, unsafe_allow_html=True)

def show_visualizations_simple():
    """Show simplified visualizations in 2x2 grid"""

    if not st.session_state.current_data:
        return

    st.markdown("### Neural Activity Visualization")

    # Top row - 2 graphs
    col1, col2 = st.columns(2)

    with col1:
        show_raster_plot_simple()

    with col2:
        show_population_analysis_simple()

    # Bottom row - 2 more graphs
    col3, col4 = st.columns(2)

    with col3:
        show_firing_rate_dist()

    with col4:
        show_spike_count_dist()

def show_raster_plot_simple():
    """Show Vision Pro style raster plot with glow effects"""

    spike_data = st.session_state.current_data
    trial_spikes = spike_data['spike_trains'][0]

    st.markdown("**Individual neuron spike activity**")

    # Smaller figure with dark background
    fig, ax = plt.subplots(figsize=(5.5, 2.75), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Plot spikes with cyan glow effect
    for neuron_idx, spikes in enumerate(trial_spikes):
        if len(spikes) > 0:
            # Main spike markers with glow
            ax.plot(spikes, [neuron_idx] * len(spikes), '|',
                   markersize=8, color='#0a84ff', alpha=0.9, linewidth=1.5)
            # Outer glow layer
            ax.plot(spikes, [neuron_idx] * len(spikes), '|',
                   markersize=12, color='#0a84ff', alpha=0.15, linewidth=3)

    # Vision Pro styling
    ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='600',
                 color='#a8b5c7', family='SF Pro Display')
    ax.set_ylabel('Neuron ID', fontsize=11, fontweight='600',
                 color='#a8b5c7', family='SF Pro Display')
    ax.set_title('Neural Spike Activity', fontsize=13, fontweight='600',
                color='#f5f5f7', family='SF Pro Display', pad=15)

    # Minimal grid with blue glow
    ax.grid(True, alpha=0.06, color='#0a84ff', linestyle='-', linewidth=0.5)

    # Style ticks
    ax.tick_params(colors='#a8b5c7', labelsize=9)

    # Style spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#1c1c1e')
        spine.set_linewidth(1)

    # Add subtle annotation with glassmorphism
    ax.text(0.02, 0.98, 'Vertical patterns indicate synchronized firing',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#1c1c1e',
                    edgecolor='#0a84ff', alpha=0.4, linewidth=1),
           color='#a8b5c7', family='SF Pro Display', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Explanation
    st.markdown("""
    <div style="background: rgba(15, 25, 35, 0.4); padding: 0.8rem; border-radius: 8px;
                border-left: 2px solid #0a84ff; margin-top: 0.5rem;">
        <small style="color: #a8b5c7;">
        <strong>What this shows:</strong> Each vertical line represents a spike (neural firing event).
        Rows show individual neurons. Vertical alignment indicates synchronized firing across neurons.
        </small>
    </div>
    """, unsafe_allow_html=True)

def show_population_analysis_simple():
    """Show Vision Pro style population analysis with glowing line"""

    spike_data = st.session_state.current_data
    trial_spikes = spike_data['spike_trains'][0]
    trial_duration = 2.0
    bin_size = 0.02

    # Calculate population activity
    time_bins = np.arange(0, trial_duration + bin_size, bin_size)
    pop_activity = np.zeros(len(time_bins) - 1)

    for spikes in trial_spikes:
        if len(spikes) > 0:
            spike_counts, _ = np.histogram(spikes, bins=time_bins)
            pop_activity += spike_counts

    st.markdown("**Aggregate population activity**")

    # Smaller figure with dark background
    fig, ax = plt.subplots(figsize=(5.5, 2.75), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    time_axis = np.linspace(0, trial_duration, len(pop_activity))

    # Create glowing line effect with multiple layers
    # Outer glow
    ax.plot(time_axis, pop_activity, linewidth=8, color='#0a84ff',
           alpha=0.1, solid_capstyle='round')
    # Middle glow
    ax.plot(time_axis, pop_activity, linewidth=4, color='#0a84ff',
           alpha=0.3, solid_capstyle='round')
    # Main line
    ax.plot(time_axis, pop_activity, linewidth=2, color='#0a84ff',
           alpha=0.95, solid_capstyle='round')

    # Subtle fill with gradient effect
    ax.fill_between(time_axis, pop_activity, alpha=0.15, color='#0a84ff')

    # Vision Pro styling
    ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='600',
                 color='#a8b5c7', family='SF Pro Display')
    ax.set_ylabel('Population Spike Count', fontsize=11, fontweight='600',
                 color='#a8b5c7', family='SF Pro Display')
    ax.set_title('Population Activity Over Time', fontsize=13, fontweight='600',
                color='#f5f5f7', family='SF Pro Display', pad=15)

    # Minimal grid with blue glow
    ax.grid(True, alpha=0.06, axis='y', color='#0a84ff', linestyle='-', linewidth=0.5)

    # Style ticks
    ax.tick_params(colors='#a8b5c7', labelsize=9)

    # Style spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#1c1c1e')
        spine.set_linewidth(1)

    # Add interpretation with glassmorphism
    max_activity = np.max(pop_activity)
    mean_activity = np.mean(pop_activity)

    if max_activity > 3 * mean_activity:
        annotation = "High peaks indicate synchronized bursting"
    elif np.std(pop_activity) < mean_activity * 0.5:
        annotation = "Steady activity indicates regular firing"
    else:
        annotation = "Variable activity indicates mixed patterns"

    ax.text(0.98, 0.96, annotation, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#1c1c1e',
                    edgecolor='#0a84ff', alpha=0.4, linewidth=1),
           color='#a8b5c7', family='SF Pro Display', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Explanation
    st.markdown("""
    <div style="background: rgba(15, 25, 35, 0.4); padding: 0.8rem; border-radius: 8px;
                border-left: 2px solid #0a84ff; margin-top: 0.5rem;">
        <small style="color: #a8b5c7;">
        <strong>What this shows:</strong> Total spike activity across all neurons over time.
        Peaks indicate moments of high network activity. Helps identify bursts and synchronization patterns.
        </small>
    </div>
    """, unsafe_allow_html=True)

def show_firing_rate_dist():
    """Show firing rate distribution across neurons"""

    spike_data = st.session_state.current_data
    trial_spikes = spike_data['spike_trains'][0]
    trial_duration = 2.0

    st.markdown("**Firing Rate Distribution**")

    # Calculate firing rates
    firing_rates = [len(spikes) / trial_duration for spikes in trial_spikes]

    # Create figure
    fig, ax = plt.subplots(figsize=(5.5, 2.75), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Create histogram with gradient colors
    n, bins, patches = ax.hist(firing_rates, bins=15, color='#8b5cf6',
                               alpha=0.8, edgecolor='#a78bfa', linewidth=1.5)

    # Add glow effect to bars
    for patch in patches:
        patch.set_edgecolor('#8b5cf6')
        patch.set_linewidth(1)

    # Vision Pro styling
    ax.set_xlabel('Firing Rate (Hz)', fontsize=11, fontweight='600',
                 color='#a8b5c7', family='SF Pro Display')
    ax.set_ylabel('Neuron Count', fontsize=11, fontweight='600',
                 color='#a8b5c7', family='SF Pro Display')
    ax.set_title('Firing Rate Distribution', fontsize=13, fontweight='600',
                color='#f5f5f7', family='SF Pro Display', pad=15)

    # Minimal grid
    ax.grid(True, alpha=0.06, axis='y', color='#8b5cf6', linestyle='-', linewidth=0.5)

    # Style ticks and spines
    ax.tick_params(colors='#a8b5c7', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1c1c1e')
        spine.set_linewidth(1)

    # Add mean line
    mean_rate = np.mean(firing_rates)
    ax.axvline(mean_rate, color='#64d2ff', linestyle='--', linewidth=2,
              alpha=0.8, label=f'Mean: {mean_rate:.1f} Hz')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.3,
             facecolor='#1c1c1e', edgecolor='#0a84ff')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Explanation
    st.markdown("""
    <div style="background: rgba(15, 25, 35, 0.4); padding: 0.8rem; border-radius: 8px;
                border-left: 2px solid #8b5cf6; margin-top: 0.5rem;">
        <small style="color: #a8b5c7;">
        <strong>What this shows:</strong> Distribution of firing rates (spikes per second) across the neural population.
        Mean line shows average activity level. Helps identify if neurons have similar or varied firing rates.
        </small>
    </div>
    """, unsafe_allow_html=True)

def show_spike_count_dist():
    """Show total spike count distribution"""

    spike_data = st.session_state.current_data
    trial_spikes = spike_data['spike_trains'][0]

    st.markdown("**Spike Count per Neuron**")

    # Calculate spike counts
    spike_counts = [len(spikes) for spikes in trial_spikes]

    # Create figure
    fig, ax = plt.subplots(figsize=(5.5, 2.75), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Create bar plot with neuron IDs
    neuron_ids = list(range(len(spike_counts)))
    bars = ax.bar(neuron_ids, spike_counts, color='#32d74b',
                  alpha=0.8, edgecolor='#3ddc84', linewidth=1.5)

    # Add glow to bars
    for bar in bars:
        bar.set_edgecolor('#32d74b')

    # Vision Pro styling
    ax.set_xlabel('Neuron ID', fontsize=11, fontweight='600',
                 color='#a8b5c7', family='SF Pro Display')
    ax.set_ylabel('Spike Count', fontsize=11, fontweight='600',
                 color='#a8b5c7', family='SF Pro Display')
    ax.set_title('Spike Count per Neuron', fontsize=13, fontweight='600',
                color='#f5f5f7', family='SF Pro Display', pad=15)

    # Minimal grid
    ax.grid(True, alpha=0.06, axis='y', color='#32d74b', linestyle='-', linewidth=0.5)

    # Style ticks and spines
    ax.tick_params(colors='#a8b5c7', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1c1c1e')
        spine.set_linewidth(1)

    # Add mean line
    mean_count = np.mean(spike_counts)
    ax.axhline(mean_count, color='#64d2ff', linestyle='--', linewidth=2,
              alpha=0.8, label=f'Mean: {mean_count:.1f}')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.3,
             facecolor='#1c1c1e', edgecolor='#0a84ff')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Explanation
    st.markdown("""
    <div style="background: rgba(15, 25, 35, 0.4); padding: 0.8rem; border-radius: 8px;
                border-left: 2px solid #32d74b; margin-top: 0.5rem;">
        <small style="color: #a8b5c7;">
        <strong>What this shows:</strong> Total number of spikes produced by each neuron during the recording.
        Mean line shows average activity. Bars above mean indicate more active neurons, below mean indicate less active.
        </small>
    </div>
    """, unsafe_allow_html=True)

def train_classifier():
    """Train the classifier with clear progress"""

    with st.spinner("Training AI to recognize neural patterns..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Setting up AI brain...")
        progress_bar.progress(10)
        time.sleep(0.5)

        st.session_state.decoder = OptimizedMultiClassDecoder(random_state=42)

        status_text.text("Generating training examples (150 neural patterns)...")
        progress_bar.progress(30)
        time.sleep(1)

        status_text.text("Teaching AI to recognize patterns...")
        progress_bar.progress(60)

        # Train the model
        results = st.session_state.decoder.train(n_trials_per_class=30, verbose=False)

        progress_bar.progress(90)
        status_text.text("Finalizing training...")
        time.sleep(0.5)

        st.session_state.training_results = results
        st.session_state.decoder_trained = True

        progress_bar.progress(100)
        status_text.text("AI training complete!")

        time.sleep(1)
        st.rerun()

def retrain_classifier():
    """Reset and retrain"""
    st.session_state.decoder = None
    st.session_state.decoder_trained = False
    st.session_state.training_results = None
    st.session_state.prediction_results = None
    st.rerun()

def show_training_summary():
    """Show a simple training summary"""

    if not st.session_state.training_results:
        return

    results = st.session_state.training_results

    st.markdown("### AI Training Complete!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best AI Model", results['best_classifier'])

    with col2:
        st.metric("Accuracy", f"{results['final_accuracy']:.1%}")

    with col3:
        cv_score = results['results'][results['best_classifier']]['cv_mean']
        st.metric("Reliability Score", f"{cv_score:.1%}")

    st.success("Your AI is now ready to analyze neural patterns!")

def main():
    """Main application with clear user flow"""
    
    initialize_session_state()
    
    show_header()
    show_disclaimer()
    
    st.markdown("---")
    
    # Route based on user mode
    if st.session_state.user_mode == 'getting_started' or not st.session_state.decoder_trained:
        show_getting_started()
    else:
        show_analysis_mode()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #a8b5c7; margin-top: 2rem;">
        Neural Code Research Tool | Educational Neural Pattern Analysis<br>
        <em>Developed by Shelly Normatov</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()