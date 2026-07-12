import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from data_processor import DataProcessor
from ml_model import ATSModelEngine

# Set page configurations
st.set_page_config(
    page_title="TalentBridge ATS & Role Matcher",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Slate & Neon Theme)
st.markdown("""
<style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
        color: #e2e8f0;
    }
    
    /* Header styling */
    .title-text {
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Cards and Glassmorphism */
    .card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Score display */
    .score-badge {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin: 15px 0;
        line-height: 1;
    }
    
    .fit-high {
        color: #10b981; /* Neon green */
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    
    .fit-medium {
        color: #f59e0b; /* Amber */
        text-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
    }
    
    .fit-low {
        color: #ef4444; /* Rose */
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }
    
    .status-text {
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Suggestions list styling */
    .suggestion-item {
        background: rgba(239, 68, 68, 0.08);
        border-left: 4px solid #ef4444;
        border-radius: 4px;
        padding: 10px 15px;
        margin-bottom: 10px;
        font-size: 0.95rem;
        color: #f87171;
    }
    
    .success-item {
        background: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10b981;
        border-radius: 4px;
        padding: 10px 15px;
        margin-bottom: 10px;
        font-size: 0.95rem;
        color: #a7f3d0;
    }
    
    /* Input adjustments */
    div[data-baseweb="textarea"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    /* Hover animations */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED RESOURCES -----------------

@st.cache_resource
def get_data_processor():
    """Caches DataProcessor initialization."""
    return DataProcessor()

@st.cache_resource
def get_model_engine():
    """Caches ATSModelEngine initialization."""
    return ATSModelEngine()

# Initialize data and model
try:
    dp = get_data_processor()
    engine = get_model_engine()
except Exception as e:
    st.error(f"Error loading system components: {e}")
    st.stop()

# ----------------- SESSION STATE -----------------
if 'jd_text' not in st.session_state:
    st.session_state['jd_text'] = ""

# ----------------- JD TEMPLATES -----------------
JD_TEMPLATES = {
    "--- Select Template ---": "",
    "Senior Data Scientist": """We are looking for a Senior Data Scientist to lead our talent analytics department.
Required Experience: 6+ years of working experience in AI, machine learning and statistical modeling.
Required Skills: Python, Machine Learning, Deep Learning, PyTorch, SQL, data structures, and algorithms.
Tools & Platforms: AWS, Git, mlflow.
Education: Masters or PhD in Computer Science, Data Science, Mathematics, or Statistics.
Responsibilities include mentoring junior developers and working with stakeholders to design predictive systems.""",
    
    "Full Stack Software Developer": """Looking for a Full Stack Software Developer to design and develop cloud-native enterprise web applications.
Required Experience: 4+ years of professional software development experience.
Technical Skills: JavaScript, React, Node.js, Python, git, data structures, and software architecture.
Tools & Platforms: AWS, Docker, Kubernetes, Jira.
Education: Bachelors in Computer Science, Information Technology, or related fields.
Experience working in Agile/Scrum and coordinating with cross-border/offshore teams is highly desirable.""",
    
    "HR Talent Acquisition Specialist": """TalentBridge Solutions is hiring a Talent Acquisition Specialist to support high-volume hiring operations.
Required Experience: 3+ years in recruitment and talent sourcing.
Required Skills: recruitment, talent acquisition, sourcing, interview coordination, employee relations.
Tools & Platforms: ATS, LinkedIn Recruiter.
Education: Bachelors or MBA in Human Resources, Psychology, or Business Administration.
Must have excellent stakeholder management and mentoring experience.""",
    
    "Corporate Compliance Lawyer": """Senior Corporate Compliance Lawyer / Compliance Officer to manage legal risk and process audits.
Required Experience: 5+ years of legal counsel or compliance experience in enterprise systems.
Required Skills: compliance, corporate law, contract management, legal risk assessment.
Education: LLM or law degree is required.
Strong document writing skills, stakeholder management, and experience in regulatory environments are mandatory."""
}

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8; font-weight: 800;'>TalentBridge ATS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation Menu
menu_option = st.sidebar.radio(
    "Navigation Menu",
    ["Screen & Analyze Resume", "Batch Rank Candidates", "Browse Candidates Database", "Model Diagnostics & Training"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Status:**
🟢 SBERT Model: `Active`
🔴 PyTorch Model: `{}`
""".format("Ready" if engine.nn_model is not None else "Needs Training"))

if engine.nn_model is None:
    st.sidebar.warning("Deep Learning model weights are missing. Go to 'Model Diagnostics & Training' to train the model, or the system will default to rule-based similarity scoring.")

st.sidebar.markdown("""
<div style='position: fixed; bottom: 10px; font-size: 0.8rem; color: #64748b;'>
TalentBridge Solutions Pvt. Ltd.<br>Enterprise Talent Analytics Engine
</div>
""", unsafe_allow_html=True)

# ----------------- MAIN TITLE -----------------
st.markdown("<div class='title-text'>TalentBridge ATS & Role Matcher</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Deep learning and semantic NLP platform for resume matching, screening, and candidate-role alignment analysis</div>", unsafe_allow_html=True)

# ----------------- SCREEN & ANALYZE TAB -----------------
if menu_option == "Screen & Analyze Resume":
    st.markdown("### 🔍 Candidate Screening & Fit Analysis")
    
    # JD and Candidate Columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📝 1. Job Description (JD) Input</div>", unsafe_allow_html=True)
        
        # JD template selector
        selected_template = st.selectbox("Load Job Description Template:", list(JD_TEMPLATES.keys()))
        if selected_template != "--- Select Template ---":
            st.session_state['jd_text'] = JD_TEMPLATES[selected_template]
            
        jd_input = st.text_area(
            "Paste Job Description details here:",
            value=st.session_state['jd_text'],
            height=280,
            placeholder="Type or paste the job description, required skills, and qualification details..."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>👤 2. Candidate Resume Source</div>", unsafe_allow_html=True)
        
        resume_mode = st.radio("Choose Resume Source:", ["Select from Database", "Upload Resume File (PDF/DOCX)", "Paste Raw Resume Text"])
        
        selected_candidate = None
        uploaded_file = None
        pasted_text = ""
        
        if resume_mode == "Select from Database":
            # Search by name or domain
            db_search = st.text_input("Filter Candidates by Role/Name:", "", placeholder="e.g. Data Scientist, Developer, Candidate_10...")
            
            # Filter options
            filtered_df = dp.df
            if db_search:
                filtered_df = dp.df[
                    dp.df['candidate_name'].str.contains(db_search, case=False) |
                    dp.df['primary_role'].str.contains(db_search, case=False) |
                    dp.df['candidate_id'].str.contains(db_search, case=False)
                ]
            
            cand_list = filtered_df[['candidate_id', 'candidate_name', 'primary_role', 'years_experience']].head(100)
            cand_options = [f"{r['candidate_id']} - {r['candidate_name']} ({r['primary_role']}, {r['years_experience']} yrs)" for _, r in cand_list.iterrows()]
            
            if cand_options:
                selected_cand_str = st.selectbox("Select Candidate from Database:", cand_options)
                cand_id = selected_cand_str.split(" - ")[0]
                selected_candidate = dp.df[dp.df['candidate_id'] == cand_id].iloc[0].to_dict()
            else:
                st.info("No candidates found matching the search keyword.")
                
        elif resume_mode == "Upload Resume File (PDF/DOCX)":
            uploaded_file = st.file_uploader("Upload PDF or Word Document Resume:", type=["pdf", "docx"])
            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")
                
        else: # Paste text
            pasted_text = st.text_area("Paste Resume Text here:", height=200, placeholder="Paste the content of the resume here...")
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Screening Trigger Action
    if st.button("🚀 Analyze and Evaluate Resume"):
        if not jd_input.strip():
            st.error("Please enter or select a Job Description first.")
        else:
            # 1. Parse JD
            jd_profile = dp.parse_job_description(jd_input)
            
            # 2. Parse/Load Resume
            resume_profile = None
            
            if resume_mode == "Select from Database":
                if selected_candidate:
                    resume_profile = selected_candidate
                else:
                    st.error("Please select a valid candidate from the database.")
            elif resume_mode == "Upload Resume File (PDF/DOCX)":
                if uploaded_file is not None:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extract text
                    if uploaded_file.name.endswith(".pdf"):
                        extracted_text = dp.extract_text_from_pdf(tmp_path)
                    else: # docx
                        extracted_text = dp.extract_text_from_docx(tmp_path)
                        
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    if not extracted_text.strip():
                        st.error("Could not extract text from the uploaded file. Please ensure it is not scanned/image-only or password-protected.")
                    else:
                        resume_profile = dp.parse_resume_text(extracted_text, file_name=uploaded_file.name)
                else:
                    st.error("Please upload a resume file first.")
            else: # Paste text
                if pasted_text.strip():
                    resume_profile = dp.parse_resume_text(pasted_text, file_name="Pasted_Resume.txt")
                else:
                    st.error("Please paste the resume text first.")

            # 3. Perform Score Evaluation
            if resume_profile:
                st.markdown("---")
                
                # Predict score using SBERT and PyTorch
                with st.spinner("Calculating semantic similarity and running neural network match predictions..."):
                    score, features = engine.predict_fit(resume_profile, jd_profile, jd_input)
                    gap_analysis = engine.get_gap_analysis(resume_profile, jd_profile)
                
                # UI Layout for Results
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card-title'>👑 Candidate: {resume_profile['candidate_name']}</div>", unsafe_allow_html=True)
                    st.markdown(f"**Role:** {resume_profile['primary_role']} | **Domain:** {resume_profile['primary_domain']}")
                    st.markdown(f"**Experience:** {resume_profile['years_experience']} Years")
                    st.markdown(f"**Education:** {resume_profile['highest_education']} in {resume_profile['education_field']}")
                    
                    # Convert decimal score to percentage
                    pct_score = int(score * 100)
                    
                    # Classify fit category
                    if pct_score >= 80:
                        fit_class = "fit-high"
                        fit_status = "Highly Aligned Candidate"
                    elif pct_score >= 50:
                        fit_class = "fit-medium"
                        fit_status = "Partially Aligned Candidate"
                    else:
                        fit_class = "fit-low"
                        fit_status = "Weakly Aligned Candidate"
                        
                    st.markdown(f"<div class='score-badge {fit_class}'>{pct_score}%</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='status-text {fit_class}'>{fit_status}</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Mini metrics list
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<div class='card-title'>📊 Key Structural Metrics</div>", unsafe_allow_html=True)
                    st.markdown(f"🔹 **Keyword Density Score:** `{resume_profile['keyword_density_score']}`")
                    st.markdown(f"🔹 **Profile Completeness:** `{resume_profile['profile_completeness_score']}`")
                    st.markdown(f"🔹 **Resume Length:** `{resume_profile['resume_length_words']} words`")
                    st.markdown("</div>", unsafe_allow_html=True)

                with res_col2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<div class='card-title'>📈 Match Breakdown Analytics</div>", unsafe_allow_html=True)
                    
                    # Interactive Radar Chart of features
                    categories = [
                        'Semantic Similarity', 'Tech Skill Match', 'Tools Match', 'Soft Skills Match',
                        'Experience Match', 'Domain Match', 'Education Match', 'Completeness',
                        'Keyword Density', 'ATS Heuristic Flags'
                    ]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=features,
                        theta=categories,
                        fill='toself',
                        fillcolor='rgba(139, 92, 246, 0.25)',
                        line=dict(color='#8b5cf6', width=2),
                        name='Match Profile'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#334155'),
                            angularaxis=dict(gridcolor='#334155'),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8', size=11),
                        margin=dict(l=70, r=70, t=20, b=20),
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Section for Gap analysis and recommendations
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>🛠️ ATS Gap Analysis & Recommendations</div>", unsafe_allow_html=True)
                
                gap_col1, gap_col2 = st.columns([1, 1])
                
                with gap_col1:
                    st.markdown("#### ✅ Strengths (Found in Profile)")
                    matched_tech = [s for s in jd_profile['technical_skills'] if s in [rs.lower() for rs in str(resume_profile['technical_skills_raw']).split(',')]]
                    matched_tools = [t for t in jd_profile['tools_platforms'] if t in [rt.lower() for rt in str(resume_profile['tools_platforms_raw']).split(',')]]
                    
                    if matched_tech:
                        st.markdown(f"<div class='success-item'>✔️ **Matched Technical Skills:** {', '.join([s.title() for s in matched_tech])}</div>", unsafe_allow_html=True)
                    if matched_tools:
                        st.markdown(f"<div class='success-item'>✔️ **Matched Tools/Platforms:** {', '.join([t.title() for t in matched_tools])}</div>", unsafe_allow_html=True)
                    if float(resume_profile['years_experience']) >= float(jd_profile['years_experience']):
                        st.markdown(f"<div class='success-item'>✔️ **Experience Requirement Met:** Candidate has {resume_profile['years_experience']} years (Required: {jd_profile['years_experience']} years)</div>", unsafe_allow_html=True)
                    if resume_profile['primary_domain'].lower() == jd_profile['domain'].lower():
                        st.markdown(f"<div class='success-item'>✔️ **Domain Alignment:** Both are in the {resume_profile['primary_domain']} domain</div>", unsafe_allow_html=True)
                    
                    if not (matched_tech or matched_tools or float(resume_profile['years_experience']) >= float(jd_profile['years_experience'])):
                        st.info("No major explicit overlaps found in skills or experience.")
                        
                with gap_col2:
                    st.markdown("#### ❌ Gaps & Missing Points")
                    if gap_analysis['suggestions']:
                        for sug in gap_analysis['suggestions']:
                            st.markdown(f"<div class='suggestion-item'>❌ {sug}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='success-item'>✔️ Perfect fit! No missing skills or experience gaps identified.</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

# ----------------- BATCH RANK TAB -----------------
elif menu_option == "Batch Rank Candidates":
    st.markdown("### 📊 Bulk Screening and Candidates Ranking")
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📝 Set Job Description for Ranking</div>", unsafe_allow_html=True)
    
    # Load template selector for batch
    selected_template = st.selectbox("Load Job Description Template:", list(JD_TEMPLATES.keys()), key="batch_template")
    if selected_template != "--- Select Template ---":
        st.session_state['jd_text'] = JD_TEMPLATES[selected_template]
        
    jd_input_batch = st.text_area(
        "Job Description:",
        value=st.session_state['jd_text'],
        height=180,
        key="batch_jd"
    )
    
    # Filtering the candidate pool
    st.markdown("#### Filter Candidate Database Pool before Ranking:")
    f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
    with f_col1:
        domain_filter = st.multiselect("Primary Domain Filter:", list(dp.df['primary_domain'].unique()))
    with f_col2:
        exp_range = st.slider("Experience Range (Years):", 0, 20, (0, 20))
    with f_col3:
        limit_pool = st.number_input("Limit candidate pool size:", min_value=10, max_value=5000, value=200)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🔍 Rank Candidates"):
        if not jd_input_batch.strip():
            st.error("Please enter a valid Job Description.")
        else:
            # 1. Parse JD
            jd_profile = dp.parse_job_description(jd_input_batch)
            
            # 2. Filter candidate pool
            df_pool = dp.df
            if domain_filter:
                df_pool = df_pool[df_pool['primary_domain'].isin(domain_filter)]
            df_pool = df_pool[(df_pool['years_experience'] >= exp_range[0]) & (df_pool['years_experience'] <= exp_range[1])]
            
            # Limit pool size for speed in browser rendering
            df_pool = df_pool.head(limit_pool)
            
            if len(df_pool) == 0:
                st.warning("No candidates in the database match your filters.")
            else:
                progress_bar = st.progress(0)
                status_txt = st.empty()
                
                scores = []
                # Compute scores
                for idx, (_, row) in enumerate(df_pool.iterrows()):
                    resume = row.to_dict()
                    # Run quick prediction (neural network or heuristic fallback)
                    fit_score, _ = engine.predict_fit(resume, jd_profile, jd_input_batch)
                    scores.append(fit_score)
                    
                    # Update progress bar occasionally
                    if idx % 10 == 0:
                        progress_bar.progress(int((idx / len(df_pool)) * 100))
                        status_txt.text(f"Evaluated {idx}/{len(df_pool)} candidates...")
                        
                progress_bar.progress(100)
                status_txt.text("All candidates evaluated successfully!")
                
                # Append score and sort
                df_results = df_pool.copy()
                df_results['fit_score'] = scores
                df_results['fit_percentage'] = (df_results['fit_score'] * 100).astype(int)
                df_results = df_results.sort_values(by='fit_score', ascending=False)
                
                # Display Results
                st.markdown("### 🏆 Ranking Results")
                
                # Plotly Distribution Chart
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>📈 Score Distribution across Evaluated Pool</div>", unsafe_allow_html=True)
                fig = px.histogram(
                    df_results, x="fit_percentage", nbins=20,
                    title="Fit Score Distribution",
                    color_discrete_sequence=['#8b5cf6'],
                    labels={'fit_percentage': 'Fit Score (%)'}
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8'),
                    yaxis=dict(gridcolor='#334155'),
                    xaxis=dict(gridcolor='#334155')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Table of results
                display_cols = [
                    'candidate_id', 'candidate_name', 'primary_domain', 
                    'primary_role', 'years_experience', 'highest_education', 
                    'fit_percentage'
                ]
                
                # Styled Table
                st.markdown("#### Ranked Candidates Table:")
                st.dataframe(
                    df_results[display_cols].reset_index(drop=True),
                    column_config={
                        "candidate_id": "ID",
                        "candidate_name": "Name",
                        "primary_domain": "Domain",
                        "primary_role": "Role",
                        "years_experience": "Experience (Yrs)",
                        "highest_education": "Highest Education",
                        "fit_percentage": st.column_config.ProgressColumn(
                            "Fit Score (%)",
                            help="Evaluation score predicted by the deep learning model",
                            format="%d%%",
                            min_value=0,
                            max_value=100
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )

# ----------------- BROWSE DATABASE TAB -----------------
elif menu_option == "Browse Candidates Database":
    st.markdown("### 📂 Candidates Database Explorer")
    st.markdown("Search, filter, and review profiles inside TalentBridge's parsed candidate database.")
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🔍 Search and Filter Options</div>", unsafe_allow_html=True)
    
    b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
    
    with b_col1:
        search_term = st.text_input("Search (ID, Name, Skills, Role):", "", placeholder="e.g. Python, CND_100, DevOps...")
    with b_col2:
        domain_sel = st.multiselect("Filter by Domain:", list(dp.df['primary_domain'].unique()), key="browse_domain")
    with b_col3:
        role_sel = st.multiselect("Filter by Role:", list(dp.df['primary_role'].unique()), key="browse_role")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Apply filters
    filtered_db = dp.df
    
    if search_term:
        term = search_term.lower()
        filtered_db = filtered_db[
            filtered_db['candidate_name'].str.contains(term, case=False) |
            filtered_db['candidate_id'].str.contains(term, case=False) |
            filtered_db['technical_skills_raw'].str.contains(term, case=False) |
            filtered_db['primary_role'].str.contains(term, case=False)
        ]
        
    if domain_sel:
        filtered_db = filtered_db[filtered_db['primary_domain'].isin(domain_sel)]
        
    if role_sel:
        filtered_db = filtered_db[filtered_db['primary_role'].isin(role_sel)]
        
    st.markdown(f"Found **{len(filtered_db)}** candidates:")
    
    # Display table
    st.dataframe(
        filtered_db[[
            'candidate_id', 'candidate_name', 'primary_domain', 'primary_role', 
            'years_experience', 'highest_education', 'technical_skills_raw'
        ]].reset_index(drop=True),
        column_config={
            "candidate_id": "ID",
            "candidate_name": "Name",
            "primary_domain": "Domain",
            "primary_role": "Role",
            "years_experience": "Experience (Yrs)",
            "highest_education": "Highest Education",
            "technical_skills_raw": "Technical Skills"
        },
        use_container_width=True,
        hide_index=True
    )

# ----------------- DIAGNOSTICS & TRAINING TAB -----------------
elif menu_option == "Model Diagnostics & Training":
    st.markdown("### ⚙️ Deep Learning Model Diagnostics & Training Dashboard")
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🤖 PyTorch Feed-Forward Model Status</div>", unsafe_allow_html=True)
    
    if engine.nn_model is not None:
        st.success("✅ A trained PyTorch model weights file (`fit_model.pt`) is loaded and active.")
        st.write("**Model Architecture:**")
        st.code(str(engine.nn_model))
    else:
        st.warning("⚠️ No PyTorch weights file detected. The application is running on heuristic fallback matching.")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🏋️ Train Model on Candidate Database</div>", unsafe_allow_html=True)
    st.markdown("""
    This will generate candidate-job description training pairs by pairing candidates from the database with 5 distinct reference job descriptions. 
    It then trains a PyTorch deep neural network to predict the fit score.
    """)
    
    train_epochs = st.slider("Training Epochs:", 10, 100, 50)
    train_batch_size = st.select_slider("Batch Size:", options=[16, 32, 64, 128], value=32)
    
    if st.button("🚀 Start PyTorch Model Training"):
        with st.spinner("Preparing dataset, computing SBERT embeddings, and running training loops... This may take up to a minute."):
            try:
                # Redirect standard print to Streamlit logs
                import sys
                from io import StringIO
                
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                engine.train_fit_model(dp, epochs=train_epochs, batch_size=train_batch_size)
                
                sys.stdout = old_stdout
                
                st.success("✅ Deep learning model trained and saved successfully!")
                
                # Show stdout output containing loss metrics
                st.markdown("#### Training Log:")
                st.code(mystdout.getvalue())
                
                # Refresh page state to recognize the loaded model
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during training: {e}")
                
    st.markdown("</div>", unsafe_allow_html=True)
