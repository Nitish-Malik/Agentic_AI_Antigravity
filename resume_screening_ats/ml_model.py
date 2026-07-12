import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import re

class FitPredictorNN(nn.Module):
    """Deep learning neural network to predict continuous resume fit score."""
    def __init__(self, input_dim=10):
        super(FitPredictorNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Fit score between 0 and 1
        )
        
    def forward(self, x):
        return self.network(x)

class ATSModelEngine:
    def __init__(self, model_save_path="/home/malik/.gemini/antigravity/scratch/resume_screening_ats/fit_model.pt"):
        self.model_save_path = model_save_path
        # Use a lightweight SentenceTransformer model
        print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        # Set cache folder inside workspace
        os.environ["HF_HOME"] = "/home/malik/.gemini/antigravity/scratch/resume_screening_ats/.cache"
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nn_model = None
        self.input_dim = 10
        
        # Load or initialize NN model
        if os.path.exists(self.model_save_path):
            self.load_model()
        else:
            print("No saved PyTorch model found. A model will need to be trained.")

    def load_model(self):
        """Loads the saved PyTorch model state."""
        self.nn_model = FitPredictorNN(self.input_dim)
        self.nn_model.load_state_dict(torch.load(self.model_save_path, map_location=torch.device('cpu')))
        self.nn_model.eval()
        print("PyTorch model loaded successfully.")

    def save_model(self):
        """Saves the PyTorch model state."""
        if self.nn_model is not None:
            torch.save(self.nn_model.state_dict(), self.model_save_path)
            print(f"PyTorch model saved to {self.model_save_path}")

    @staticmethod
    def get_education_val(edu):
        edu_lower = str(edu).lower()
        if 'llm' in edu_lower:
            return 3
        elif 'masters' in edu_lower or 'mba' in edu_lower:
            return 2
        elif 'bachelors' in edu_lower:
            return 1
        return 0

    def compute_features(self, resume_profile, jd_profile, jd_text=""):
        """
        Computes the 10-dimensional feature vector for a resume-JD pair:
        1. Semantic similarity (SBERT cosine similarity)
        2. Skill match ratio
        3. Tool match ratio
        4. Soft skill match ratio
        5. Experience fit score
        6. Domain match flag
        7. Education match score
        8. Profile completeness
        9. Keyword density score
        10. ATS Flag Match Ratio
        """
        # 1. Semantic Similarity using SBERT
        res_text = (
            str(resume_profile.get('experience_summary', '')) + " " +
            str(resume_profile.get('project_summary', '')) + " " +
            str(resume_profile.get('key_achievements', '')) + " " +
            str(resume_profile.get('technical_skills_raw', ''))
        )
        # Use job description text for embedding matching
        if not jd_text:
            jd_text = (
                "Domain: " + str(jd_profile.get('domain', '')) + ". " +
                "Skills: " + ", ".join(jd_profile.get('technical_skills', [])) + " " +
                ", ".join(jd_profile.get('tools_platforms', []))
            )
        
        res_emb = self.sbert_model.encode(res_text, convert_to_tensor=True)
        jd_emb = self.sbert_model.encode(jd_text, convert_to_tensor=True)
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(res_emb.unsqueeze(0), jd_emb.unsqueeze(0)).item()
        cos_sim = max(0.0, min(1.0, cos_sim))
        
        # 2. Skill Match Ratio
        jd_tech_skills = set([s.lower() for s in jd_profile.get('technical_skills', [])])
        res_tech_skills = set([s.strip().lower() for s in str(resume_profile.get('technical_skills_raw', '')).split(',') if s.strip()])
        skill_ratio = 1.0
        if jd_tech_skills:
            matching_skills = jd_tech_skills.intersection(res_tech_skills)
            skill_ratio = len(matching_skills) / len(jd_tech_skills)
            
        # 3. Tool Match Ratio
        jd_tools = set([t.lower() for t in jd_profile.get('tools_platforms', [])])
        res_tools = set([t.strip().lower() for t in str(resume_profile.get('tools_platforms_raw', '')).split(',') if t.strip()])
        tool_ratio = 1.0
        if jd_tools:
            matching_tools = jd_tools.intersection(res_tools)
            tool_ratio = len(matching_tools) / len(jd_tools)

        # 4. Soft Skill Match Ratio
        jd_soft = set([s.lower() for s in jd_profile.get('soft_skills', [])])
        res_soft = set([s.strip().lower() for s in str(resume_profile.get('soft_skills_raw', '')).split(',') if s.strip()])
        soft_ratio = 1.0
        if jd_soft:
            matching_soft = jd_soft.intersection(res_soft)
            soft_ratio = len(matching_soft) / len(jd_soft)

        # 5. Experience fit score (1.0 if meets or exceeds JD requirements)
        jd_exp = float(jd_profile.get('years_experience', 0))
        res_exp = float(resume_profile.get('years_experience', 0))
        if jd_exp == 0:
            exp_score = 1.0
        else:
            exp_score = 1.0 if res_exp >= jd_exp else res_exp / jd_exp
            
        # 6. Domain match
        domain_match = 1.0 if str(resume_profile.get('primary_domain')).lower() == str(jd_profile.get('domain')).lower() else 0.0
        
        # 7. Education match score
        jd_edu = self.get_education_val(jd_profile.get('highest_education', 'Bachelors'))
        res_edu = self.get_education_val(resume_profile.get('highest_education', 'NA'))
        edu_score = 1.0 if res_edu >= jd_edu else (0.5 if res_edu > 0 else 0.0)
        
        # 8. Profile completeness
        completeness = float(resume_profile.get('profile_completeness_score', 0.5))
        
        # 9. Keyword density score
        density = float(resume_profile.get('keyword_density_score', 0.5))
        
        # 10. Flag Match Ratio (Does candidate match key flags like cloud, ml, management if mentioned in JD)
        jd_lower = jd_text.lower()
        active_flags = []
        candidate_flag_vals = []
        
        flag_keywords = {
            'cloud_experience_flag': ['cloud', 'aws', 'azure', 'gcp', 'devops', 'kubernetes'],
            'ml_experience_flag': ['machine learning', 'deep learning', 'nlp', 'ai', 'data science'],
            'management_experience_flag': ['management', 'manager', 'lead', 'director', 'people management'],
            'agile_scrum_experience_flag': ['agile', 'scrum', 'sprint', 'kanban']
        }
        
        for flag_name, keywords in flag_keywords.items():
            if any(kw in jd_lower for kw in keywords):
                active_flags.append(flag_name)
                candidate_flag_vals.append(resume_profile.get(flag_name, 0))
        
        if active_flags:
            flag_match_ratio = sum(candidate_flag_vals) / len(active_flags)
        else:
            # Default fallback: average of key flags
            fallback_flags = ['cloud_experience_flag', 'ml_experience_flag', 'management_experience_flag', 'agile_scrum_experience_flag']
            flag_match_ratio = sum(resume_profile.get(f, 0) for f in fallback_flags) / len(fallback_flags)
            
        features = [
            cos_sim, skill_ratio, tool_ratio, soft_ratio, exp_score,
            domain_match, edu_score, completeness, density, flag_match_ratio
        ]
        
        return features

    def train_fit_model(self, data_processor, epochs=50, batch_size=32):
        """
        Synthesizes a pairwise dataset of resumes and typical job descriptions,
        computes features, calculates heuristic targets, and trains the PyTorch model.
        """
        print("Generating training data from resumes database...")
        df = data_processor.df
        if df is None or len(df) == 0:
            raise ValueError("DataProcessor has empty or unloaded resume database.")
            
        # Define 5 template job descriptions representing different roles to pair with resumes
        sample_jds = [
            {
                'text': "We are looking for a Senior Data Scientist with 6 years experience. Must have Python, Machine Learning, PyTorch, SQL. Strong background in statistics and data science is required.",
                'profile': {
                    'domain': 'Data Science', 'years_experience': 6,
                    'technical_skills': ['python', 'machine learning', 'pytorch', 'sql'],
                    'tools_platforms': ['aws'], 'soft_skills': ['communication'], 'highest_education': 'Masters'
                }
            },
            {
                'text': "Junior Software Engineer role. Requires 1+ years experience in Software Engineering, Java, C++, git. Computer Science education field is preferred.",
                'profile': {
                    'domain': 'IT', 'years_experience': 1,
                    'technical_skills': ['java', 'c++', 'git'],
                    'tools_platforms': ['jira'], 'soft_skills': ['teamwork'], 'highest_education': 'Bachelors'
                }
            },
            {
                'text': "HR Manager with 8+ years experience. Required skills: talent acquisition, employee relations, recruitment, HR manager. MBA or equivalent human resources degree required.",
                'profile': {
                    'domain': 'HR', 'years_experience': 8,
                    'technical_skills': ['recruitment', 'talent acquisition', 'employee relations'],
                    'tools_platforms': [], 'soft_skills': ['leadership', 'communication'], 'highest_education': 'MBA'
                }
            },
            {
                'text': "Looking for a Cloud Engineer / DevOps Engineer with 4 years experience. Skills: AWS, Docker, Kubernetes, DevOps, cloud experience. Must be agile and team oriented.",
                'profile': {
                    'domain': 'IT', 'years_experience': 4,
                    'technical_skills': ['devops', 'kubernetes', 'docker'],
                    'tools_platforms': ['aws', 'jira'], 'soft_skills': ['teamwork'], 'highest_education': 'Bachelors'
                }
            },
            {
                'text': "Corporate Lawyer with 5 years experience in legal counsel, compliance officer, contracts. LLM or law degree required.",
                'profile': {
                    'domain': 'Legal', 'years_experience': 5,
                    'technical_skills': ['contracts', 'compliance'],
                    'tools_platforms': [], 'soft_skills': ['communication'], 'highest_education': 'LLM'
                }
            }
        ]
        
        X = []
        y = []
        
        # Pair a subset of resumes with the JDs (e.g. 300 random candidates for each of the 5 JDs = 1500 samples)
        # Using a subset keeps training very fast but robust.
        np.random.seed(42)
        sample_indices = np.random.choice(len(df), size=min(400, len(df)), replace=False)
        
        print(f"Computing features for {len(sample_indices) * len(sample_jds)} pairings...")
        for idx in sample_indices:
            resume = df.iloc[idx].to_dict()
            for jd in sample_jds:
                features = self.compute_features(resume, jd['profile'], jd['text'])
                
                # Compute target fit score using our heuristic formula
                # y_val = 0.3 * semantic_sim + 0.25 * skill_match + 0.15 * exp_fit + 0.1 * domain_match + 0.05 * edu + 0.05 * tool_match + 0.05 * soft_match + 0.05 * completeness
                y_val = (
                    0.30 * features[0] + # Cosine Similarity
                    0.25 * features[1] + # Skill Match
                    0.05 * features[2] + # Tool Match
                    0.05 * features[3] + # Soft Skill Match
                    0.15 * features[4] + # Exp Match
                    0.10 * features[5] + # Domain Match
                    0.05 * features[6] + # Edu Match
                    0.05 * features[7]   # Completeness
                )
                
                # Add minor random noise to make the neural network learn robust mappings
                y_val = max(0.0, min(1.0, y_val + np.random.normal(0, 0.02)))
                
                X.append(features)
                y.append(y_val)
                
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        # Train-Test Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Conver to PyTorch Tensors
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Instantiate NN model
        self.nn_model = FitPredictorNN(self.input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.nn_model.parameters(), lr=0.003, weight_decay=1e-4)
        
        print("Training PyTorch Neural Network...")
        self.nn_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.nn_model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
            
            # Validation loss check
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.nn_model.eval()
                with torch.no_grad():
                    val_pred = self.nn_model(torch.tensor(X_val))
                    val_loss = criterion(val_pred, torch.tensor(y_val)).item()
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(X_train):.5f} | Val Loss: {val_loss:.5f}")
                self.nn_model.train()
                
        self.nn_model.eval()
        print("Training completed.")
        
        # Save model
        self.save_model()

    def predict_fit(self, resume_profile, jd_profile, jd_text=""):
        """Predicts the fit score for a single candidate profile and JD."""
        if self.nn_model is None:
            # Fallback to pure heuristic if neural network is not loaded
            features = self.compute_features(resume_profile, jd_profile, jd_text)
            score = (
                0.30 * features[0] + 0.25 * features[1] + 0.05 * features[2] +
                0.05 * features[3] + 0.15 * features[4] + 0.10 * features[5] +
                0.05 * features[6] + 0.05 * features[7]
            )
            return score, features
            
        features = self.compute_features(resume_profile, jd_profile, jd_text)
        features_tensor = torch.tensor([features], dtype=torch.float32)
        
        with torch.no_grad():
            score = self.nn_model(features_tensor).item()
            
        return score, features

    @staticmethod
    def get_gap_analysis(resume_profile, jd_profile):
        """Analyzes what is missing in the resume relative to the JD requirements."""
        jd_skills = set([s.lower() for s in jd_profile.get('technical_skills', [])])
        res_skills = set([s.strip().lower() for s in str(resume_profile.get('technical_skills_raw', '')).split(',') if s.strip()])
        
        jd_tools = set([t.lower() for t in jd_profile.get('tools_platforms', [])])
        res_tools = set([t.strip().lower() for t in str(resume_profile.get('tools_platforms_raw', '')).split(',') if t.strip()])
        
        jd_soft = set([s.lower() for s in jd_profile.get('soft_skills', [])])
        res_soft = set([s.strip().lower() for s in str(resume_profile.get('soft_skills_raw', '')).split(',') if s.strip()])
        
        missing_skills = jd_skills - res_skills
        missing_tools = jd_tools - res_tools
        missing_soft = jd_soft - res_soft
        
        # Years of experience check
        jd_exp = jd_profile.get('years_experience', 0)
        res_exp = resume_profile.get('years_experience', 0)
        exp_gap = 0
        if res_exp < jd_exp:
            exp_gap = jd_exp - res_exp
            
        # Education level check
        jd_edu = jd_profile.get('highest_education', 'Bachelors')
        res_edu = resume_profile.get('highest_education', 'NA')
        
        edu_val_jd = ATSModelEngine.get_education_val(jd_edu)
        edu_val_res = ATSModelEngine.get_education_val(res_edu)
        edu_gap = (edu_val_res < edu_val_jd)
        
        # Domain mismatch
        domain_mismatch = (resume_profile.get('primary_domain', '').lower() != jd_profile.get('domain', '').lower())
        
        # Heuristics suggestions
        suggestions = []
        if missing_skills:
            suggestions.append(f"Add key technical skills: {', '.join([s.title() for s in missing_skills])}")
        if missing_tools:
            suggestions.append(f"List tools and platforms: {', '.join([t.title() for t in missing_tools])}")
        if missing_soft:
            suggestions.append(f"Include soft skills: {', '.join([s.title() for s in missing_soft])}")
        if exp_gap > 0:
            suggestions.append(f"Highlight additional leadership or hands-on experience (Job requests {jd_exp} years, profile has {res_exp} years).")
        if edu_gap:
            suggestions.append(f"Clarify advanced degrees or related certifications (Job requests {jd_edu} level education).")
        if domain_mismatch:
            suggestions.append(f"Tailor profile summary to highlight projects in the {jd_profile.get('domain')} domain.")
            
        return {
            'missing_technical_skills': [s.title() for s in missing_skills],
            'missing_tools': [t.title() for t in missing_tools],
            'missing_soft_skills': [s.title() for s in missing_soft],
            'experience_gap': exp_gap,
            'education_gap': edu_gap,
            'domain_mismatch': domain_mismatch,
            'suggestions': suggestions
        }
