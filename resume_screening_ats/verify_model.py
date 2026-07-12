import os
import sys
import torch

def verify_pipeline():
    print("=== STARTING ATS SYSTEM PIPELINE VERIFICATION ===")
    
    # 1. Test Imports and Environment
    try:
        from data_processor import DataProcessor
        from ml_model import ATSModelEngine
        print("✅ Imports successful.")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
        
    # 2. Test DataProcessor Loading
    try:
        dp = DataProcessor()
        print(f"✅ DataProcessor initialized successfully.")
        print(f"   Database shape: {dp.df.shape}")
        print(f"   Unique Skills extracted: {len(dp.skill_vocabulary)}")
        print(f"   Unique Tools extracted: {len(dp.tool_vocabulary)}")
        print(f"   Unique Soft Skills extracted: {len(dp.soft_skill_vocabulary)}")
        
        # Check clean up
        nas_count = dp.df['highest_education'].isna().sum()
        print(f"   NaN values in highest_education after cleaning: {nas_count}")
    except Exception as e:
        print(f"❌ DataProcessor failed: {e}")
        sys.exit(1)
        
    # 3. Test Model Engine Initialization
    try:
        engine = ATSModelEngine()
        print("✅ ATSModelEngine initialized successfully (Sentence-BERT loaded).")
    except Exception as e:
        print(f"❌ Model Engine initialization failed: {e}")
        sys.exit(1)
        
    # 4. Run PyTorch Training (Fast run for 5 epochs)
    try:
        print("Running quick model training check (5 epochs)...")
        engine.train_fit_model(dp, epochs=5, batch_size=64)
        
        # Verify model saved file
        if os.path.exists(engine.model_save_path):
            print(f"✅ Model successfully trained and weights file saved at: {engine.model_save_path}")
        else:
            print(f"❌ Model saved file not found at {engine.model_save_path}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        sys.exit(1)
        
    # 5. Test Model Reloading
    try:
        engine.load_model()
        print("✅ Model loaded successfully from weights file.")
    except Exception as e:
        print(f"❌ Model reloading failed: {e}")
        sys.exit(1)
        
    # 6. Test Model Evaluation and Gap Analysis
    try:
        # Define a sample JD
        jd_text = """
        Looking for a Senior ML Engineer with 5+ years experience. 
        Skills required: Python, PyTorch, Machine Learning, Deep Learning, SQL.
        Tools: AWS, Docker.
        Highest education: Masters in Data Science or Computer Science.
        """
        jd_profile = dp.parse_job_description(jd_text)
        
        # Select first candidate in database
        sample_candidate = dp.df.iloc[0].to_dict()
        
        # Run prediction
        score, features = engine.predict_fit(sample_candidate, jd_profile, jd_text)
        print(f"✅ Matching evaluation complete.")
        print(f"   Candidate Name: {sample_candidate['candidate_name']}")
        print(f"   Primary Role: {sample_candidate['primary_role']} | Domain: {sample_candidate['primary_domain']}")
        print(f"   Job Requirements Domain: {jd_profile['domain']} | Experience: {jd_profile['years_experience']} yrs")
        print(f"   Predicted Match Fit Score: {score * 100:.2f}%")
        
        # Gap analysis check
        gaps = engine.get_gap_analysis(sample_candidate, jd_profile)
        print(f"✅ Gap analysis successful.")
        print(f"   Missing Tech Skills: {gaps['missing_technical_skills']}")
        print(f"   Missing Tools: {gaps['missing_tools']}")
        print(f"   Actionable Suggestions:")
        for sug in gaps['suggestions'][:3]:
            print(f"      - {sug}")
            
    except Exception as e:
        print(f"❌ Matching evaluation / Gap analysis failed: {e}")
        sys.exit(1)
        
    print("\n🎉 ALL PIPELINE VERIFICATIONS COMPLETED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    verify_pipeline()
