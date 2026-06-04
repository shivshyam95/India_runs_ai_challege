import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================================
# 1. DATA PREPROCESSING & SYNTHESIS
# ==========================================================
def create_candidate_profile_string(row):
    # Safe check for columns. Adjust these strings to match Redrob's exact CSV headers.
    name = row.get('name', 'Candidate')
    skills = row.get('skills', '')
    experience = row.get('experience_summary', '')
    education = row.get('education', '')
    return f"Candidate Name: {name}. Skills: {skills}. Experience: {experience}. Education: {education}."

# ==========================================================
# 2. THE AI RETRIEVAL & RE-RANKING ENGINE
# ==========================================================
class AIRecruitingBrain:
    def __init__(self):
        print("[INFO] Initializing State-of-the-Art Embedding Models...")
        # Using a highly-optimized, fast-loading version perfect for your laptop hardware
        self.bi_encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("[INFO] AI Models loaded successfully!")

    def rank_candidates(self, jd_text, df_candidates):
        # Build contextual profiles
        if 'profile_string' not in df_candidates.columns:
            df_candidates['profile_string'] = df_candidates.apply(create_candidate_profile_string, axis=1)
        
        candidates_list = df_candidates['profile_string'].tolist()
        
        # Stage 1: Fast Bi-Encoder Filtering
        print(f"[INFO] Running Stage 1 semantic filtering for {len(candidates_list)} candidates...")
        jd_embedding = self.bi_encoder.encode([jd_text], normalize_embeddings=True)
        profile_embeddings = self.bi_encoder.encode(candidates_list, normalize_embeddings=True)
        
        scores = cosine_similarity(jd_embedding, profile_embeddings)[0]
        df_candidates['stage1_score'] = scores
        
        # Select top candidates to advance to Stage 2 Re-ranking
        top_k = min(10, len(df_candidates))
        df_top = df_candidates.nlargest(top_k, 'stage1_score').copy()
        
        # Stage 2: Deep Cross-Encoder Re-ranking
        print(f"[INFO] Running Stage 2 deep cross-encoder re-ranking on top matches...")
        pairs = [[jd_text, profile] for profile in df_top['profile_string'].tolist()]
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Convert raw models scores into clean 0-100% match percentages
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        df_top['match_percentage'] = [round(sigmoid(score) * 100, 2) for score in cross_scores]
        
        # Generate final clean sorted leaderboard
        leaderboard = df_top.sort_values(by='match_percentage', ascending=False)
        return leaderboard[['name', 'skills', 'match_percentage']]

# ==========================================================
# 3. PIPELINE RUNNER
# ==========================================================
if __name__ == "__main__":
    # Define what type of candidate you are searching for
    target_job_description = (
        "Looking for a Full Stack Software Engineer proficient in Python, React, and PostgreSQL. "
        "Experience with building scalable web applications and REST APIs is required."
    )
    
    # Filename configuration
    data_file = "candidates.csv"
    
    if os.path.exists(data_file):
        print(f"[INFO] Found dataset file: '{data_file}'. Processing data...")
        df_candidates = pd.read_csv(data_file)
    else:
        print(f"[WARNING] '{data_file}' not found in folder. Synthesizing testing dataset...")
        mock_data = {
            'name': ['Amit Sharma', 'Priya Patel', 'Rohan Das', 'Sneha Reddy'],
            'skills': ['Python, Django, PostgreSQL, React', 'Java, Spring Boot, MySQL', 'HTML, CSS, JavaScript, React', 'Python, Flask, AWS, Docker'],
            'experience_summary': ['3 years developing full stack web apps.', '5 years backend engineering.', '1 year frontend development.', '4 years cloud and backend automation.'],
            'education': ['B.Tech Computer Science', 'M.Tech IT', 'BCA', 'B.Tech IT']
        }
        df_candidates = pd.DataFrame(mock_data)
        df_candidates.to_csv(data_file, index=False)
        print(f"[INFO] Created a placeholder '{data_file}' file with sample candidates.")

    # Initialize and execute pipeline
    brain = AIRecruitingBrain()
    ranked_results = brain.rank_candidates(target_job_description, df_candidates)
    
    # Print results out to terminal
    print("\n🏆 FINAL RANKED CANDIDATE LEADERBOARD 🏆")
    print(ranked_results.to_string(index=False))
    
    # Save the output CSV file required for submission
    output_filename = "ranked_candidates_output.csv"
    ranked_results.to_csv(output_filename, index=False)
    print(f"\n[SUCCESS] Leaderboard exported perfectly to '{output_filename}'!")
