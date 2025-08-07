from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Expanded Career Data (100 Roles with Skills, Interests, etc.)
CAREER_DATA = {
    'Software Developer': {
        'skills': ['programming', 'python', 'java', 'javascript', 'problem solving', 'algorithms', 'debugging'],
        'interests': ['technology', 'coding', 'innovation', 'logical thinking'],
        'salary_range': '$60,000 - $120,000',
        'growth_rate': 'High (22%)',
        'description': 'Design, develop, and maintain software applications and systems.'
    },
    'Data Scientist': {
        'skills': ['python', 'statistics', 'machine learning', 'sql', 'data analysis', 'visualization'],
        'interests': ['mathematics', 'research', 'analytics', 'problem solving'],
        'salary_range': '$70,000 - $140,000',
        'growth_rate': 'Very High (35%)',
        'description': 'Analyze complex data to help organizations make informed decisions.'
    },
    'IAS Officer': {
        'skills': ['leadership', 'administration', 'policy analysis', 'decision making'],
        'interests': ['governance', 'public service', 'leadership'],
        'salary_range': '$15,000 - $40,000 (India)',
        'growth_rate': 'Medium',
        'description': 'Work in administrative services to implement government policies at various levels.'
    },
    'Mechanical Engineer': {
        'skills': ['CAD', 'mechanical design', 'thermodynamics', 'manufacturing', 'problem solving'],
        'interests': ['machines', 'engineering', 'design', 'physics'],
        'salary_range': '$60,000 - $110,000',
        'growth_rate': 'Moderate',
        'description': 'Design, analyze, and manufacture mechanical systems and devices.'
    },
    'ISRO Scientist': {
        'skills': ['aerospace', 'programming', 'physics', 'mathematics', 'research'],
        'interests': ['space', 'research', 'technology'],
        'salary_range': '$20,000 - $40,000 (India)',
        'growth_rate': 'High',
        'description': 'Contribute to India’s space research and satellite missions.'
    },
    'Indian Army Officer': {
        'skills': ['leadership', 'strategy', 'fitness', 'discipline'],
        'interests': ['service', 'discipline', 'national pride'],
        'salary_range': '$15,000 - $35,000 (India)',
        'growth_rate': 'Stable',
        'description': 'Lead and manage operations in the Indian Armed Forces.'
    },
    'Marine Biologist': {
        'skills': ['biology', 'scuba diving', 'data analysis', 'observation'],
        'interests': ['marine life', 'research', 'environment'],
        'salary_range': '$45,000 - $90,000',
        'growth_rate': 'Moderate',
        'description': 'Study ocean life and ecosystems, often through field research.'
    },
    'Game Developer': {
        'skills': ['game engines', 'c++', 'unity', 'graphics', 'animation'],
        'interests': ['gaming', 'design', 'programming'],
        'salary_range': '$50,000 - $100,000',
        'growth_rate': 'High',
        'description': 'Create and develop interactive video games and gaming systems.'
    },
    'AI Researcher': {
        'skills': ['machine learning', 'deep learning', 'python', 'research'],
        'interests': ['AI', 'innovation', 'problem solving'],
        'salary_range': '$90,000 - $150,000',
        'growth_rate': 'Very High',
        'description': 'Conduct advanced research in artificial intelligence and machine learning.'
    },
    'Civil Services Officer': {
        'skills': ['law', 'governance', 'general studies', 'policy'],
        'interests': ['public service', 'administration', 'nation building'],
        'salary_range': '$15,000 - $40,000 (India)',
        'growth_rate': 'Moderate',
        'description': 'Serve the country in key administrative roles across ministries.'
    },
}

# Automatically generate placeholder roles up to 100
for i in range(len(CAREER_DATA)+1, 101):
    CAREER_DATA[f'Role {i}'] = {
        'skills': [f'skill_{i}_1', f'skill_{i}_2'],
        'interests': [f'interest_{i}_1', f'interest_{i}_2'],
        'salary_range': '$50,000 - $100,000',
        'growth_rate': 'Moderate',
        'description': f'Placeholder description for Role {i}.'
    }

class CareerRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.career_vectors = None
        self.setup_model()

    def setup_model(self):
        career_texts = []
        for career, data in CAREER_DATA.items():
            text = ' '.join(data['skills'] + data['interests'])
            career_texts.append(text)

        self.career_vectors = self.vectorizer.fit_transform(career_texts)

    def recommend_careers(self, user_skills, user_interests, top_n=5):
        user_text = ' '.join(user_skills + user_interests)
        user_vector = self.vectorizer.transform([user_text])
        similarities = cosine_similarity(user_vector, self.career_vectors)[0]

        career_names = list(CAREER_DATA.keys())
        recommendations = []
        for i in range(len(similarities)):
            recommendations.append({
                'career': career_names[i],
                'similarity_score': round(similarities[i] * 100, 2),
                'data': CAREER_DATA[career_names[i]]
            })
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return recommendations[:top_n]

recommender = CareerRecommender()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        skills = [s.strip().lower() for s in data.get('skills', '').split(',') if s.strip()]
        interests = [i.strip().lower() for i in data.get('interests', '').split(',') if i.strip()]

        if not skills and not interests:
            return jsonify({'error': 'Please provide at least some skills or interests'}), 400

        recommendations = recommender.recommend_careers(skills, interests)
        return jsonify({'recommendations': recommendations})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Contact form submission - Name: {name}, Email: {email}, Message: {message}")
        return render_template('contact.html', success=True)
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)