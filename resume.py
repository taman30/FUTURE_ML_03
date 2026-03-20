# ---------------------------------------
# Import Libraries
# ---------------------------------------

import streamlit as st
import pandas as pd
import re
import nltk
import pdfplumber

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------
# Custom Styling
# ---------------------------------------

st.markdown("""
<style>

.main-title{
font-size:70px;
font-weight:900;
text-align:center;
color:#2E8B57;
letter-spacing:1px;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:24px;
color:gray;
margin-bottom:30px;
}

.about-box{
background-color:#f8f9fa;
padding:15px;
border-radius:10px;
border-left:5px solid #2E8B57;
font-size:16px;
margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">AI Resume Screening System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Resume Ranking & Skill Gap Analysis</p>', unsafe_allow_html=True)

# ---------------------------------------
# Load Dataset
# ---------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Resume.csv")
    df = df[["Category","Resume_str"]]
    return df

df = load_data()

# ---------------------------------------
# NLP Setup
# ---------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------
# Text Cleaning
# ---------------------------------------

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_resume"] = df["Resume_str"].apply(clean_text)

# ---------------------------------------
# Skill Dictionary
# ---------------------------------------

skills = [

# Programming Languages
"python","java","c","c++","c#","javascript","typescript","r","matlab","go","ruby","php","scala","kotlin","swift",

# Data Science / ML
"machine learning","deep learning","data science","nlp","computer vision",
"predictive modeling","data mining","statistical modeling","feature engineering",

# ML Libraries
"scikit learn","tensorflow","keras","pytorch","xgboost","lightgbm","catboost",

# Data Analysis
"pandas","numpy","scipy","statistics","data analysis","data visualization",

# BI Tools
"tableau","power bi","excel","looker","qlikview",

# Databases
"sql","mysql","postgresql","mongodb","sqlite","oracle","cassandra","redis",

# Big Data
"hadoop","spark","pyspark","hive","kafka",

# Cloud Platforms
"aws","azure","google cloud","gcp","cloud computing",

# DevOps
"docker","kubernetes","jenkins","git","github","gitlab","ci cd",

# Web Development
"html","css","react","angular","node.js","flask","django","fastapi","spring boot",

# Data Engineering
"etl","data pipeline","airflow","databricks",

# Visualization Libraries
"matplotlib","seaborn","plotly","ggplot",

# Other Tools
"linux","bash","shell scripting","jira","agile","scrum"

]

skills = sorted(skills, key=len, reverse=True)

# ---------------------------------------
# Skill Extraction
# ---------------------------------------

def extract_skills(text):

    text = text.lower()
    found = set()

    for skill in skills:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found.add(skill)

    return list(found)

df["skills"] = df["clean_resume"].apply(extract_skills)

# ---------------------------------------
# PDF Resume Extraction
# ---------------------------------------

def extract_text_from_pdf(file):

    text = ""

    try:

        file.seek(0)

        with pdfplumber.open(file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

    except:
        st.error("Unable to read the uploaded PDF.")

    return text

# ---------------------------------------
# Job Role Descriptions
# ---------------------------------------

job_descriptions = {

"Data Scientist": """
Looking for a Data Scientist with strong experience in Python,
Machine Learning, Deep Learning, NLP, SQL, Pandas, NumPy,
TensorFlow, and predictive modeling.
""",

"Machine Learning Engineer": """
Seeking a Machine Learning Engineer with expertise in Python,
TensorFlow, PyTorch, Deep Learning, NLP, and model deployment.
Experience with Docker and Kubernetes is preferred.
""",

"Data Analyst": """
Hiring a Data Analyst with experience in SQL, Excel,
Power BI, Tableau, Python, Pandas, and data visualization.
""",

"Software Developer": """
Looking for a Software Developer skilled in Java, Python,
JavaScript, SQL, Git, and web development frameworks.
""",

"Data Engineer": """
Seeking a Data Engineer experienced in ETL pipelines,
Spark, Hadoop, SQL, Airflow, and cloud platforms such as AWS or Azure.
""",

"Business Intelligence Analyst": """
Looking for a BI Analyst with expertise in Tableau,
Power BI, SQL, Excel, and data visualization.
""",

"AI Engineer": """
Hiring an AI Engineer with strong knowledge in Python,
Deep Learning, NLP, Computer Vision, TensorFlow, and PyTorch.
""",

"Backend Developer": """
Looking for a Backend Developer with experience in
Python, Node.js, Django, Flask, FastAPI, SQL, and REST APIs.
""",

"Frontend Developer": """
Seeking a Frontend Developer skilled in HTML, CSS,
JavaScript, React, Angular, and modern UI frameworks.
""",

"Cloud Engineer": """
Looking for a Cloud Engineer experienced in AWS,
Azure, Google Cloud, Docker, Kubernetes, and cloud infrastructure.
""",

"DevOps Engineer": """
Hiring a DevOps Engineer with expertise in Docker,
Kubernetes, Jenkins, CI/CD pipelines, Git, and Linux.
""",

"Big Data Engineer": """
Seeking a Big Data Engineer with experience in
Hadoop, Spark, Kafka, Hive, and distributed data processing.
""",

"Database Administrator": """
Looking for a DBA skilled in MySQL, PostgreSQL,
Oracle, MongoDB, database optimization, and backup management.
"""
}

# ---------------------------------------
# Sidebar
# ---------------------------------------

st.sidebar.header("Job Role Selection")

selected_role = st.sidebar.selectbox(
"Select Job Role",
list(job_descriptions.keys())
)

job_description = job_descriptions[selected_role]

st.sidebar.write("### Job Description")
st.sidebar.write(job_description)

# ---------------------------------------
# Dashboard Cards
# ---------------------------------------

st.markdown("### 📊 Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="background:#f0f2f6;padding:20px;border-radius:10px;text-align:center;">
    <h3>Total Resumes</h3>
    <h2>{len(df)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background:#f0f2f6;padding:20px;border-radius:10px;text-align:center;">
    <h3>Skill Dictionary</h3>
    <h2>{len(skills)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="background:#f0f2f6;padding:20px;border-radius:10px;text-align:center;">
    <h3>Selected Role</h3>
    <h2>{selected_role}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------
# Resume Upload
# ---------------------------------------

st.header("Upload Your Resume")

uploaded_file = st.file_uploader(
"Upload Resume (PDF)",
type=["pdf"]
)

user_resume = None

if uploaded_file is not None:

    user_resume = extract_text_from_pdf(uploaded_file)

    st.success("Resume uploaded successfully!")

# ---------------------------------------
# TF-IDF Similarity
# ---------------------------------------

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(
df["clean_resume"].tolist() + [job_description]
)

cosine_scores = cosine_similarity(
tfidf_matrix[-1],
tfidf_matrix[:-1]
)

df["tfidf_score"] = cosine_scores.flatten()

# ---------------------------------------
# Skill Matching
# ---------------------------------------

jd_skills = extract_skills(clean_text(job_description))

def skill_overlap(candidate):

    matched = set(candidate).intersection(set(jd_skills))

    return len(matched)/len(jd_skills) if len(jd_skills)>0 else 0

df["skill_score"] = df["skills"].apply(skill_overlap)

# ---------------------------------------
# Final Score
# ---------------------------------------

df["final_score"] = (
0.7 * df["tfidf_score"] +
0.3 * df["skill_score"]
)

ranked = df.sort_values(
by="final_score",
ascending=False
)

ranked.reset_index(drop=True, inplace=True)

# ---------------------------------------
# Candidate Ranking
# ---------------------------------------

st.header("Candidate Ranking")

top_n = st.slider(
"Select number of top candidates",
min_value=5,
max_value=50,
value=10
)

top_candidates = ranked.head(top_n)

st.dataframe(
top_candidates[["Category","final_score"]]
)

st.markdown("---")

# ---------------------------------------
# Score Visualization
# ---------------------------------------

st.header("Candidate Score Visualization")

fig, ax = plt.subplots()

sns.barplot(
x=top_candidates["final_score"],
y=top_candidates["Category"],
errorbar=None,    
ax=ax
)

st.pyplot(fig)

st.markdown("---")

# ---------------------------------------
# Skill Heatmap
# ---------------------------------------

st.header("Skill Coverage Heatmap")

top_heatmap = ranked.head(20)

skill_matrix = []

for s in top_heatmap["skills"]:

    row = []

    for skill in jd_skills:
        row.append(1 if skill in s else 0)

    skill_matrix.append(row)

skill_df = pd.DataFrame(skill_matrix, columns=jd_skills)

fig2, ax2 = plt.subplots(figsize=(10,5))

sns.heatmap(
skill_df,
cmap="YlGnBu",
annot=True,
ax=ax2
)

# Axis labels
ax2.set_xlabel("Required Skills")
ax2.set_ylabel("Candidates")

st.pyplot(fig2)

st.markdown("---")

# ---------------------------------------
# Word Cloud
# ---------------------------------------

st.header("Resume Word Cloud")

text = " ".join(df["clean_resume"])

wordcloud = WordCloud(
width=800,
height=400,
background_color="white"
).generate(text)

fig3, ax3 = plt.subplots()

ax3.imshow(wordcloud)
ax3.axis("off")

st.pyplot(fig3)

st.markdown("---")

# ---------------------------------------
# Top Skills Dashboard
# ---------------------------------------

st.header("Top Skills in Resume Dataset")

all_skills = []

for skill_list in df["skills"]:
    all_skills.extend(skill_list)

skill_counts = pd.Series(all_skills).value_counts().head(10)

fig4, ax4 = plt.subplots()

sns.barplot(
x=skill_counts.values,
y=skill_counts.index,
hue=skill_counts.index,
palette="viridis",
legend=False,
ax=ax4
)
# Axis labels
ax4.set_xlabel("Frequency")
ax4.set_ylabel("Skill")

st.pyplot(fig4)


st.markdown("---")

# ---------------------------------------
# Uploaded Resume Analysis
# ---------------------------------------

if user_resume:

    st.header("Your Resume Analysis")

    cleaned_user_resume = clean_text(user_resume)

    user_skills = extract_skills(cleaned_user_resume)

    st.subheader("Detected Skills")
    st.write(user_skills)

    # ATS Match Score
    user_vector = vectorizer.transform([cleaned_user_resume])
    job_vector = vectorizer.transform([job_description])

    similarity = cosine_similarity(user_vector, job_vector)[0][0]

    skill_match = len(set(user_skills).intersection(set(jd_skills))) / len(jd_skills) if jd_skills else 0

    match_score = (0.7 * similarity) + (0.3 * skill_match)

    match_percentage = round(match_score * 100, 2)

    st.subheader("ATS Match Score")
    st.metric("Resume Match", f"{match_percentage}%")

    st.progress(match_score)

    missing_skills = list(set(jd_skills) - set(user_skills))

    if missing_skills:

        st.subheader("Missing Skills")
        st.warning(missing_skills)

    else:

        st.success("Your resume matches the job requirements!")
