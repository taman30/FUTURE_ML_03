# 🤖 AI Resume Screening System

### Future Interns Internship — Task 3

---

# 📌 Project Overview

This project was developed as part of the **Future Interns Internship – Task 3**.

Recruiters often receive **hundreds of resumes for a single job role**, making manual screening slow and inefficient.

This project builds an **AI-powered Resume Screening System** that automatically:

* analyzes resume text
* extracts technical skills
* compares resumes with job descriptions
* ranks candidates based on job relevance
* identifies missing skills
* calculates an **ATS-style match score**

The system demonstrates how **Machine Learning and Natural Language Processing (NLP)** can support automated recruitment systems.

---

# 🚀 Key Features

- Resume text preprocessing and cleaning
- Skill extraction using **regex-based NLP techniques**
- Job description matching
- **TF-IDF vectorization** for resume similarity
- **Cosine similarity scoring** for candidate ranking
- Skill overlap analysis
- Final candidate ranking system
- **ATS-style resume match score (%)**
- Skill gap detection for uploaded resumes
- Interactive **Streamlit web dashboard**

---

# ⚙️ Project Workflow

```
Resume Dataset
      ↓
Text Cleaning & Preprocessing
      ↓
Skill Extraction
      ↓
Job Description Selection
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity Calculation
      ↓
Skill Overlap Scoring
      ↓
Final Candidate Ranking
```

### Final Candidate Score Formula

```
Final Score = (0.7 × TF-IDF Similarity) + (0.3 × Skill Match)
```

This weighted scoring system prioritizes:

* overall resume relevance
* important job-related skills

---

# 🧠 Technologies Used

| Technology   | Purpose                                |
| ------------ | -------------------------------------- |
| Python       | Core programming language              |
| Pandas       | Data processing                        |
| NLTK         | Text preprocessing                     |
| Scikit-learn | TF-IDF & cosine similarity             |
| Streamlit    | Interactive web application            |
| Matplotlib   | Data visualization                     |
| Seaborn      | Statistical visualization              |
| WordCloud    | Resume keyword visualization           |
| pdfplumber   | Extract text from uploaded PDF resumes |

---

# 📊 Application Capabilities

The system allows users to:

* Select a **job role**
* Upload a **PDF resume**
* Extract **skills automatically**
* Calculate an **ATS compatibility score**
* Identify **missing skills**
* Visualize **candidate rankings**
* Explore **resume analytics dashboards**

---

# 🖥️ Running the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/taman30/FUTURE_ML_03
```

### 2️⃣ Install required libraries

```
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit application

```
streamlit run resume.py
```

The Streamlit dashboard will open automatically in your browser.

---

# 📂 Project Structure

```
resume.py                      # Streamlit web application
resume_screening_system.ipynb  # Machine learning pipeline notebook
Resume.csv                     # Resume dataset
requirements.txt               # Python dependencies
README.md                      # Project documentation
```

---

# 🔮 Future Improvements

Possible improvements include:

- Use **transformer models (BERT / Sentence-BERT)** for better resume matching
- Implement **Named Entity Recognition (NER)** for advanced skill extraction
- Support **multiple job description comparisons**
- Deploy the system on **cloud platforms**

---

# 🎯 Conclusion

This project demonstrates how **Machine Learning and NLP techniques can automate resume screening and candidate ranking**.

By combining **text similarity analysis and skill-based evaluation**, the system helps recruiters quickly identify the most relevant candidates while reducing manual effort.

This project represents a **simplified AI recruitment platform similar to modern hiring systems**.

---

# 👤 Author

👨‍💻 **SIRIKI TAMAN**

Developed as part of:

🎓 **Future Interns – Machine Learning Task 3**

Project Title:  
🤖 **AI Resume Screening System**



---

#
