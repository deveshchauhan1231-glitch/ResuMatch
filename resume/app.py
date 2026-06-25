import streamlit as st
import pandas as pd
import fitz
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="ResuMatch", layout="wide")
st.title("ResuMatch")
st.caption("Drop your resume. Paste the JD. See how you stack up.")
st.divider()


# ─────────────────────────────────────────────
#  CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_dataset():
    df = pd.read_csv("india_job_market_tech_skills.csv")
    skills_dataset = [skill.strip() for sublist in df['Skills_Required'] for skill in sublist.split(",")]
    edu_dataset    = [edu.strip()   for sublist in df['Education_Required'] for edu in sublist.split("/")]
    roles          = df['Job_Title']
    role_skill = (
        df.groupby('Job_Title')['Skills_Required']
          .apply(lambda x: list(dict.fromkeys(
              skill.strip() for item in x for skill in str(item).split(',')
          )))
    )
    return skills_dataset, edu_dataset, roles, role_skill


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def build_role_embeddings(_model, _role_skill):
    role_dict = {}
    for i in range(len(_role_skill)):
        role_dict[_role_skill.index[i]] = _model.encode(
            _role_skill.iloc[i], convert_to_tensor=True
        ).mean(dim=0)
    return role_dict


# ─────────────────────────────────────────────
#  CORE FUNCTIONS  (logic untouched)
# ─────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)


def extract_entities(text, nlp, skills_dataset, roles, edu_dataset):
    info    = nlp(text)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("SKILLS",    [nlp.make_doc(s) for s in skills_dataset])
    matcher.add("ROLE",      [nlp.make_doc(r) for r in roles])
    matcher.add("EDUCATION", [nlp.make_doc(e) for e in edu_dataset])

    skills_found = set()
    roles_found  = set()
    edu_found    = set()

    for match_id, start, end in matcher(info):
        label     = nlp.vocab.strings[match_id]
        matched   = info[start:end].text
        if label == "SKILLS":      skills_found.add(matched)
        elif label == "ROLE":      roles_found.add(matched)
        elif label == "EDUCATION": edu_found.add(matched)

    return skills_found, roles_found, edu_found


def compare(r, j, model):
    r_emb = model.encode(list(r), convert_to_tensor=True).mean(dim=0)
    j_emb = model.encode(list(j), convert_to_tensor=True).mean(dim=0)
    return util.cos_sim(r_emb, j_emb).item()


def find_missing_skills(r, j, model, threshold):
    missing     = []
    resume_embs = model.encode(list(r), convert_to_tensor=True)
    jd_embs     = model.encode(list(j), convert_to_tensor=True)
    for i, jd_skill in enumerate(list(j)):
        if util.cos_sim(jd_embs[i], resume_embs)[0].max().item() < threshold:
            missing.append(jd_skill)
    return missing


def find_similarity(role_dict, resume_skills, model, threshold):
    resume_emb = model.encode(list(resume_skills), convert_to_tensor=True).mean(dim=0)
    return {
        key: util.cos_sim(role_dict[key], resume_emb).item()
        for key in role_dict
        if util.cos_sim(role_dict[key], resume_emb).item() > threshold
    }


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
with st.spinner("Loading models..."):
    skills_dataset, edu_dataset, roles, role_skill = load_dataset()
    model     = load_model()
    nlp       = load_nlp()
    role_dict = build_role_embeddings(model, role_skill)

# ─────────────────────────────────────────────
#  INPUTS
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
with col2:
    jd_text = st.text_area("Paste Job Description", height=220, placeholder="Paste the full JD here...")

run = st.button("Analyse Match", use_container_width=True)

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if run:
    if not uploaded_resume:
        st.warning("Please upload your resume PDF.")
    elif not jd_text.strip():
        st.warning("Please paste a Job Description.")
    else:
        with st.spinner("Parsing resume and JD..."):
            resume_text = extract_text_from_pdf(uploaded_resume)
            skills_found, roles_found, edu_found    = extract_entities(resume_text, nlp, skills_dataset, roles, edu_dataset)
            skills_needed, roles_needed, edu_needed = extract_entities(jd_text,     nlp, skills_dataset, roles, edu_dataset)

        with st.spinner("Running semantic matching..."):
            skill_score = 0
            if skills_found and skills_needed:
                semantic_skill_score = compare(skills_found, skills_needed, model)
                resume_embs = model.encode(list(skills_found), convert_to_tensor=True)
                jd_embs = model.encode(list(skills_needed), convert_to_tensor=True)
                matched_count = sum(
                    1 for jd_emb in jd_embs
                    if util.cos_sim(jd_emb, resume_embs)[0].max().item() >= 0.75
                )
                coverage_score = matched_count / len(skills_needed)
                skill_score = (0.5 * coverage_score) + (0.5 * semantic_skill_score)
            role_score  = compare(roles_found,  roles_needed,  model) if roles_found  and roles_needed  else 0
            edu_score   = compare(edu_found,    edu_needed,    model) if edu_found    and edu_needed    else 0
            if(edu_score<0.7):
                edu_score=edu_score*(-0.2)
            else:
                edu_score=edu_score*(0.1)
            if(role_score<0.4):
                role_score=role_score*(-0.2)
            else:
                role_score=role_score*(0.2)
            final  = (0.7 * skill_score) + (role_score) + (edu_score)
            missing        = find_missing_skills(skills_found, skills_needed, model, threshold=0.75)
            career_matches = find_similarity(role_dict, skills_found, model, threshold=0.75)

        st.divider()

        # Score
        pct = round(final * 100, 1)
        st.metric("Overall JD Match Score", f"{pct}%")
        if pct >= 80:
            st.success("Strong Match — you have a decent chance!")
        elif pct >= 40:
            st.warning("Moderate Match — some gaps to address.")
        else:
            st.error("Low Match — consider upskilling before applying.")

        st.divider()

        # Resume vs JD side by side
        st.subheader(" Resume  vs   JD")
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("**Skills — Resume**")
            st.write(", ".join(sorted(skills_found)) if skills_found else "_None detected_")
            st.markdown("**Skills — JD**")
            st.write(", ".join(sorted(skills_needed)) if skills_needed else "_None detected_")

        with r2:
            st.markdown("**Roles — Resume**")
            st.write(", ".join(sorted(roles_found)) if roles_found else "_None detected_")
            st.markdown("**Roles — JD**")
            st.write(", ".join(sorted(roles_needed)) if roles_needed else "_None detected_")

        with r3:
            st.markdown("**Education — Resume**")
            st.write(", ".join(sorted(edu_found)) if edu_found else "_None detected_")
            st.markdown("**Education — JD**")
            st.write(", ".join(sorted(edu_needed)) if edu_needed else "_None detected_")

        st.divider()

        # Missing skills
        st.subheader("Missing Skills")
        if missing:
            st.write(", ".join(sorted(missing)))
        else:
            st.success("No significant skill gaps detected!")

        st.divider()

        # Alternate career paths
        st.subheader("Alternate Career Paths")
        if career_matches:
            sorted_careers = sorted(career_matches.items(), key=lambda x: x[1], reverse=True)[:10]
            for role, score in sorted_careers:
                st.write(f"**{role}** — {round(score * 100)}%")
        else:
            st.caption("No strong alternate role matches found.")
