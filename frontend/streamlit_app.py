import streamlit as st
import requests
import io
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -------- CONFIG --------
API_BASE = "https://internify-po1q.onrender.com"  # <-- change if your Flask runs elsewhere

st.set_page_config(page_title="Internify", layout="wide")
st.title("⚡ Internify — Resume → Job Matches")

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "user_name" not in st.session_state:
    st.session_state["user_name"] = None
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None
if "history" not in st.session_state:
    st.session_state["history"] = None

# -------- SIDEBAR: Auth & Nav --------
with st.sidebar:
    st.header("Account")
    if not st.session_state["user_id"]:
        tabs = st.tabs(["Login", "Signup"])
        with tabs[0]:
            st.subheader("Login")
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                # If backend has /login implement it; fallback: show error
                try:
                    r = requests.post(f"{API_BASE}/login", json={"email": login_email, "password": login_password}, timeout=10)
                    if r.status_code == 200:
                        data = r.json()
                        st.session_state["user_id"] = data.get("user_id") or data.get("id")
                        st.session_state["user_name"] = data.get("name")
                        st.success("Logged in")
                    else:
                        try:
                            st.error(r.json().get("error", f"Login failed: {r.status_code}"))
                        except Exception:
                            st.error(f"Login failed: {r.status_code}")
                except Exception as e:
                    st.error(f"Network/login error: {e}")
        with tabs[1]:
            st.subheader("Signup")
            signup_name = st.text_input("Name", key="signup_name")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            if st.button("Signup"):
                try:
                    r = requests.post(f"{API_BASE}/signup", json={"name": signup_name, "email": signup_email, "password": signup_password}, timeout=10)
                    if r.status_code == 200 or r.status_code == 201:
                        st.success("Signup successful — please login")
                    else:
                        try:
                            st.error(r.json().get("error", "Signup failed"))
                        except Exception:
                            st.error("Signup failed")
                except Exception as e:
                    st.error(f"Network/signup error: {e}")
    else:
        st.markdown(f"**{st.session_state['user_name'] or 'User'}** (id: {st.session_state['user_id']})")
        if st.button("Logout"):
            st.session_state["user_id"] = None
            st.session_state["user_name"] = None
            st.success("Logged out")
        st.markdown("---")
        if st.button("Fetch history"):
            try:
                r = requests.get(f"{API_BASE}/matches", params={"user_id": st.session_state["user_id"]}, timeout=10)
                if r.status_code == 200:
                    st.session_state["history"] = r.json()
                    st.success("History loaded")
                else:
                    try:
                        st.error(r.json().get("error", "Could not load history"))
                    except Exception:
                        st.error("Could not load history")
            except Exception as e:
                st.error(f"Network error: {e}")

# -------- MAIN: Upload & Results --------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload resume (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF resume", type=["pdf"])
    # allow user to input user id if not logged in
    if not st.session_state["user_id"]:
        user_id_input = st.text_input("User ID (or signup/login first)", key="ui_user_id")

    run_btn = st.button("Run matching")
    if run_btn:
        if not uploaded_file:
            st.error("Please upload a PDF file.")
        else:
            uid = st.session_state["user_id"] or (user_id_input if 'user_id_input' in locals() else None)
            if not uid:
                st.error("Please login or enter a user_id.")
            else:
                with st.spinner("Uploading and running pipeline..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {"user_id": str(uid)}
                        r = requests.post(f"{API_BASE}/upload_resume", files=files, data=data, timeout=120)
                        if r.status_code in (200, 201):
                            try:
                                resp = r.json()
                            except Exception:
                                st.error("Server returned non-JSON response")
                                resp = {}
                            st.session_state["last_results"] = resp.get("results", [])
                            st.success(resp.get("message", "Processed"))
                        else:
                            st.error(f"Error: {r.status_code} — {r.text}")
                    except Exception as e:
                        st.error(f"Network or server error: {e}")

    if st.session_state.get("last_results"):
        st.subheader("Top matches")
        df = pd.DataFrame(st.session_state["last_results"])
        # Normalize column names if backend returns 'score' instead of 'final_score'
        if "final_score" not in df.columns and "score" in df.columns:
            df.rename(columns={"score": "final_score"}, inplace=True)
        # Ensure numeric score and create percentage column for display
        if "final_score" in df.columns:
            try:
                df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
                df["Score (%)"] = (df["final_score"] * 100).round(1)
            except Exception:
                df["Score (%)"] = None
        # Optional: Skill match percentage from backend
        if "skill_match_pct" in df.columns:
            try:
                df["skill_match_pct"] = pd.to_numeric(df["skill_match_pct"], errors="coerce")
                df["Skill Match (%)"] = df["skill_match_pct"].round(1)
            except Exception:
                df["Skill Match (%)"] = None
        # Normalize link column name and values if present
        for link_col in ["link", "url", "apply_link", "apply_url", "application_link"]:
            if link_col in df.columns:
                if link_col != "link":
                    df.rename(columns={link_col: "link"}, inplace=True)
                break
        if "link" in df.columns:
            def _norm_url(x):
                try:
                    s = str(x).strip()
                    if not s:
                        return None
                    if s.startswith("http://") or s.startswith("https://"):
                        return s
                    return "https://" + s
                except Exception:
                    return None
            df["link"] = df["link"].apply(_norm_url)
        # Rank column
        df_display = df.copy()
        # Preferred column order if columns are present
        preferred = [c for c in ["company", "title", "Score (%)", "Skill Match (%)", "cluster", "link"] if c in df_display.columns]
        df_display = df_display[preferred] if preferred else df_display
        df_display.insert(0, "Rank", range(1, len(df_display) + 1))
        # Configure link as clickable column if present
        column_config = {}
        if "link" in df_display.columns:
            column_config["link"] = st.column_config.LinkColumn("Link")
        st.dataframe(df_display.reset_index(drop=True), use_container_width=True, column_config=column_config)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="matches.csv", mime="text/csv")

with col2:
    st.subheader("Analytics")
    if st.session_state.get("last_results"):
        df = pd.DataFrame(st.session_state["last_results"])
        if "final_score" not in df.columns and "score" in df.columns:
            df.rename(columns={"score": "final_score"}, inplace=True)
        try:
            scores = pd.to_numeric(df["final_score"], errors="coerce").dropna()
            if not scores.empty:
                fig1, ax1 = plt.subplots()
                ax1.hist(scores, bins=10)
                ax1.set_xlabel("Final score")
                ax1.set_ylabel("Count")
                ax1.set_title("Score distribution")
                st.pyplot(fig1)
                st.caption(f"Mean: {scores.mean():.3f} | Max: {scores.max():.3f}")
        except Exception:
            st.info("Scores unavailable for histogram.")

        # Jobs vs Skill Match bar chart
        if "skill_match_pct" in df.columns:
            try:
                dfa = df.copy()
                dfa["skill_match_pct"] = pd.to_numeric(dfa["skill_match_pct"], errors="coerce")
                dfa = dfa.dropna(subset=["skill_match_pct"])
                if not dfa.empty:
                    # Build job labels
                    def _label(row):
                        comp = str(row.get("company", "")).strip()
                        title = str(row.get("title", "")).strip()
                        return f"{comp} — {title}" if comp and title else (title or comp or "Job")
                    labels = dfa.apply(_label, axis=1)
                    vals = dfa["skill_match_pct"].round(1)
                    # Limit to top 10 by skill match
                    top_idx = vals.sort_values(ascending=False).index[:10]
                    labels = labels.loc[top_idx]
                    vals = vals.loc[top_idx]
                    fig3, ax3 = plt.subplots(figsize=(7, 5))
                    ax3.barh(labels, vals, color="#4e79a7")
                    ax3.invert_yaxis()
                    ax3.set_xlabel("Skill Match (%)")
                    ax3.set_title("Jobs vs Skill Match")
                    for i, v in enumerate(vals):
                        ax3.text(v + 0.5, i + 0.05, f"{v:.1f}%", fontsize=8)
                    st.pyplot(fig3)
            except Exception:
                st.info("Skill match data unavailable.")

        if "cluster" in df.columns:
            try:
                cluster_counts = df["cluster"].value_counts()
                fig2, ax2 = plt.subplots()
                ax2.pie(cluster_counts.values, labels=cluster_counts.index.astype(str), autopct="%1.1f%%")
                ax2.set_title("Cluster distribution")
                st.pyplot(fig2)
            except Exception:
                st.info("Cluster distribution unavailable.")
    else:
        st.info("Run matching to see analytics.")

# -------- HISTORY (if available) --------
st.markdown("---")
st.subheader("History")
if st.session_state.get("history"):
    try:
        hist = st.session_state["history"]
        # History can be a list of dicts or simple tuples depending on backend
        if isinstance(hist, list):
            for idx, item in enumerate(hist):
                header = None
                if isinstance(item, dict):
                    title = item.get('top_title') or item.get('title', '')
                    company = item.get('company')
                    if company:
                        header = f"{company} — {title}"
                    else:
                        header = f"{item.get('filename', 'resume')} — top: {title}"
                else:
                    # tuple like (title, score, cluster)
                    try:
                        title, score, cluster = item[0], item[1], item[2]
                        header = f"{title} — score: {score:.3f}, cluster: {cluster}"
                    except Exception:
                        header = f"Item {idx+1}"
                with st.expander(header or f"Item {idx+1}"):
                    st.write(item)
        else:
            st.write(hist)
    except Exception:
        st.write(st.session_state["history"])
else:
    st.write("No history loaded. Use the sidebar 'Fetch history' after logging in.")

# Footer
st.markdown("---")
st.caption("Internify — AI INTERNSHIP FINDER.")
