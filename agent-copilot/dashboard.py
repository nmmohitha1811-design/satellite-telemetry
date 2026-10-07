import streamlit as st
import pandas as pd
import sys
sys.path.append(r"C:\Users\ADMIN\space-copilot\rag-copilot")
from rag_answer import answer_question
from graph_pipeline import build_graph


st.set_page_config(page_title="Satellite Telemetry Copilot", layout="wide")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f4f7fb 0%, #e8eef5 100%);
    color: #1a1a2e;
}
[data-testid="stMetricValue"] {
    color: #1565c0;
}
h1, h2, h3 {
    color: #0d1b2a;
}
.stButton>button {
    background-color: #e63946;
    color: white;
    border: none;
}
[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #d0d7e2;
}
</style>
""", unsafe_allow_html=True)
RESULTS_FILE = r"C:\Users\ADMIN\space-copilot\telemanom\results\full_benchmark_82_channels.csv"


@st.cache_resource
def load_pipeline():
    return build_graph()


@st.cache_data
def load_channel_list():
    return pd.read_csv(RESULTS_FILE)


pipeline = load_pipeline()
df = load_channel_list()

st.title("Satellite Telemetry Anomaly Copilot")
st.caption(
    "LSTM anomaly detection on real NASA JPL SMAP/MSL telemetry, grounded in real mission "
    "documents via RAG, orchestrated through a 4-agent LangGraph pipeline."
)

col1, col2, col3 = st.columns(3)
col1.metric("Overall Precision", "87%")
col2.metric("Overall Recall", "83%")
col3.metric("Channels Evaluated", "82")

st.divider()

channel_options = sorted(df[df["true_positives"] > 0]["chan_id"].tolist())
selected = st.selectbox("Select a channel with a caught anomaly to investigate", channel_options)

if st.button("Run Analysis", type="primary"):
    with st.spinner("Running Detector → Retrieval → Report Writer → Guardrail..."):
        result = pipeline.invoke({"channel_id": selected})

    st.subheader(f"Channel {result['channel_id']} ({result['spacecraft']})")
    st.write(result["detection_summary"])

    score = result["best_retrieval_score"]
    grounded = result["is_grounded"]

    score_col, badge_col = st.columns([1, 2])
    score_col.metric("Retrieval Confidence", f"{score:.3f}")
    if grounded:
        badge_col.success("Grounded — sufficient source material found")
    else:
        badge_col.warning("Not grounded — escalated to human review")

    with st.expander("View retrieved source passages"):
        for i, chunk in enumerate(result["retrieved_chunks"]):
            st.markdown(
                f"**Source {i+1}: {chunk['source']} (chunk {chunk['chunk_index']}) "
                f"— similarity {chunk['score']:.3f}**"
            )
            st.text(chunk["text"][:400] + "...")

    st.subheader("Final Report")
    st.write(result["final_report"])
    st.divider()
st.header("Ask a Question")
st.caption("Ask anything about the source documents — grounded answers only, with citations.")

user_question = st.text_input("Your question")

from rag_answer import retrieve, SIMILARITY_THRESHOLD

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            retrieved = retrieve(user_question, top_k=3)
            answer = answer_question(user_question)

        best_score = retrieved[0]["score"]
        score_col, badge_col = st.columns([1, 2])
        score_col.metric("Best Match Score", f"{best_score:.3f}")
        if best_score >= SIMILARITY_THRESHOLD:
            badge_col.success(f"Above threshold ({SIMILARITY_THRESHOLD}) — sources retrieved")
        else:
            badge_col.warning(f"Below threshold ({SIMILARITY_THRESHOLD}) — likely ungrounded")

        with st.expander("View retrieved source passages"):
            for i, chunk in enumerate(retrieved):
                st.markdown(f"**Source {i+1}: {chunk['source']} (chunk {chunk['chunk_index']}) — similarity {chunk['score']:.3f}**")
                st.text(chunk["text"][:300] + "...")

        st.subheader("Answer")
        st.write(answer)