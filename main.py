import streamlit as st
import langchain_helper as lch

st.set_page_config(layout="wide")

# --- Layout: two main columns ---
col_input, col_output = st.columns([1, 2], gap= "large")  # narrow input, wider output

with col_input:
    st.header("🎬 Transcript Q&A")

    with st.form(key="my_form"):
        youtube_url = st.text_area(
            label="Paste your YouTube URL",
            max_chars=50,
            placeholder="https://youtu.be/..."
        )
        query = st.text_area(
            label="What is your question?",
            max_chars=50,
            placeholder="e.g. Summarize the main points"
        )

        # Spacer before the button
        st.markdown("### ")
        submit_button = st.form_submit_button(label="Submit", use_container_width=True)

with col_output:
    st.header("📝 Answer")
    if submit_button:
        # Placeholder answer area
        st.success(f"Processing question: {query}")
        # You could put your transcript processing logic 
        db = lch.create_vec_db_from_url(youtube_url)
        response = lch.get_response_from_query(db, query)
        st.markdown(response, unsafe_allow_html=True)
