# ------------------------------
# QUIZ LOOP (with Skip)
# ------------------------------
total_traits = len(traits)
if st.session_state.step < total_traits:
    trait = traits[st.session_state.step]
    row = trait_desc[trait_desc['Trait'] == trait].iloc[0]

    st.markdown(f"**Question {st.session_state.step+1} of {total_traits}**")
    st.progress((st.session_state.step)/total_traits)

    st.subheader(trait)
    with st.expander("Trait description and rating scale"):
        st.write(row['Description'])
        st.write(f"1 = {row['Trait_1']} — 5 = {row['Trait_5']}")

    val = st.slider("Importance?", 1, 5, st.session_state.answers.get(trait, 3), key=f"slider_{trait}")

    col_back, col_skip, col_next = st.columns(3)
    with col_back:
        if st.button("⬅ Back", key=f"back_{trait}"):
            st.session_state.step = max(0, st.session_state.step-1)
            st.experimental_rerun()
    with col_skip:
        if st.button("Skip ❌", key=f"skip_{trait}"):
            st.session_state.skipped.add(trait)
            if trait in st.session_state.answers:
                del st.session_state.answers[trait]
            st.session_state.step += 1
            st.experimental_rerun()
    with col_next:
        if st.button("Next ➡", key=f"next_{trait}"):
            st.session_state.answers[trait] = val
            if trait in st.session_state.skipped:
                st.session_state.skipped.remove(trait)
            st.session_state.step += 1
            st.experimental_rerun()

# ------------------------------
# RESULTS (after PDF)
# ------------------------------
st.subheader("Actions")
choice_col1, choice_col2, choice_col3 = st.columns(3)

with choice_col1:
    if st.button("Retake Quiz", key="retake_pdf"):
        st.session_state.started = True
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.skipped = set()
        st.experimental_rerun()
with choice_col2:
    if st.button("Exit", key="exit_pdf"):
        st.success(f"Thank you for using my assistance today. Good luck with your furry friend, {user_name}!")
        st.stop()
with choice_col3:
    if st.button("Explain Results", key="explain_pdf"):
        st.subheader("Selection logic used")
        st.write(
            "Calculates breed scores = sum(breed_trait_value * user_importance). "
            "Only traits you rated (not skipped) are considered. "
            "The breed with the maximum score is recommended."
        )
