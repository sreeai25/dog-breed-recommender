st.title("🐶 Dog Breed Recommender")

# --------------------
# Welcome message + name input
# --------------------
st.markdown("**Welcome to Dog breed recommender.**")
st.markdown("I am **D**, your friendly assistant to help you choose your furry friend.")

# session state for name and start
if "name" not in st.session_state:
    st.session_state.name = ""
if "started" not in st.session_state:
    st.session_state.started = False

# Name input (allow pressing Enter to start)
name_input = st.text_input("What is your name?", value=st.session_state.name, key="name_input")
st.session_state.name = name_input.strip()

# Show Start/Exit buttons only if name entered
if st.session_state.name:
    st.success(f"Thank you, {st.session_state.name}, for choosing me today. Let's start! "
               "Please rate the following traits you prefer in your dog based on your personality and lifestyle. "
               "I will be happy to help you with the top three recommended breeds. Click **Start** once ready.")

    cols = st.columns([1,1])
    with cols[0]:
        if st.button("Start") or st.session_state.name:  # Enter key triggers text_input, then start
            st.session_state.started = True
            st.experimental_rerun()
    with cols[1]:
        if st.button("Exit"):
            st.success(f"Thank you, {st.session_state.name}! Come back any time 🐾")
            st.stop()
else:
    st.info("Please enter your name to continue.")

# Stop if not started
if not st.session_state.started:
    st.stop()
