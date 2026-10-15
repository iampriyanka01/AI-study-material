import streamlit as st
from main import answer_student_query
from menuAssistant import MenuAssistant

assistant = MenuAssistant()

st.set_page_config(page_title=" Restaurant Menu Q&A Assistant", page_icon="🎓")

st.title("🎓 Restaurant Menu Assistant")
st.write("Ask me about Restaurant Menu!")

# Input box with placeholder text
user_query = st.text_input(
    "Enter your question:",
    placeholder="What should I order at Sakura Sushi?"
)

if st.button("Submit"):
    if user_query.strip():
        response = assistant.answer(user_query)

        st.subheader("Response")

        # Graceful error handling
        if isinstance(response, str) and "RESOURCE_EXHAUSTED" in response:
            st.error("⚠️ Sorry, the service has reached its daily quota limit. Please try again later.")
        elif isinstance(response, dict):
            event_text = response.get("response", "No response text available.")
            st.write(event_text)
        else:
            st.write(response)
    else:
        st.warning("Please enter a query before submitting.")




