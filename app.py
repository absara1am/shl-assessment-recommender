import streamlit as st
import pandas as pd
import requests  # To call the FastAPI endpoint
import logging
import os

# --- Configuration ---
# URL of your running FastAPI application
# If running locally, it might be http://localhost:8000 or http://127.0.0.1:8000
# If deployed, use the deployed API URL. Use environment variable for flexibility.
API_BASE_URL = os.getenv("RECOMMENDER_API_URL", "http://127.0.0.1:8000")
RECOMMEND_ENDPOINT = f"{API_BASE_URL}/recommend"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Streamlit App ---
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("📚 SHL Assessment Recommender")
st.markdown(
    "Enter a job description, role requirements, or desired skills to get assessment recommendations."
)

# --- Input Area ---
query = st.text_area(
    "Enter your query (or job description):",
    height=150,
    placeholder="e.g., 'Java developer with good communication skills needed for client-facing role, max 60 minute assessment time'",
)

# --- Recommendation Button and Logic ---
if st.button("🔍 Get Recommendations"):
    if query:
        logging.info(f"Streamlit App: Sending query to API: '{query[:100]}...'")
        payload = {"text": query, "top_k": 10}  # Request top 10 recommendations

        try:
            # Make POST request to the FastAPI endpoint
            with st.spinner("Fetching recommendations..."):
                response = requests.post(
                    RECOMMEND_ENDPOINT, json=payload, timeout=60
                )  # Increased timeout
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            api_result = response.json()
            results = api_result.get("results", [])
            logging.info(f"Streamlit App: Received {len(results)} results from API.")

            if results:
                st.write("### Recommended Assessments:")
                # Convert list of dicts to DataFrame for display
                try:
                    df = pd.DataFrame(results)

                    # --- Data Formatting for Display ---
                    # Select and order columns as required by assignment
                    display_columns = {
                        "name": "Assessment Name",
                        "test_type": "Test Type",
                        "duration_minutes": "Duration (min)",
                        "remote_testing": "Remote Testing",
                        "adaptive_support": "Adaptive Support",
                        "url": "Link",  # Will be handled by st.column_config.LinkColumn
                    }

                    # Filter DataFrame to only include desired columns, handling missing ones
                    cols_to_display = [
                        col for col in display_columns.keys() if col in df.columns
                    ]
                    df_display = df[cols_to_display].rename(columns=display_columns)

                    # Format duration to integer or 'N/A'
                    if "Duration (min)" in df_display.columns:
                        df_display["Duration (min)"] = df_display[
                            "Duration (min)"
                        ].apply(lambda x: int(x) if pd.notna(x) else "N/A")

                    # Use st.dataframe for better interactivity and link handling
                    st.dataframe(
                        df_display,
                        column_config={
                            "Link": st.column_config.LinkColumn(
                                "Link",
                                display_text="View Details 🔗",  # Text shown in the link cell
                            )
                        },
                        hide_index=True,
                        use_container_width=True,
                    )

                except Exception as e:
                    logging.error(
                        f"Streamlit App: Error processing results into DataFrame: {e}"
                    )
                    st.error("An error occurred while displaying the recommendations.")
                    # Optionally show raw results for debugging if needed:
                    # st.write("Raw Results:", results)

            else:
                st.warning("No matching assessments found based on your query.")

        except requests.exceptions.ConnectionError:
            logging.error(
                f"Streamlit App: Could not connect to API at {RECOMMEND_ENDPOINT}."
            )
            st.error(
                f"Could not connect to the recommendation service. Please ensure it's running at {API_BASE_URL} and accessible."
            )
        except requests.exceptions.Timeout:
            logging.error(f"Streamlit App: Request to API timed out.")
            st.error(
                "The request to the recommendation service timed out. Please try again later."
            )
        except requests.exceptions.RequestException as e:
            logging.error(f"Streamlit App: API request failed: {e}")
            st.error(f"An error occurred while fetching recommendations: {e}")
        except Exception as e:
            logging.exception(f"Streamlit App: An unexpected error occurred: {e}")
            st.error("An unexpected error occurred. Please check the logs.")

    else:
        st.error("Please enter a query.")

st.markdown("---")
st.markdown("Developed based on SHL product catalog.")

# To run this Streamlit app:
# 1. Make sure the FastAPI app (`api.py`) is running.
# 2. Set the RECOMMENDER_API_URL environment variable if the API is not at the default location.
# 3. Run: streamlit run app.py
