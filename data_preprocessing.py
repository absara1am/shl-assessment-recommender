import pandas as pd
import logging

# --- Configuration ---
# Consider moving mapping and paths to a config file
TEST_TYPE_MAPPING = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}
DEFAULT_INPUT_PATH = "./data/shl_assessments.csv"
DEFAULT_OUTPUT_PATH = "./data/processed_assessments.csv"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Functions ---


def map_test_types(type_codes_str):
    """Maps comma-separated test type codes to human-readable names."""
    if pd.isna(type_codes_str):
        return ""
    codes = str(type_codes_str).split(",")
    readable_types = [
        TEST_TYPE_MAPPING.get(code.strip(), code.strip())
        for code in codes
        if code.strip()
    ]
    return ", ".join(readable_types)


def preprocess_data(input_path, output_path):
    """
    Loads raw assessment data, cleans it, maps test types, creates a
    searchable text field, and saves the processed data.

    Args:
        input_path (str): Path to the raw CSV data file.
        output_path (str): Path to save the processed CSV data file.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    logging.info(
        f"Starting data preprocessing. Input: {input_path}, Output: {output_path}"
    )

    try:
        df = pd.read_csv(input_path)
        logging.info(f"Loaded {len(df)} records from {input_path}")
    except FileNotFoundError:
        logging.error(f"Input file not found at {input_path}. Preprocessing aborted.")
        raise
    except Exception as e:
        logging.error(f"Error loading data from {input_path}: {e}")
        raise

    # --- Data Cleaning and Transformation ---

    # Duration: Ensure it's numeric, coercing errors to NaT (Not a Time -> effectively NaN for numbers)
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    logging.info("Processed 'duration_minutes' column.")

    # Test Type Mapping: Apply mapping using the helper function
    df["test_type_readable"] = df["test_type"].apply(map_test_types)
    logging.info("Processed 'test_type' column into 'test_type_readable'.")

    # Create Search Text: Combine relevant fields for embedding/searching
    # Handle potential NaN values gracefully using fillna('') or similar checks
    df["search_text"] = (
        df["name"].fillna("")
        + ". "
        + df["description"].fillna("")
        + ". "
        + "Job Levels: "
        + df["job_levels"].fillna("")
        + ". "
        + "Test Includes: "
        + df["test_type_readable"].fillna("")
        + ". "  # Use mapped types
        + "Duration: "
        + df["duration_minutes"].apply(
            lambda x: f"{int(x)} minutes" if pd.notna(x) else "Not Specified"
        )
        + ". "
        + "Remote: "
        + df["remote_testing"].fillna("Unknown")
        + ". "  # Fill NaNs for boolean-like fields
        + "Adaptive: "
        + df["adaptive_support"].fillna("Unknown")
    )
    logging.info("Created 'search_text' column.")

    # --- Column Selection ---
    # Select and potentially rename columns for the final output
    processed_df = df[
        [
            "name",
            "url",
            "remote_testing",
            "adaptive_support",
            "duration_minutes",
            "test_type_readable",  # Use the human-readable version
            "search_text",
            # Keep original test type codes if needed for specific filtering later
            "test_type",
            "job_levels",
            "languages",
            "description",  # Keep original description if useful
        ]
    ].rename(
        columns={"test_type_readable": "test_type"}
    )  # Rename for consistency if desired

    logging.info(f"Selected final columns: {list(processed_df.columns)}")

    # --- Save Processed Data ---
    try:
        processed_df.to_csv(output_path, index=False)
        logging.info(
            f"Successfully saved {len(processed_df)} processed records to {output_path}"
        )
    except IOError as e:
        logging.error(f"Failed to save processed data to {output_path}: {e}")
        raise

    return processed_df


# --- Main Execution ---
if __name__ == "__main__":
    try:
        preprocess_data(
            input_path=DEFAULT_INPUT_PATH,
            output_path=DEFAULT_OUTPUT_PATH,
        )
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
