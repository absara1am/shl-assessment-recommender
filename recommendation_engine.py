import os
import re
import json
import pandas as pd
import torch
import logging
from typing import List, Dict, Any

# LangChain Imports - Using recommended structure
from langchain.prompts import PromptTemplate

try:
    # Newer langchain structure
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import Runnable
except ImportError:
    # Fallback for older structure (may need adjustments)
    logging.warning(
        "Using fallback LangChain imports. Ensure langchain, langchain-huggingface, "
        "langchain-community, langchain-core are installed and up-to-date."
    )
    from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

    # Output parser might differ in older versions, adjust if needed
    from langchain_core.output_parsers import (
        StrOutputParser,
    )  # Kept core version assuming it's available
    from langchain_core.runnables import Runnable

from sentence_transformers import SentenceTransformer, util, CrossEncoder

# --- Configuration ---
# Consider moving model names, paths, and keywords to a config file or env vars
DATA_PATH = "./data/processed_assessments.csv"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEFAULT_DEVICE = "cpu"  # Or 'cuda' if GPU is available and preferred
DEFAULT_TOP_K = 10
INITIAL_RETRIEVAL_K = 50  # Retrieve more candidates for reranking
SKILL_BOOST_FACTOR = 0.1  # How much to boost score per skill match

# Pre-defined skill keywords (customize or make configurable)
SKILL_KEYWORDS = [
    "python",
    "java",
    "sql",
    "javascript",
    "c++",
    "communication",
    "leadership",
    "management",
    "problem solving",
    "data analysis",
    "project management",
    "clerical",
    "administrative",
    "sales",
    "customer service",
    "financial",
    "collaboration",
    "analytical",
    "critical thinking",
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AssessmentRecommender:
    """
    Recommends assessments based on user queries using semantic search,
    re-ranking, and LLM-extracted constraints.
    """

    def __init__(self, data_path: str = DATA_PATH, device: str = DEFAULT_DEVICE):
        """
        Initializes models, loads data, precomputes embeddings, and sets up the LLM chain.

        Args:
            data_path (str): Path to the **processed** CSV file.
            device (str): Device to run models on ('cpu' or 'cuda').
        """
        logging.info("Initializing AssessmentRecommender...")
        self.device = device
        self.skill_keywords = SKILL_KEYWORDS  # Use configured list

        # Initialize models
        try:
            logging.info(
                f"Loading embedding model: {EMBEDDING_MODEL} on device: {self.device}"
            )
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL, device=self.device
            )
            logging.info(f"Embedding model loaded successfully.")

            logging.info(f"Loading reranker model: {RERANKER_MODEL}")
            self.reranker = CrossEncoder(RERANKER_MODEL)  # Typically runs fine on CPU
            logging.info("Reranker model loaded successfully.")

        except Exception as e:
            logging.error(
                f"Error loading SentenceTransformer or CrossEncoder models: {e}"
            )
            raise

        # Initialize HuggingFace Endpoint LLM
        hf_token = os.getenv("HF_API_KEY")
        if not hf_token:
            logging.error("HF_API_KEY environment variable not set.")
            raise ValueError("HF_API_KEY environment variable not set.")

        try:
            self.llm = HuggingFaceEndpoint(
                repo_id=LLM_REPO_ID,
                temperature=0.1,
                max_new_tokens=250,
                huggingfacehub_api_token=hf_token,
                task="text-generation",  # Explicitly set the task
            )
            logging.info(
                f"HuggingFace LLM endpoint initialized for repo: {LLM_REPO_ID}"
            )
        except Exception as e:
            logging.error(f"Failed to initialize HuggingFaceEndpoint: {e}")
            raise

        # Load and validate data
        self._load_data(data_path)

        # Define skill keywords
        logging.info(f"Using {len(self.skill_keywords)} skill keywords.")

        # Precompute embeddings
        self._precompute_embeddings()

        # --- Define LangChain prompt and chain using LCEL ---
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""Extract JSON with these keys strictly following the format:
             - duration_max: number (integer) or null
             - test_types: list of strings from ["Ability & Aptitude", "Biodata & Situational Judgement", "Competencies", "Development & 360", "Assessment Exercises", "Knowledge & Skills", "Personality & Behavior", "Simulations"] or empty list []
             - skills: list of relevant technical or soft skills mentioned or implied, or empty list []
             - remote_only: boolean (true or false)
             - adaptive_only: boolean (true or false)

             Query: {query}
             JSON:""",
            # Note: Adjusted test_types list to match the mapped values from preprocessing
        )

        # Define the chain: Prompt -> LLM -> String Output Parser
        try:
            self.chain: Runnable = self.prompt | self.llm | StrOutputParser()
            logging.info("LCEL chain for constraint parsing initialized.")
        except Exception as e:
            logging.error(f"Failed to create LCEL chain: {e}")
            raise

        logging.info("AssessmentRecommender initialization complete.")

    def _load_data(self, data_path: str):
        """Loads and validates the processed assessment data."""
        logging.info(f"Loading data from {data_path}...")
        try:
            self.df = pd.read_csv(data_path)
            logging.info(f"Data loaded successfully. Shape: {self.df.shape}")

            # --- Validate required columns ---
            # Ensure columns needed for filtering and searching are present
            required_cols = [
                "name",
                "url",
                "search_text",
                "duration_minutes",
                "test_type",
                "remote_testing",
                "adaptive_support",
            ]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                logging.error(f"Missing required columns in CSV: {missing_cols}")
                raise ValueError(f"Missing required columns in CSV: {missing_cols}")

            # Convert relevant columns to expected types early if not already done
            # (Preprocessing script should handle this, but double-check)
            self.df["duration_minutes"] = pd.to_numeric(
                self.df["duration_minutes"], errors="coerce"
            )
            # Ensure boolean-like fields are strings for consistent filtering
            self.df["remote_testing"] = self.df["remote_testing"].astype(str)
            self.df["adaptive_support"] = self.df["adaptive_support"].astype(str)

        except FileNotFoundError:
            logging.error(f"Error: Data file not found at {data_path}")
            raise
        except ValueError as ve:  # Catch specific validation errors
            logging.error(f"Data validation error: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error loading or processing data from {data_path}: {e}")
            raise

    def _precompute_embeddings(self):
        """Precompute document embeddings for faster search."""
        if "search_text" not in self.df.columns:
            logging.error("Column 'search_text' not found in DataFrame for embedding.")
            raise ValueError("Column 'search_text' not found for embedding.")

        logging.info("Precomputing embeddings for 'search_text'...")
        # Handle potential NaN values in search_text before creating list
        self.texts = self.df["search_text"].fillna("").astype(str).tolist()

        try:
            self.embeddings = self.embedding_model.encode(
                self.texts,
                convert_to_tensor=True,
                show_progress_bar=True,  # Show progress for potentially long task
                device=self.device,
            )
            logging.info(
                f"Embeddings computed for {len(self.texts)} documents. Tensor shape: {self.embeddings.shape}"
            )
        except Exception as e:
            logging.error(f"Error during embedding computation: {e}")
            raise

    def parse_constraints(self, query: str) -> Dict[str, Any]:
        """
        Use LangChain LCEL chain + Mixtral for constraint extraction.
        Falls back to regex parsing if LLM fails.
        """
        logging.info(f"Attempting LLM constraint parsing for query: '{query}'")
        default_constraints = {
            "duration_max": None,
            "test_types": [],
            "skills": [],
            "remote_only": False,
            "adaptive_only": False,
        }
        try:
            # --- Enhanced Logging/Tracing ---
            logging.debug(f"LLM Input Prompt:\n{self.prompt.format(query=query)}")
            result_text = self.chain.invoke({"query": query})

            if not isinstance(result_text, str):
                logging.warning(
                    f"LLM returned non-string type: {type(result_text)}. Converting."
                )
                result_text = str(result_text)

            logging.debug(f"LLM Raw Output:\n---\n{result_text}\n---")
            # --- End Enhanced Logging ---

            parsed = self._safe_json_parse(result_text)

            # Validate parsed keys and types against defaults
            validated_constraints = default_constraints.copy()
            if parsed:  # Only validate if JSON parsing succeeded somewhat
                for key, default_value in default_constraints.items():
                    if key in parsed:
                        # Basic type check (more specific checks could be added)
                        if isinstance(parsed[key], type(default_value)):
                            validated_constraints[key] = parsed[key]
                        # Allow integer duration even if default is None
                        elif key == "duration_max" and isinstance(parsed[key], int):
                            validated_constraints[key] = parsed[key]
                        else:
                            logging.warning(
                                f"Type mismatch for key '{key}' in LLM output. "
                                f"Expected {type(default_value)}, got {type(parsed[key])}. Using default."
                            )
                    # else: key not in parsed, default value remains

                logging.info(
                    f"Successfully parsed constraints via LLM: {validated_constraints}"
                )
                return validated_constraints
            else:
                logging.warning(
                    "LLM output did not contain valid JSON. Falling back to regex."
                )
                return self._fallback_constraint_parsing(query)

        except Exception as e:
            logging.error(f"LLM constraint parsing failed: {e}")
            if "doesn't support task" in str(e):
                logging.error("Hint: Check 'task' parameter in HuggingFaceEndpoint.")
            elif "authorization" in str(e).lower():
                logging.error("Hint: Check HF_API_KEY validity and permissions.")
            # Fallback if LLM fails for any reason
            return self._fallback_constraint_parsing(query)

    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON potentially embedded in LLM output text."""
        logging.debug("Attempting safe JSON parse...")
        # Try to find JSON object boundaries { ... }
        # More robust regex to handle potential leading/trailing text or markdown
        json_match = re.search(r"\{.*\}", text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            try:
                parsed_json = json.loads(json_str)
                logging.debug(f"Successfully parsed JSON: {parsed_json}")
                return parsed_json
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing failed on extracted string: {e}")
                logging.warning(f"Attempted to parse: '{json_str}'")
                return {}  # Return empty dict on failure
        else:
            logging.warning("No JSON object boundaries found in LLM output.")
            return {}

    def _fallback_constraint_parsing(self, text: str) -> Dict[str, Any]:
        """Regex-based fallback parser if LLM fails."""
        logging.warning("Using fallback regex-based constraint parser.")
        constraints = {
            "duration_max": None,
            "test_types": [],
            "skills": [],
            "remote_only": False,
            "adaptive_only": False,
        }
        text_lower = text.lower()

        # Duration parsing (finds number before "min", "minute", "minutes")
        # Looks for patterns like "under 30 minutes", "max 45 min", "less than 60min"
        duration_match = re.search(
            r"(?:under|max|maximum|less than|within)\s*(\d+)\s*min", text_lower
        )
        if duration_match:
            try:
                constraints["duration_max"] = int(duration_match.group(1))
            except ValueError:
                logging.warning(
                    f"Fallback duration parse failed for match: {duration_match.group(1)}"
                )

        # Test type parsing (simple keyword check) - Use mapped types for consistency
        # Match full words/phrases where possible
        possible_types = [
            "Ability & Aptitude",
            "Biodata & Situational Judgement",
            "Competencies",
            "Development & 360",
            "Assessment Exercises",
            "Knowledge & Skills",
            "Personality & Behavior",
            "Simulations",
        ]
        constraints["test_types"] = [
            t
            for t in possible_types
            if re.search(r"\b" + re.escape(t.lower()) + r"\b", text_lower)
        ]

        # Skill matching (using pre-defined list)
        constraints["skills"] = [
            s
            for s in self.skill_keywords
            if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_lower)
        ]

        # Boolean flags parsing (check for keywords)
        if re.search(r"\bremote\b", text_lower):
            constraints["remote_only"] = True
        if re.search(r"\badaptive\b", text_lower):
            constraints["adaptive_only"] = True

        logging.info(f"Fallback parsed constraints: {constraints}")
        return constraints

    def recommend(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Generates top_k assessment recommendations for a given query.

        Args:
            query (str): The user's query for assessment recommendations.
            top_k (int): The maximum number of recommendations to return.

        Returns:
            List[Dict[str, Any]]: A list of recommended assessment dictionaries,
                                   sorted by relevance score. Returns empty list on failure.
        """
        logging.info(f"Generating recommendations for query: '{query}' (top_k={top_k})")

        # 1. Parse Constraints (try LLM first, then fallback)
        try:
            constraints = self.parse_constraints(query)
        except Exception as e:
            logging.error(
                f"Constraint parsing failed unexpectedly: {e}. Proceeding without constraints."
            )
            constraints = {
                "duration_max": None,
                "test_types": [],
                "skills": [],
                "remote_only": False,
                "adaptive_only": False,
            }

        # 2. Semantic Search (Embedding Similarity)
        logging.info("Performing semantic search...")
        try:
            query_embedding = self.embedding_model.encode(
                query, convert_to_tensor=True, device=self.device
            )
            # Ensure embeddings tensor is on the same device as query embedding
            embeddings_on_device = self.embeddings.to(query_embedding.device)

            hits = util.semantic_search(
                query_embedding, embeddings_on_device, top_k=INITIAL_RETRIEVAL_K
            )

            if not hits or not hits[0]:
                logging.warning("No semantic search hits found.")
                return []
            logging.info(
                f"Found {len(hits[0])} initial candidates via semantic search."
            )
            initial_hits = hits[0]

        except Exception as e:
            logging.error(f"Semantic search failed: {e}")
            return []  # Cannot proceed without initial candidates

        # 3. Re-ranking with CrossEncoder
        logging.info("Re-ranking candidates...")
        try:
            # Filter hits with invalid corpus_id before accessing self.texts
            valid_hits = [
                hit for hit in initial_hits if 0 <= hit["corpus_id"] < len(self.texts)
            ]
            if not valid_hits:
                logging.warning("No valid candidates after index check for reranking.")
                return []

            pairs = [(query, self.texts[hit["corpus_id"]]) for hit in valid_hits]
            # Run prediction - this might take time depending on pairs and CPU/GPU
            rerank_scores = self.reranker.predict(
                pairs, show_progress_bar=True, convert_to_numpy=True
            )  # Get numpy array

        except IndexError as ie:
            logging.error(
                f"Index error during reranking preparation: {ie}. Check corpus_id validity."
            )
            return []
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            # Optionally proceed without reranking, or return empty
            return []

        # 4. Combine Scores & Initial Sort (Embedding + Rerank Score)
        combined_results = []
        for i, hit in enumerate(valid_hits):
            # Ensure rerank_scores[i] is treated as float
            combined_score = float(hit["score"]) + float(rerank_scores[i])
            combined_results.append((hit["corpus_id"], combined_score))

        # Sort by the combined score, descending
        sorted_candidates = sorted(combined_results, key=lambda x: x[1], reverse=True)
        logging.info("Candidates re-ranked and sorted by combined score.")

        # 5. Apply Filters and Boosting based on Constraints
        logging.info("Applying filters and boosting based on constraints...")
        try:
            filtered_boosted_results = self._apply_filters_and_boost(
                sorted_candidates, constraints
            )
        except Exception as e:
            logging.error(
                f"Error during filtering/boosting: {e}. Returning unfiltered results."
            )
            # Fallback: return top_k from sorted_candidates if filtering fails
            # Convert index back to dictionary before returning
            unfiltered_results = []
            for idx, score in sorted_candidates[:top_k]:
                if 0 <= idx < len(self.df):
                    row_dict = (
                        self.df.iloc[idx]
                        .where(pd.notna(self.df.iloc[idx]), None)
                        .to_dict()
                    )
                    unfiltered_results.append(row_dict)
                else:
                    logging.warning(
                        f"Index {idx} out of bounds when creating fallback results."
                    )
            return unfiltered_results

        # 6. Format and Return top_k results
        # filtered_boosted_results contains (dict, boosted_score) tuples
        final_recommendations = [
            item_dict for item_dict, score in filtered_boosted_results[:top_k]
        ]
        logging.info(f"Returning {len(final_recommendations)} recommendations.")
        return final_recommendations

    def _apply_filters_and_boost(
        self, sorted_candidates: list, constraints: dict
    ) -> list:
        """Filters candidates based on constraints and boosts scores for skill matches."""
        filtered_boosted = []
        logging.debug(f"Applying constraints: {constraints}")

        for idx, combined_score in sorted_candidates:
            if not (0 <= idx < len(self.df)):
                logging.warning(
                    f"Candidate index {idx} out of bounds for DataFrame. Skipping."
                )
                continue

            row = self.df.iloc[idx]
            passes_filter = True  # Assume it passes unless a filter fails

            # --- Duration Filter ---
            if constraints["duration_max"] is not None and pd.notna(
                row["duration_minutes"]
            ):
                try:
                    # Ensure comparison is float to int/float
                    if float(row["duration_minutes"]) > float(
                        constraints["duration_max"]
                    ):
                        passes_filter = False
                        logging.debug(
                            f"Idx {idx}: Failed duration filter ({row['duration_minutes']} > {constraints['duration_max']})"
                        )
                except (ValueError, TypeError) as e:
                    logging.warning(
                        f"Idx {idx}: Could not compare duration '{row['duration_minutes']}' with constraint '{constraints['duration_max']}': {e}"
                    )
                    # Decide if invalid format should fail the filter (depends on requirements)
                    # passes_filter = False

            # --- Test Type Filter ---
            if passes_filter and constraints["test_types"]:
                # Item must have a test type if filtering by type
                if pd.isna(row["test_type"]) or not row["test_type"]:
                    passes_filter = False
                    logging.debug(
                        f"Idx {idx}: Failed test type filter (item has no type)"
                    )
                else:
                    item_test_types = str(row["test_type"]).lower()
                    # Check if *any* required type is present in the item's types
                    # Assumes test_type column contains the mapped, human-readable names
                    match_found = False
                    for required_type in constraints["test_types"]:
                        # Use word boundaries for more precise matching
                        if re.search(
                            r"\b" + re.escape(required_type.lower()) + r"\b",
                            item_test_types,
                        ):
                            match_found = True
                            break
                    if not match_found:
                        passes_filter = False
                        logging.debug(
                            f"Idx {idx}: Failed test type filter (Types '{item_test_types}' don't contain required '{constraints['test_types']}')"
                        )

            # --- Remote Only Filter ---
            if passes_filter and constraints["remote_only"]:
                # Check if 'remote_testing' column is explicitly 'Yes' (case-insensitive)
                if str(row["remote_testing"]).lower() != "yes":
                    passes_filter = False
                    logging.debug(
                        f"Idx {idx}: Failed remote_only filter (Value: '{row['remote_testing']}')"
                    )

            # --- Adaptive Only Filter ---
            if passes_filter and constraints["adaptive_only"]:
                # Check if 'adaptive_support' column is explicitly 'Yes' (case-insensitive)
                if str(row["adaptive_support"]).lower() != "yes":
                    passes_filter = False
                    logging.debug(
                        f"Idx {idx}: Failed adaptive_only filter (Value: '{row['adaptive_support']}')"
                    )

            # --- If all filters passed, calculate boosted score ---
            if passes_filter:
                boosted_score = combined_score
                # --- Score Boosting for Skill Matches ---
                if constraints["skills"] and pd.notna(row["search_text"]):
                    search_text_lower = str(row["search_text"]).lower()
                    num_matches = sum(
                        1
                        for s in constraints["skills"]
                        if re.search(
                            r"\b" + re.escape(s.lower()) + r"\b", search_text_lower
                        )
                    )
                    if num_matches > 0:
                        boost = num_matches * SKILL_BOOST_FACTOR
                        boosted_score += boost
                        logging.debug(
                            f"Idx {idx}: Boosting score by {boost:.2f} for {num_matches} skill match(es). New score: {boosted_score:.4f}"
                        )

                # Convert row to dict safely, handle potential NaN conversion issues
                try:
                    # Fill NaN with None for JSON compatibility before converting
                    row_dict = row.where(pd.notna(row), None).to_dict()
                    filtered_boosted.append((row_dict, boosted_score))
                except Exception as e:
                    logging.error(f"Error converting row {idx} to dict: {e}")
                    # Optionally skip this row if conversion fails

        # Sort final list by boosted score
        return sorted(filtered_boosted, key=lambda x: x[1], reverse=True)


# --- Example Usage ---
if __name__ == "__main__":
    # Ensure the HF_API_KEY environment variable is set before running
    # Example: export HF_API_KEY='your_hugging_face_api_key'

    if not os.getenv("HF_API_KEY"):
        logging.error(
            "FATAL: HF_API_KEY environment variable not set. Cannot initialize LLM."
        )
    elif not os.path.exists(DATA_PATH):
        logging.error(f"FATAL: Data file not found at {DATA_PATH}")
    else:
        try:
            print("\n--- Starting Recommendation Engine Test ---")
            # Initialize with default path
            recommender = AssessmentRecommender(data_path=DATA_PATH)

            # --- Test Queries from Assignment ---
            test_queries = [
                "I am hiring for Java developers who can also collaborate effectively with my business teams.",
                "Looking for an assessment(s) that can be completed in 40 minutes.",
                "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script.",
                "Need an assessment package that can test all skills with max duration of 60 minutes.",
                "Here is a JD text, can you recommend some assessment that can help me screen applications. Time limit is less than 30 minutes.",  # Corrected 'ID' to 'JD' based on context
                "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins.",
            ]

            for test_query in test_queries:
                print(f"\n--- Testing Query: '{test_query}' ---")
                recommendations = recommender.recommend(
                    test_query, top_k=5
                )  # Get top 5 for demo

                print("\n--- Top 5 Recommendations ---")
                if recommendations:
                    for i, rec in enumerate(recommendations):
                        # Extract required fields safely using .get()
                        name = rec.get("name", "N/A")
                        url = rec.get("url", "#")
                        duration = rec.get("duration_minutes", "N/A")
                        test_type = rec.get(
                            "test_type", "N/A"
                        )  # Already mapped readable name
                        remote = rec.get("remote_testing", "N/A")
                        adaptive = rec.get("adaptive_support", "N/A")

                        print(f"\n{i + 1}. {name}")
                        print(f"   URL: {url}")
                        print(f"   Duration: {duration} min | Type: {test_type}")
                        print(f"   Remote: {remote} | Adaptive: {adaptive}")
                else:
                    print("   No recommendations found for this query.")

            print("\n--- Recommendation Engine Test Finished ---")

        except ValueError as ve:
            logging.error(f"Configuration Error during test: {ve}")
        except Exception as e:
            logging.exception(f"An unexpected error occurred during the test run: {e}")
            # Using logging.exception automatically includes traceback
