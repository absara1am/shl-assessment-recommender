import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import re
import time

BASE_URL = "https://www.shl.com"
MAIN_CATALOG_URL = BASE_URL + "/solutions/products/product-catalog/"
OUTPUT_CSV = "shl_assessments.csv"


def parse_duration(text):
    """Improved duration parser that handles different text patterns"""
    try:
        # Look for various duration patterns
        patterns = [
            r"(\d+)\s*min(?:ute)?s?",  # "45 minutes"
            r"(\d+)\s*-\s*(\d+)\s*min",  # "30-45 min"
            r"approx\.?\s*(\d+)\s*min",  # "Approx. 30 min"
            r"time:\s*(\d+)\s*min",  # "Time: 60 min"
            r"(\d+)\s*minutes?\s*=",  # "minutes = 22"
            r"\b(\d{2,3})\b(?![\.\d])",  # Standalone 2-3 digit numbers
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the first number found
                numbers = [
                    int(num)
                    for group in matches
                    for num in (group if isinstance(group, tuple) else [group])
                ]
                return max(numbers) if numbers else None

        return None
    except:
        return None


def get_assessment_details(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        details = {
            "duration": None,
            "description": "",
            "job_levels": [],
            "languages": [],
        }

        # Parse duration with improved logic
        duration_label = soup.find(
            ["h4", "h3"], string=lambda t: t and "assessment length" in t.lower()
        )
        if duration_label:
            duration_text = duration_label.find_next("p").get_text()
            details["duration"] = parse_duration(duration_text)

        # Parse description
        desc_label = soup.find(
            ["h4", "h3"], string=lambda t: t and "description" in t.lower()
        )
        if desc_label:
            details["description"] = desc_label.find_next("p").get_text(strip=True)

        # Parse job levels
        levels_label = soup.find(
            ["h4", "h3"], string=lambda t: t and "job levels" in t.lower()
        )
        if levels_label:
            details["job_levels"] = [
                l.strip()
                for l in levels_label.find_next("p").get_text().split(",")
                if l.strip()
            ]

        # Parse languages
        lang_label = soup.find(
            ["h4", "h3"], string=lambda t: t and "languages" in t.lower()
        )
        if lang_label:
            details["languages"] = [
                lang.strip()
                for lang in lang_label.find_next("p").get_text().split(",")
                if lang.strip()
            ]

        return details

    except Exception as e:
        print(f"Error parsing {url}: {str(e)}")
        return None


def scrape_shl_catalog():
    all_assessments = []

    # Pagination parameters
    start = 0
    per_page = 12
    total_pages = 32

    for page in range(total_pages):
        url = f"{MAIN_CATALOG_URL}?start={start}&type=1"
        print(f"Scraping page {page + 1}/{total_pages}: {url}")

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table")

            if not table:
                print("No table found on page")
                continue

            for row in table.find_all("tr")[1:]:  # Skip header
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue

                # Extract basic info
                name_link = cols[0].find("a")
                assessment_url = urljoin(BASE_URL, name_link["href"])
                name = name_link.get_text(strip=True)

                remote_testing = "Yes" if cols[1].find("span", class_="-yes") else "No"
                adaptive_support = (
                    "Yes" if cols[2].find("span", class_="-yes") else "No"
                )
                test_type = [
                    span.get_text()
                    for span in cols[3].find_all(
                        "span", class_="product-catalogue__key"
                    )
                ]

                # Get detailed info with retry logic
                max_retries = 3
                details = None
                for attempt in range(max_retries):
                    try:
                        time.sleep(1 + attempt)  # Increasing delay
                        details = get_assessment_details(assessment_url)
                        if details:
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(
                                f"Failed to parse {assessment_url} after {max_retries} attempts"
                            )

                if details:
                    assessment_data = {
                        "name": name,
                        "url": assessment_url,
                        "remote_testing": remote_testing,
                        "adaptive_support": adaptive_support,
                        "test_type": ", ".join(test_type),
                        "duration_minutes": details["duration"],
                        "description": details["description"],
                        "job_levels": ", ".join(details["job_levels"]),
                        "languages": ", ".join(details["languages"]),
                    }
                    all_assessments.append(assessment_data)

            start += per_page

        except Exception as e:
            print(f"Error scraping page {page + 1}: {str(e)}")
            continue

    # Create DataFrame and clean data
    df = pd.DataFrame(all_assessments)

    # Post-processing for durations
    df["duration_minutes"] = df["duration_minutes"].combine_first(
        df["description"].apply(parse_duration)
    )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Scraping complete! Saved {len(df)} assessments to {OUTPUT_CSV}")
    return df


# Start scraping
scrape_shl_catalog()
