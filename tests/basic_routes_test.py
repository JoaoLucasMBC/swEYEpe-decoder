import requests
import json
import time
from pathlib import Path

def load_json_file(filepath):
    """Load JSON data from a file."""
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {filepath}")
        return None

def make_post_request(url, data):
    """Make a POST request with JSON data."""
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        print(f"Success: POST to {url}")
        print(f"Response status code: {response.status_code}")
        print(f"Response body: {response.text}\n")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {str(e)}\n")
        return False

def main():
    # Configuration
    base_url = "http://localhost:5000"  # Adjust port if needed
    setup_endpoint = f"{base_url}/setup"
    general_endpoint = f"{base_url}/general"
    delay_seconds = 3

    # Load JSON data
    setup_data = load_json_file(Path("layout.txt"))
    general_data = load_json_file(Path("incoming.txt"))

    if setup_data is None or general_data is None:
        print("Failed to load required JSON files. Exiting.")
        return

    # Make setup request
    print("Making setup request...")
    if not make_post_request(setup_endpoint, setup_data):
        print("Setup request failed. Exiting.")
        return

    # Wait for specified delay
    print(f"Waiting for {delay_seconds} seconds...")
    time.sleep(delay_seconds)

    # Make general request
    print("Making general request...")
    make_post_request(general_endpoint, general_data)

if __name__ == "__main__":
    main()