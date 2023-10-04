import os
import json
import requests

# Read the API key from environment variables
api_key = os.environ.get("PROXYCURL_API_KEY")

# Print the API key for debugging purposes (remove in production)
print("API Key:", api_key)

# Define the API endpoint and parameters
api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"

params = {
    "linkedin_profile_url": "https://www.linkedin.com/in/vp",
}

# Add the Authorization header
headers = {"Authorization": f"Bearer {api_key}"}

# Perform the GET request
response = requests.get(api_endpoint, headers=headers, params=params)

# Print the JSON response
print(response.json())

# obtain content
# open json validator" jasonlint.com"
# save in github gist
response._content

# Fetch the Gist content
gist_response = requests.get(
    "https://gist.githubusercontent.com/F0R-L00P/7631ec7999d94eaedd33c9a803b79153/raw/93eb61f182828a66954bb3f73cb366411c8b9580/prof-profile.json"
)

# Replace improperly escaped apostrophes
corrected_text = gist_response.text.replace("\\'", "'")

# Try to decode the JSON
try:
    json_data = json.loads(corrected_text)
    print("JSON data successfully loaded.")
except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")

# Print the JSON data
print(json_data)
print(json_data["full_name"])
