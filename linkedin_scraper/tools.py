# fmt: off
from langchain.serpapi import SerpAPIWrapper
# fmt: on


# function to get profile url
def get_profile_url(text: str) -> str:
    """ search for linkedin profile page and return url"""
    search = SerpAPIWrapper()
    results = search.run(f"{text}")
    return results
