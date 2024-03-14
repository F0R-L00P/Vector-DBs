import json
import numpy as np
import matplotlib.pyplot as plt

import ray
from pathlib import Path
from bs4 import BeautifulSoup


# Set the directory where your scikit-learn documentation is stored
EFS_DIR = "C:\\Users\\behna\\OneDrive\\Documents\\GitHub\\LangChain-LLMs\\vectorDB"  # Update this path as necessary
DOCS_DIR = Path(
    EFS_DIR
)  # No need to add "docs.ray.io/en/master/" since we are using sklearn docs

# Initialize Ray
ray.init()

# Create a dataset from the HTML files in the sklearn documentation directory
ds = ray.data.from_items(
    [{"path": str(path)} for path in DOCS_DIR.rglob("*.html") if not path.is_dir()]
)
print(f"{ds.count()} documents")

# check sample outputs
ds.take(3)


def extract_sections(file_info):
    # Open and read the HTML file
    with open(file_info["path"], "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find all sections - assuming sections are under 'h2' headings
    sections = soup.find_all("h2")
    extracted_data = []
    for section in sections:
        # Assuming next_siblings can be used to get text following a header until the next header
        text = "".join(sibling.get_text() for sibling in section.find_next_siblings())
        extracted_data.append(
            {"url": file_info["path"], "section": section.get_text(), "text": text}
        )

    return extracted_data


# Testing function
# Take one sample from the dataset
sample_data = ds.take(1)
sample_file_info = sample_data[0]  # Get the first (and only) item in the list
sample_output = extract_sections(sample_file_info)
print(json.dumps(sample_output, indent=4))

# extracting the entire ds
# Apply the function to each item in the dataset and flatten the result
sections_ds = ds.flat_map(extract_sections)

# Retrieve all results
sections = sections_ds.take_all()
# Print out the number of sections extracted
print(f"Extracted {len(sections)} sections")

# print out a few sections to check
print(json.dumps(sections[:3], indent=4))

# Lets visualize the length of text
section_lengths = [len(section["text"]) for section in sections]

# Create the scatter plot
plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
plt.scatter(
    range(len(section_lengths)), section_lengths, alpha=0.5
)  # Plot section lengths
plt.title("Section lengths")
plt.xlabel("Section")
plt.ylabel("# Characters")
plt.show()

# now lets start chuncking
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Text splitter
chunk_size = 1000
chunk_overlap = 100
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)

# Chunk a sample section
sample_section = sections_ds.take(1)[0]
chunks = text_splitter.create_documents(
    texts=[sample_section["text"].replace("¶", "")],
    metadatas=[{"source": sample_section["section"].replace("¶", "")}],
)
print(chunks[0])


# building a chunking function to scale
def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        texts=[section["text"].replace("¶", "")],
        metadatas=[{"source": section["section"].replace("¶", "")}],
    )
    return [
        {"text": chunk.page_content, "source": chunk.metadata["source"]}
        for chunk in chunks
    ]


# now lets scale this
chunks_ds = sections_ds.flat_map(
    partial(chunk_section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
)
print(f"{chunks_ds.count()} chunks")
chunks_ds.show(1)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Time points and resilience values
time = np.linspace(0, 10, 100)
t1 = 2  # Time of disruption
t2 = 8  # Time of recovery
resilience = np.piecewise(
    time,
    [time < t1, (time >= t1) & (time <= t2), time > t2],
    [1, lambda t: 1 - (0.5 * (t - t1) / (t2 - t1)), 1],
)

# Data preparation
data = pd.DataFrame({"Time": time, "Resilience Indicator (%)": resilience * 100})

# Seaborn settings and plot
sns.set_theme(
    style="whitegrid"
)  # Set Seaborn theme to show white background with grid lines
plt.figure(figsize=[8, 4])
ax = sns.lineplot(
    x="Time",
    y="Resilience Indicator (%)",
    data=data,
    linestyle="--",
    color="k",
    label="Expected functionality",
)

# Matplotlib for filling between lines and adding lines
ax.fill_between(
    time,
    0,
    resilience * 100,
    where=(time >= t1) & (time <= t2),
    color="grey",
    alpha=0.4,
    label="Resilience Loss",
)
ax.axhline(y=100, color="r", linestyle="-", linewidth=2.5, label="Normal Functionality")
ax.axvline(x=t1, color="b", linestyle=":", label="Event Occurs")
ax.axvline(x=t2, color="g", linestyle=":", label="Full Recovery")

# Final touches
ax.set(title="System Resiliency", xlabel="Time", ylabel="Resilience Indicator (%)")
ax.set_xticks([0, t1, t2, 10])
ax.set_xticklabels(["0", "$t_1$ (Disruption)", "$t_2$ (Recovery)", "10"])
ax.set_ylim(0, 110)
ax.set_xlim(0, 10)
ax.legend()
sns.despine()  # Removes the top and right border
plt.tight_layout()  # Adjusts plot parameters for best fit
plt.show()
