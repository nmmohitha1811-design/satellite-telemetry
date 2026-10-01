import urllib.request
import os

# The 3 verified, real documents for our RAG corpus
documents = {
    "telemanom_paper.pdf": "https://arxiv.org/pdf/1802.04431",
    "curiosity_mobility_faults.pdf": "https://www-robotics.jpl.nasa.gov/media/documents/ROB-20-0040_R3.pdf",
    "ds1_beacon_monitor.pdf": "https://ntrs.nasa.gov/api/citations/20000056912/downloads/20000056912.pdf",
}

os.makedirs("documents", exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}

for filename, url in documents.items():
    path = os.path.join("documents", filename)
    print(f"Downloading {filename} ...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        data = response.read()
    with open(path, "wb") as f:
        f.write(data)
    size_kb = len(data) / 1024
    print(f"  Saved {path} ({size_kb:.1f} KB)")

print("Done.")