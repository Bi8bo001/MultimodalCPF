import pickle

with open("data/jarvis__megnet/train/raw/raw_data.pkl", "rb") as f:
    data = pickle.load(f)

count = 0
for i, sample in enumerate(data):
    if sample.get("structure") is None:
        print(f"[BAD] idx={i}, id={sample.get('material_id', 'N/A')}")
        count += 1

print(f"\nTotal bad samples: {count} / {len(data)}")
