import pickle
from sklearn.preprocessing import LabelEncoder

# Load old encoder
with open("models/label_encoder.pkl", "rb") as f:
    old_le = pickle.load(f)

print(f"✓ Loaded encoder")
print(f"  Classes : {len(old_le.classes_)}")
print(f"  Sample  : {list(old_le.classes_[:5])}")

# Re-save with current sklearn version
new_le = LabelEncoder()
new_le.classes_ = old_le.classes_

with open("models/label_encoder_v2.pkl", "wb") as f:
    pickle.dump(new_le, f)

print(f"✓ Saved label_encoder_v2.pkl")

# Verify
with open("models/label_encoder_v2.pkl", "rb") as f:
    test = pickle.load(f)
print(f"✓ Verified: {len(test.classes_)} classes")