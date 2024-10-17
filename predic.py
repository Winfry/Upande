import json

# Open the .ipynb file
with open('UpandeModel.ipynb', 'r', encoding='utf-8') as f:
    notebook_data = json.load(f)

# Save it as a .json file
with open('UpandModel.json', 'w', encoding='utf-8') as f:
    json.dump(notebook_data, f, indent=4)

print("Conversion complete!")
