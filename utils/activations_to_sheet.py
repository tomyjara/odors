
import pickle
import pandas as pd

ACTIVATIONS_PATH = '/home/tomas/PycharmProjects/odors/results/13_mayo/activations/GAT_0.pickle'
PREDS_PATH = '/home/tomas/PycharmProjects/odors/results/13_mayo/predictions/GAT_0.pickle'

# Load activations and predictions
with open(ACTIVATIONS_PATH, "rb") as handle:
    activations = pickle.load(handle)

with open(PREDS_PATH, "rb") as handle:
    preds = pickle.load(handle)

# Prepare data for Excel
data = []
for molecule_name, molecule_activations in activations.items():
    for layer_name, activation in molecule_activations.items():
        row = {
            "Molecule Name": molecule_name,
            "Layer": layer_name,
            "Activations": activation,
            "Predictions": preds.get(molecule_name, "N/A")
        }
        data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel
excel_path = '../results/activations/activations_predictions.xlsx'
df.to_excel(excel_path, index=False)

print(f'Data saved to {excel_path}')