import pickle

ACTIVATIONS_PATH = '/Users/tomas/PycharmProjects/odors/results/activations/GAT_0.pickle'

# Now you can pickle only one object:
with open(ACTIVATIONS_PATH, "rb") as handle:
    activations = pickle.load(handle)

vanilin_activations = activations['vanillin']
acetophenone_activations = activations['acetophenone']

for key, value in activations.items():
    print(key, value)

import pickle
import csv


def pickle_to_csv(pickle_file, output_csv_file):
    # Load the dictionary from the pickle file
    with open(pickle_file, 'rb') as pfile:
        data = pickle.load(pfile)

    # Prepare to write to CSV
    fieldnames = ['molecule']
    # Generate field names based on assumed length of vectors for GAT entries
    example_molecule = next(iter(data.values()))
    example_keys = list(example_molecule.keys())  # assuming all molecules have the same structure
    fieldnames += [f'GAT{i}_{j}' for i in range(len(example_keys)) for j in
                   range(len(example_molecule[example_keys[0]]))]

    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write each molecule's data to the CSV
        for molecule, activations in data.items():
            row = {'molecule': molecule}
            # Renaming and flattening the activations into the row dictionary
            for i, (key, values) in enumerate(activations.items()):
                for idx, value in enumerate(values):
                    row[f'GAT{i}_{idx}'] = value
            writer.writerow(row)

    print(f"Data has been successfully saved to '{output_csv_file}'")


def pickle_to_individual_csvs(pickle_file, outputs):
    # Load the dictionary from the pickle file
    with open(pickle_file, 'rb') as pfile:
        data = pickle.load(pfile)

    # Process each molecule individually
    for molecule, activations in data.items():
        # Determine filename based on molecule name
        output_csv_file = f"{outputs}/{molecule}.csv"
        # Create a CSV file for each molecule
        with open(output_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Assuming there are 5 GAT entries and each has a vector of 64 activations
            for i in range(5):  # we are assuming the keys are in the correct order
                key = list(activations.keys())[i]  # get the key for the current GAT
                writer.writerow(activations[key])  # write each vector as a row

        print(f"Data for {molecule} has been successfully saved to '{output_csv_file}'")


pickle_to_csv(ACTIVATIONS_PATH, '/Users/tomas/PycharmProjects/odors/results/output_activations.csv')
pickle_to_individual_csvs(ACTIVATIONS_PATH, outputs='/Users/tomas/PycharmProjects/odors/results')