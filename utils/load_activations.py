import pickle

ACTIVATIONS_PATH='../results/activations/GAT_0.pickle'
PREDS_PATH='../results/predictions/GAT_0.pickle'

# Now you can pickle only one object:
with open(ACTIVATIONS_PATH, "rb") as handle:
    activations = pickle.load(handle)

with open(PREDS_PATH, "rb") as handle:
    preds = pickle.load(handle)

vanilin_activations = activations['Vanillin']
acetophenone_activations = activations['Acetophenone']

print('Activations:')
for key, value in activations.items():
    print(key, value)

#for key, value in preds.items():
#    print(key, value)
