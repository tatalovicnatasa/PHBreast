from utils.utils import compute_classweights

fake_data = [(None, label) for label in [1, 0, 3, 4, 4, 3, 2, 1, 2, 2, 2, 2, 2, 2]]
fakes = []
# print(fake_data)
for data in fake_data:
    _, label = data
    fakes.append(label)

print(fakes)
weights = compute_classweights(fake_data, num_classes=5)
print(weights)
