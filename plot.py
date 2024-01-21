import matplotlib.pyplot as plt


balanced_accuracies = {
    "full classification ViT": [0.31236, 0.50427, 0.63532, 0.741976, 0.79160, 0.8227, 0.834662, 0.84274, 0.84202, 0.84576, 0.84667, 0.84684],
    "probabilities match": [0.40522095, 0.527750074, 0.612671554, 0.66180676, 0.7370291, 0.8250614, 0.8513296],
    "cascading distillation": [0.4286494, 0.651873, 0.757191, 0.80902, 0.817527115, 0.833937, 0.8433085, 0.847497, 0.851631, 0.85169345, 0.8554955, 0.858396],
    "ViT": [0.8373],
    "SkinDistilViT": [0.8234],

}

accuracies = {
    "full classification ViT": [0.6362739, 0.7223209, 0.791000, 0.83797121, 0.870337, 0.88277089, 0.8928360, 0.89283597, 0.8924413, 0.89500689, 0.8959937, 0.8963884],
    "probabilities match": [0.59818440, 0.7128478, 0.7677126, 0.8113282, 0.851391375, 0.89046776, 0.896388411],
    "cascading distillation": [0.699230313, 0.801657, 0.8606669, 0.879810, 0.8857312, 0.89145, 0.898362, 0.8993487, 0.902309, 0.90230911, 0.9034931, 0.902703],
    "ViT": [0.8918],
    "SkinDistilViT": [0.8851]
}

fig, (ax1, ax2) = plt.subplots(1,2)

for m, key in zip("xos", balanced_accuracies):
    addition = 6 if 'probabi' in key else 1
    ax1.plot([i+addition for i in range(len(balanced_accuracies[key]))], balanced_accuracies[key], marker = m, label = key, markersize=4)
ax1.plot([12], [balanced_accuracies['ViT']], marker = "*", label="ViT")
ax1.plot([6], [balanced_accuracies['SkinDistilViT']],marker = "*", label="SkinDistilViT")
ax1.legend()
ax1.set_xlabel("Number of layers")
ax1.set_ylabel("Balanced accuracy")

for m, key in zip("xos", accuracies):
    addition = 6 if 'probabi' in key else 1
    ax2.plot([i+addition for i in range(len(accuracies[key]))], accuracies[key], marker = m, label = key, markersize=4)
ax2.plot([12], [accuracies['ViT']], marker = "*", label="ViT")
ax2.plot([6], [accuracies['SkinDistilViT']], marker = "*", label="SkinDistilViT")
ax2.legend()
ax2.set_xlabel("Number of layers")
ax2.set_ylabel("Accuracy")

plt.show()