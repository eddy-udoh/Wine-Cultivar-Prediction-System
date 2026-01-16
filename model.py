import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load wine dataset from sklearn
wine_dataset = load_wine()
feature_matrix = wine_dataset.data
target_vector = wine_dataset.target
cultivar_names = wine_dataset.target_names

# Split into training and validation sets
X_training, X_validation, y_training, y_validation = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42
)

# Create and fit normalizer
feature_normalizer = StandardScaler()
X_training_normalized = feature_normalizer.fit_transform(X_training)
X_validation_normalized = feature_normalizer.transform(X_validation)

# Initialize and train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_training_normalized, y_training)

# Calculate validation accuracy
validation_score = knn_classifier.score(X_validation_normalized, y_validation)
print(f"Validation accuracy: {validation_score:.2%}")

# Package model components
model_components = {
    'model': knn_classifier,
    'scaler': feature_normalizer,
    'target_names': cultivar_names
}

# Save to file
with open('model.h5', 'wb') as output_file:
    pickle.dump(model_components, output_file)

print("Model components successfully saved to 'model.h5'")