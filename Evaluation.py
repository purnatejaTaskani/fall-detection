# Evaluate the model on test data
y_pred = model.predict(X_test_processed) # Get model predictions on preprocessed test data
test_accuracy = np.mean(y_pred == y_test)  # Calculate accuracy manually

# Print accuracy as a number
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
