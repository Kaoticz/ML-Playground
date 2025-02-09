# Confusion Matrix

[Click here][Notebook] to go to the Jupyter Notebook.

## Summary

This project features the computation of metrics that are important for Machine Learning.

- **Accuracy**: Measures the proportion of correctly classified instances out of the total instances. It is useful when the classes are balanced but can be misleading in imbalanced datasets.
- **Sensitivity (Recall or True Positive Rate)**: Measures the proportion of actual positives that are correctly identified. It is important in scenarios where missing positive cases is costly, such as disease detection.
- **Specificity (True Negative Rate)**: Measures the proportion of actual negatives that are correctly identified. It is crucial in situations where false positives are costly, like fraud detection.
- **Precision (Positive Predictive Value)**: Measures how many of the predicted positives are actually positive. It is useful when false positives need to be minimized, such as in spam detection.
- **F-Score (F1-Score)**: The harmonic mean of precision and recall, balancing the trade-off between them. It is useful when both false positives and false negatives need to be considered equally.



## Running it on Google Colab

Upload the [Confusion_Matrix.ipynb][Notebook] file to Google Colab, then click `Runtime > Run all` or press `Ctrl + F9` on your keyboard.


[Notebook]: ./Confusion_Matrix.ipynb
