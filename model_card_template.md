# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was developed to predict whether an individual's income is greater than or less than $50,000 annually. For this binary classification task, a Random Forest classifier from scikit-learn was used. The model was trained using a grid search over a modest set of hyperparameters, including n_estimators, max_depth, and min_samples_split, with 3-fold cross-validation. The best-performing model was selected based on F1 score. During preprocessing, categorical features were one-hot encoded and the target labels were binarized.

## Intended Use
The primary purpose of this model is to demonstrate the development of a machine learning pipeline, including data processing, training, evaluation, and deployment. It is intended for use by students, data scientists, or researchers for educational purposes or exploratory analysis of income patterns. This model is not intended for real-world or production use and should not be used to support individual-level decisions, such as hiring, lending, or other financial determinations.

## Training Data
The data used to train this model comes from the publicly available UCI Adult Census dataset. This dataset contains demographic and socioeconomic features such as age, education, occupation, marital status, race, and sex. The target label indicates whether an individual's income is less than or equal to $50,000 annually or greater than $50,000 annually. The dataset was split into training and test sets using an 80/20 ratio.

## Evaluation Data
The model was evaluated on a test dataset comprising 20% of the original UCI Adult Census dataset. This subset was held out during training and was not used to fit the model. Its purpose is to evaluate the performance of the trained model on unseen data.

## Metrics
The model was evaluated using precision, recall, and F1 score. It achieved a precision of 0.7610, recall of 0.6384, and an F1 score of 0.6944.

These results indicate that when the model predicts an individual earns more than $50,000, it is correct approximately 76% of the time (precision). However, it only identifies about 64% of all individuals who actually earn more than $50,000 (recall), meaning it misses a notable portion of positive cases. This suggests the model is somewhat conservative in predicting higher income, favoring fewer false positives at the cost of increased false negatives. The F1 score reflects this imbalance between precision and recall.

## Ethical Considerations
This dataset includes several sensitive features, such as race, sex, education, and occupation, which introduces the potential for biased outcomes. Because the data reflects real-world societal inequalities, the model may learn and reinforce these patterns, potentially resulting in disparate performance across demographic groups. Slice analysis conducted on the dataset showed that model performance varies across subgroups, with lower performance observed in some underrepresented groups. Due to these concerns, this model should not be used for high-stakes decision-making

## Caveats and Recommendations
This model has several limitations. The hyperparameter tuning was conducted over a relatively small search space, and model performance could likely be improved with a more extensive grid search or alternative optimization methods. Additionally, only a Random Forest classifier was used, and no comparison was made with other models such as logistic regression or gradient boosting methods, which may yield better performance.

While cross-validation was incorporated during training, increasing the number of folds could provide a more robust estimate of model performance. The model also exhibits a tradeoff between precision and recall, with lower recall indicating that a significant portion of positive cases are missed. This may suggest issues such as class imbalance or the need for further tuning.

Finally, the model is trained on a dataset that may not generalize well to other populations or time periods, and it inherits potential biases present in the data. Future improvements could include expanding model experimentation, improving hyperparameter tuning, and addressing potential data imbalance or bias.