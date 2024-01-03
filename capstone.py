# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split as tsplit
from surprise import accuracy as spaccuracy
import seaborn as sns
sns.set(style="whitegrid")

# set seed
seed_n = 15547187

# load the dataset 
file_path = '../data/spotify52kData.csv'  
df = pd.read_csv(file_path)

# check data types and null values
print(df.dtypes)
print(df.isnull().sum())

#### Question 1 ####
# create dataframe for Question 1
df1 = df[['track_name', 'album_name', 'artists', 'duration', 'popularity']]

# Find duplicates based on 'track_name' and 'artists'
duplicates = df1.sort_values(by=['track_name', 'artists'])[df1.sort_values(by=['track_name', 'artists']).duplicated(subset=['track_name', 'artists', 'duration'], keep=False)]
# Drop duplicates and keep the mean popularity
duplicates['popularity'] = duplicates.groupby(['track_name', 'artists'])['popularity'].transform('mean')
duplicates = duplicates.drop_duplicates(subset=['track_name', 'artists', 'duration'], keep='first')
df1 = df1.drop_duplicates(subset=['track_name', 'artists', 'duration'], keep=False)
# Concatenate the modified duplicates to the result DataFrame and drop strings
df1 = pd.concat([df1, duplicates])
df1 = df1[['duration', 'popularity']]

# calculate pearson correlation coefficient
pearson = df1.corr(method = 'pearson')
print('pearson corr coef of all popularity')
print(pearson)

# run linear regression for popularity using duration
x = df1['duration'].values.reshape(-1,1)
y = df1['popularity'].values.reshape(-1,1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y)
        
model = LinearRegression()
model.fit(x_scaled, y_scaled)

y_pred = model.predict(x_scaled)

# evaluate model
beta0 = model.intercept_
beta1 = model.coef_
r2 = r2_score(y_scaled, y_pred)
print('beta0 =', beta0, 'beta1 =', beta1, 'R^2 =', r2)

# scatterplot of duration vs popularity
fig = plt.figure(figsize = (9,6)) 
plt.scatter(x_scaled, y_scaled, s=10, c="dodgerblue", marker='o', edgecolor="skyblue")
plt.plot(x_scaled, y_pred, color='red', label='Linear Regression Line')
plt.title('Scaled Scatterplot of Duration vs Popularity', fontsize=15)
plt.xlabel('Song Length(ms)', fontsize=15)
plt.ylabel('Popularity', fontsize=15)
plt.xticks(fontsize=10) 
plt.yticks(fontsize=10)
plt.legend()
plt.show()

# check 0 popularity songs
print('proportion of 0 popularity =', len(df1[df1['popularity']==0]) / len(df1))

# drop 0 popularity and calculate correlation
df1_2 = df1[df1['popularity']>=1]
pearson = df1_2.corr(method = 'pearson')
print('pearson corr coef of popularity > 0')
print(pearson)

# run linear regression for popularity using duration
x = df1_2['duration'].values.reshape(-1,1)
y = df1_2['popularity'].values.reshape(-1,1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y)
        
model = LinearRegression()
model.fit(x_scaled, y_scaled)

y_pred = model.predict(x_scaled)

# evaluate model
beta0 = model.intercept_
beta1 = model.coef_
r2 = r2_score(y_scaled, y_pred)
print('beta0 =', beta0, 'beta1 =', beta1, 'R^2 =', r2)


#### Question 2 ####
# Create dataframe for Question 2
df2 = df[['track_name', 'album_name', 'artists', 'explicit', 'popularity']]

# Find duplicates based on 'track_name' and 'artists'
duplicates = df2.sort_values(by=['track_name', 'artists'])[df2.sort_values(by=['track_name', 'artists']).duplicated(subset=['track_name', 'artists', 'explicit'], keep=False)]
# Drop duplicates and keep the mean popularity
duplicates['popularity'] = duplicates.groupby(['track_name', 'artists'])['popularity'].transform('mean')
duplicates = duplicates.drop_duplicates(subset=['track_name', 'artists', 'explicit'], keep='first')
df2 = df2.drop_duplicates(subset=['track_name', 'artists', 'explicit'], keep=False)
# Concatenate the modified duplicates to the result DataFrame
df2 = pd.concat([df2, duplicates])

# calculate mean and median
print(df2.groupby('explicit')['popularity'].mean())
print(df2.groupby('explicit')['popularity'].median())

# split into two groups
explicit = df2[df2['explicit']==True]['popularity'].to_numpy()
non_explicit = df2[df2['explicit']==False]['popularity'].to_numpy()

# distribution of popularity for each group
fig = plt.figure(figsize = (9,6)) 
plt.hist(explicit, density=True, bins=20, alpha=0.5, label='Explicit', color='red')
plt.hist(non_explicit, density=True, bins=20, alpha=0.5, label='Non-Explicit', color='blue')
plt.title('Distribution of Popularity', fontsize=15)
plt.xlabel('Popularity', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.legend()
plt.show()

# run welch's t test
t, pval = stats.ttest_ind(explicit, non_explicit, alternative='greater', random_state=seed_n, equal_var=False) ##
print('welch t =', t, 'p_value =', pval)

# run mann-whiteney u test
U, pval = stats.mannwhitneyu(explicit, non_explicit, alternative='greater') ##
print('mann whiteney U =', U, 'p_value =', pval)



#### Question 3 ####
# Create dataframe for Question 3
df3 = df[['track_name', 'album_name', 'artists', 'mode', 'popularity']]

# Find duplicates based on 'track_name' and 'artists'
duplicates = df3.sort_values(by=['track_name', 'artists'])[df3.sort_values(by=['track_name', 'artists']).duplicated(subset=['track_name', 'artists', 'mode'], keep=False)]
# Drop duplicates and keep the mean popularity
duplicates['popularity'] = duplicates.groupby(['track_name', 'artists'])['popularity'].transform('mean')
duplicates = duplicates.drop_duplicates(subset=['track_name', 'artists', 'mode'], keep='first')
df3 = df3.drop_duplicates(subset=['track_name', 'artists', 'mode'], keep=False)
# Concatenate the modified duplicates to the result DataFrame
df3 = pd.concat([df3, duplicates])

# calculate mean and median
print(df3.groupby('mode')['popularity'].mean())
print(df3.groupby('mode')['popularity'].median())

# split into two groups
major = df3[df3['mode']==1]['popularity'].to_numpy()
minor = df3[df3['mode']==0]['popularity'].to_numpy()

# distribution of popularity for each group
fig = plt.figure(figsize = (9,6)) 
plt.hist(major, density=True, bins=20, alpha=0.5, label='Major', color='red')
plt.hist(minor, density=True, bins=20, alpha=0.5, label='Minor', color='blue')
plt.title('Distribution of Popularity', fontsize=15)
plt.xlabel('Popularity', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.legend()
plt.show()

# run welch's t test
t, pval = stats.ttest_ind(major, minor, alternative='greater', random_state=seed_n, equal_var=False) ##
print('welch t =', t, 'p_value =', pval)

# run mann-whiteney u test
U, pval = stats.mannwhitneyu(major, minor, alternative='greater')
print('mann whiteney U =', U, 'p_value =', pval)


#### Question 4 ####
# define predictors and outcome variables
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'popularity'

# Create dataframe for Q4
df_4 = df[['track_name', 'album_name', 'artists', 'popularity']+features]

# check duplicates
duplicates = df_4.sort_values(by=['track_name', 'artists'])[df_4.sort_values(by=['track_name', 'artists']).duplicated(subset=features + ['track_name', 'artists'], keep=False)]
# Drop duplicates and keep the mean popularity
duplicates['popularity'] = duplicates.groupby(['track_name', 'artists'])['popularity'].transform('mean')
duplicates = duplicates.drop_duplicates(subset=features+['track_name', 'artists'], keep='first')
df_4 = df_4.drop_duplicates(subset=features + ['track_name', 'artists'] , keep=False)
# Concatenate the modified duplicates to the result DataFrame
df_4 = pd.concat([df_4, duplicates])

# Initialize variables to store best feature and its R-squared value
best_r2_feature = None
best_r2 = -1
best_rmse_feature = None
best_rmse = 10000

# run simple linear regression model
X = df_4[features]
y = df_4[target].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_n)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)

X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Iterate over each feature and run linear regression
for feature in features:
    predictor = X_train_scaled[feature].values.reshape(-1,1)
    test_predictor = X_test_scaled[feature].values.reshape(-1,1)
    
    model = LinearRegression()
    model.fit(predictor, y_train_scaled)

    y_pred = model.predict(test_predictor)

    # Evaluate the model
    r2 = r2_score(y_test_scaled, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred))

    # Update best feature
    if r2 > best_r2:
        best_r2 = r2
        best_r2_feature = feature
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_rmse_feature = feature
        
    # Print results
    print(f"Feature: {feature}")
    print(f"R-squared: {r2}")
    print(f"RMSE: {rmse}")
    print("-"*60)

# Print the best feature
print(f"The best predictor of popularity is: {best_r2_feature} with R-squared value: {best_r2}")
print(f"The best predictor of popularity is: {best_rmse_feature} with RMSE value: {best_rmse}")



#### Question 5 ####
# run multiple regression model
model_all = LinearRegression()
model_all.fit(X_train_scaled, y_train_scaled)

y_pred_all = model_all.predict(X_test_scaled)

# Evaluate the model
r2_all = r2_score(y_test_scaled, y_pred_all)
rmse_all = np.sqrt(mean_squared_error(y_test_scaled, y_pred_all))

# Print results
print("Multiple Regression Model with All Features:")
print(f"R-squared: {r2_all}")
print(f"RMSE: {rmse_all}")


# set GridSearchCV for Ridge
kf = KFold(n_splits=5, shuffle=True, random_state=seed_n)

param_grid = {'alpha': np.logspace(-3, 3, 7)}  

ridge_model = Ridge(random_state=seed_n)

grid_search = GridSearchCV(ridge_model, param_grid, cv=kf, scoring='neg_mean_squared_error')

# Fit the model with grid search
grid_search.fit(X_train_scaled, y_train_scaled)

# Get the best hyperparameters
best_alpha = grid_search.best_params_['alpha']

y_pred_ridge = grid_search.predict(X_test_scaled)

# Evaluate model
r2_ridge = r2_score(y_test_scaled, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test_scaled, y_pred_ridge))

# Print results
print("Regularized Ridge Regression Model (GridSearchCV):")
print(f"Best Alpha: {best_alpha}")
print(f"RMSE: {rmse_ridge}")
print(f"R-squared: {r2_ridge}")



#### Question 6 ####
# define features
all_features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Create dataframe for Q6
X_pca = df[all_features]

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_pca)

# run PCA without specifying the number of components
pca = PCA(random_state=seed_n)

X_pca_transformed = pca.fit_transform(X_standardized)

# Print the eigenvalues
eigenvalues = pca.explained_variance_
print("Eigenvalues:")
print(eigenvalues)

# Print the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
print(explained_variance_ratio)

# Print cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print("Cumulative Explained Variance:")
print(cumulative_explained_variance)

# Plot the scree plot with eigenvalues
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', label='Eigenvalue')
plt.title('Scree Plot for PCA (Eigenvalues)')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.legend()
plt.show()

# Plot the scree plot explained variance ratios
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', label='Individual Component')
plt.title('Scree Plot for PCA')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.legend()
plt.show()


# based on Scree Plot and Cumulative Explained Variance, we want to use 90% of Explained Variance 
# -> cutoff Principal Component 7
x_clustering = X_pca_transformed[:, :7]

# run DBSCAN
clustering = DBSCAN(eps=0.75, min_samples=8).fit(x_clustering)
print(np.unique(clustering.labels_))

x_clustering = pd.DataFrame(x_clustering)
x_clustering['cluster'] = clustering.labels_
x_clustering['genre'] = df['track_genre']

print(x_clustering.groupby('cluster')['genre'].unique())



#### Question 7 ####
# Create dataframe for Q7
df_7 = df[['track_name', 'album_name', 'artists', 'mode', 'valence']]
# check for duplicates
duplicates = df_7.sort_values(by=['track_name', 'artists'])[df_7.sort_values(by=['track_name', 'artists']).duplicated(subset=['track_name', 'artists', 'valence', 'mode'], keep=False)]
# Drop duplicates 
duplicates = duplicates.drop_duplicates(subset=['track_name', 'artists', 'valence', 'mode'], keep='first')
df_7 = df_7.drop_duplicates(subset=['track_name', 'artists', 'valence', 'mode'], keep=False)
# Concatenate the modified duplicates to the result DataFrame
df_7 = pd.concat([df_7, duplicates])


# define predictor and outcome
X = df_7[['valence']]
y = df_7['mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_n, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# run logistic regression
logistic_model = LogisticRegression(class_weight='balanced')
logistic_model.fit(X_train_scaled, y_train)

y_prob = logistic_model.predict_proba(X_test_scaled)[:,1]
y_pred = logistic_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

fpr_lg, tpr_lg, _ = roc_curve(y_test, y_prob)
auroc_lg = roc_auc_score(y_test, y_prob)

classification_report_result = classification_report(y_test, y_pred)

# Print results
print('Logistic Regression Results')
print(f"Accuracy: {accuracy}")
print('AUROC =', auroc_lg)
print("Classification Report:")
print(classification_report_result)
print('\nConfusion Matrix')
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()

# run SVM
svm_model = SVC(class_weight='balanced', probability=True)
svm_model.fit(X_train_scaled, y_train)

y_prob = svm_model.predict_proba(X_test_scaled)[:,1]
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob)
auroc_svm = roc_auc_score(y_test, y_prob)

classification_report_result = classification_report(y_test, y_pred)

# Print results
print('SVM Results')
print(f"Accuracy: {accuracy}")
print('AUROC =', auroc_svm)
print("Classification Report:")
print(classification_report_result)
print('\nConfusion Matrix')
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()

# run XGBoost
xgb_model = XGBClassifier(scale_pos_weight=(len(y) - y.sum()) / y.sum())  # Adjust scale_pos_weight
xgb_model.fit(X_train, y_train)

y_prob = xgb_model.predict_proba(X_test)[:,1]
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob)
auroc_xgb = roc_auc_score(y_test, y_prob)

classification_report_result = classification_report(y_test, y_pred)

# Print results
print('XGBoost Results')
print(f"Accuracy: {accuracy}")
print('AUROC =', auroc_xgb)
print("Classification Report:")
print(classification_report_result)
print('\nConfusion Matrix')
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()

# plot ROC curves
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], color = 'blue', linewidth = 1, linestyle='--', label='no skill')
plt.plot(fpr_lg, tpr_lg, color='green', label=f'Logistic Regression (area={round(auroc_lg,3)})')
plt.plot(fpr_svm, tpr_svm, color='red', label=f'SVM (area={round(auroc_svm,3)})')
plt.plot(fpr_xgb, tpr_xgb, color='purple', label=f'XGBoost (area={round(auroc_xgb,3)})')
plt.title('ROC Curves', fontsize=15)
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.legend(loc=4)
plt.show()



#### Question 8 ####
# define predictors and outcome variables
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'track_genre'

# Create dataframe for Q8
df_8 = df[['track_name', 'album_name', 'artists', 'track_genre']+features]

# check duplicates
duplicates = df_8.sort_values(by=['track_name', 'artists'])[df_8.sort_values(by=['track_name', 'artists']).duplicated(subset=features + ['track_name', 'artists'] + [target], keep=False)]
# Drop duplicates
duplicates['track_genre'] = duplicates.groupby(['track_name', 'artists'])['track_genre'].transform('last')
duplicates = duplicates.drop_duplicates(subset=features+['track_name', 'artists'] + [target], keep='first')
df_8 = df_8.drop_duplicates(subset=features + ['track_name', 'artists'] + [target], keep=False)
# Concatenate the modified duplicates to the result DataFrame
df_8 = pd.concat([df_8, duplicates])

X = df_8[features].values
y = df_8[target].values

# Use label encoding for outcome variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=seed_n, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# create deep neural network
class SongGenrePredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SongGenrePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

input_size = len(features)
num_classes = len(label_encoder.classes_)
model = SongGenrePredictor(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# run the model
num_epochs = 2000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store training loss
    train_losses.append(loss.item())

    # Test the model on the test set
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())

    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Plot the training and test loss curves
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model 
with torch.no_grad():
    _, predicted_labels = torch.max(model(X_test_tensor), 1)
    accuracy = accuracy_score(y_test, predicted_labels.numpy())

# Print results
print(f'Accuracy: {accuracy:.4f}')
classification_report_result = classification_report(y_test, predicted_labels.numpy())
print("Classification Report:")
print(classification_report_result)


#### Data Preprocessing for User Data ####
# Load the dataset
file_path = '../data/starRatings.csv'
ratings = pd.read_csv(file_path, header=None)

# Filter matching songs on spotify52k data
df_rated = df.iloc[:5000]


#### Question 9 ####
## a
# calculate average user rating for each song
ratings_by_song = ratings.T
ratings_by_song['mean_rating'] = ratings_by_song.mean(axis=1)

# join with the song features dataframe
df_rated = pd.merge(df_rated, ratings_by_song[['mean_rating']], how='left', left_index=True, right_index=True)
# check for correct join
print(df_rated['mean_rating'].isnull().sum())

# calculate correlation
pearson = df_rated[['popularity', 'mean_rating']].corr(method = 'pearson')
print('pearson corr coef')
print(pearson)

## b
# sort by descending mean user rating and select unique songs
## Top 10 by Spotify Average User Rating Measure
df_rated[['album_name', 'track_name', 'artists', 'popularity', 'mean_rating']].drop_duplicates(subset=['track_name', 'artists'], keep='first').sort_values(by='mean_rating', ascending=False).head(10)



#### Question 10 ####
# fill null values with 0 for user ratings matrix
ratings_10 = ratings.copy()
ratings_10.fillna(0, inplace=True)

# flatten the matrix
stacked_ratings_10 = ratings_10.stack().reset_index()
stacked_ratings_10.columns = ['user_id', 'song', 'rating']
stacked_ratings_10 = stacked_ratings_10[stacked_ratings_10['rating'] != 0]
stacked_ratings_10.reset_index(drop=True, inplace=True)


# set recommendation model
reader = Reader(rating_scale=(0, 4))

data = Dataset.load_from_df(stacked_ratings_10, reader)
trainset, testset = tsplit(data, test_size=0.2, random_state=seed_n)


# run user-based collaborative filtering model
sim_options = {'name': 'cosine', 'user_based': True}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

predictions = model.test(testset)

# Evaluate the model
spaccuracy.rmse(predictions)

# Generate mixtapes for all users
mixtapes = {}
for user_id in range(trainset.n_users):
    unrated_items = [item for item in trainset.all_items() if not trainset.ur[user_id, item]]
    predicted_ratings = [model.predict(user_id, item).est for item in unrated_items]

    top_n_songs = [unrated_items[i] for i in sorted(range(len(predicted_ratings)), key=lambda k: predicted_ratings[k], reverse=True)[:10]]

    mixtapes[user_id] = top_n_songs

# Print mixtapes for a 5 users
for user_id in range(5):
    print(f"Mixtape for User {user_id}: {mixtapes[user_id]}")
    
# Personalized Mixtape for User 3
mixtape_songs = df_rated.iloc[mixtapes[2]]
print(f"Mixtape for User {3}: ")
mixtape_songs[['album_name', 'track_name', 'artists', 'popularity', 'mean_rating']]



#### Extra Credit ####
# define predictors and outcome variables
features = ['tempo', 'danceability', 'valence', 'acousticness']
target = 'explicit'

# create dataframe for extra credit
df_c = df[['track_name', 'album_name', 'artists', 'track_genre']+ features + [target]]

# check duplicates
duplicates = df_c.sort_values(by=['track_name', 'artists'])[df_c.sort_values(by=['track_name', 'artists']).duplicated(subset=features + ['track_name', 'artists'], keep=False)]
# Drop duplicates
duplicates = duplicates.drop_duplicates(subset=features+['track_name', 'artists'] + [target], keep='first')
df_c = df_c.drop_duplicates(subset=features + ['track_name', 'artists'] + [target], keep=False)
# Concatenate the modified duplicates to the result DataFrame
df_c = pd.concat([df_c, duplicates])

X = df_c[features]
y = df_c[target]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_n, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# run logistic regression
model = LogisticRegression(random_state=seed_n, class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_prob = model.predict_proba(X_test_scaled)[:,1]
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

fpr_lg, tpr_lg, _ = roc_curve(y_test, y_prob)
auroc_lg = roc_auc_score(y_test, y_prob)

classification_report_result = classification_report(y_test, y_pred)

# Print results
print('Logistic Regression Results')
print(f"Accuracy: {accuracy}")
print('AUROC =', auroc_lg)
print('Classification Report:\n', classification_report_result)

# run XGBoost
model = XGBClassifier(random_state=seed_n, scale_pos_weight=(1 - y_train.mean()) / y_train.mean())
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob)
auroc_xgb = roc_auc_score(y_test, y_prob)

classification_report_result = classification_report(y_test, y_pred)

# Print results
print('XGBoost Results')
print(f"Accuracy: {accuracy}")
print('AUROC =', auroc_xgb)
print('Classification Report:\n', classification_report_result)

# plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], color = 'blue', linewidth = 1, linestyle='--', label='no skill')
plt.plot(fpr_lg, tpr_lg, color='green', label=f'Logistic Regression (area={round(auroc_lg,3)})')
plt.plot(fpr_xgb, tpr_xgb, color='red', label=f'XGBoost (area={round(auroc_xgb,3)})')
plt.title('ROC Curves', fontsize=15)
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.legend(loc=4)
plt.show()










