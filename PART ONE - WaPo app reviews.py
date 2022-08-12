#!/usr/bin/env python
# coding: utf-8

# ## Washington Post app reviews analysis
# 
# Since you all are investigative journalists, time to [reproduce this story](https://www.washingtonpost.com/technology/2019/11/22/apple-says-its-app-store-is-safe-trusted-place-we-found-reports-unwanted-sexual-behavior-six-apps-some-targeting-minors/)!
# 
# I've given you `reviews-marked.csv`, a CSV file of app reviews. Some are marked as being about unwanted sexual behavior – the `sexual` column – while some are unlabeled.
# 
# > You can use [this notebook](https://github.com/jsoma/dcj-indonesia-2022-machine-learning/blob/main/classification-app-reviews/Therapy%20app%20reviews.ipynb) as a source of cut-and-paste. It will get you 99% of the way there with practically no edits!
# 
# ## Section A: simple TF-IDF Vectorizer
# 
# 1. Build a classifier to filter for reviews for the journalists to check out.
# 2. Get a list of the most useful words for the classification
# 3. Use a test/train split and confusion matrix to determine how well your process works.
# 
# Since the labeled and unlabeled data is all in one file, you'll need to filter for ones with labels to build your training dataset.
# 
# You can use whatever kind of classifier you want. For your vectorizer, use a basic TF-IDF vectorizer with a 300-feature limit:
# 
# ```python
# vectorizer = TfidfVectorizer(max_features=300)
# ```
# 
# **How many of the unlabeled reviews does your believes are about unwanted sexual behavior?** Your classifier doesn't have to do a *good* job, it just has to work.

# In[1]:


import pandas as pd
pd.options.display.max_colwidth = 400

df = pd.read_csv("reviews-marked.csv")
df.head(3)


# In[2]:


labeled_data = df[df.sexual.notna()]
labeled_data.head(3)


# In[3]:


unlabeled_data = df[df.sexual.isna()]
unlabeled_data.head(3)


# In[4]:


print("Labeled is", len(labeled_data))
print("Unlabeled is", len(unlabeled_data))


# In[5]:


get_ipython().system('pip install pystemmer')
get_ipython().system('pip install sklearn')


# In[6]:


get_ipython().system('pip install --upgrade setuptools')


# In[7]:


get_ipython().system('pip install pystemmer')
get_ipython().system('pip install sklearn')


# In[8]:


get_ipython().system('pip install Cython')


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
import Stemmer

stemmer = Stemmer.Stemmer('en')
analyzer = TfidfVectorizer().build_analyzer()

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: stemmer.stemWords(analyzer(doc))

vectorizer = StemmedTfidfVectorizer(max_features=300)
matrix = vectorizer.fit_transform(labeled_data.Review)


# In[10]:


vectorizer.get_feature_names_out()


# In[11]:


pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())


# In[12]:


from sklearn.svm import LinearSVC

X = matrix
y = labeled_data.sexual

clf = LinearSVC(class_weight='balanced')
clf.fit(X, y)


# In[13]:


get_ipython().system('pip install eli5')


# In[14]:


import eli5

eli5.explain_weights(clf, feature_names=vectorizer.get_feature_names_out())


# In[20]:


X = vectorizer.fit_transform(labeled_data.Review)

labeled_data['predicted'] = clf.predict(X)
labeled_data['predicted_proba'] = clf.decision_function(X)


# In[21]:


labeled_data.sort_values(by='predicted_proba', ascending=False)


# In[22]:


labeled_data.sort_values(by='predicted_proba', ascending=False)


# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X = vectorizer.fit_transform(labeled_data.Review)
y = labeled_data.sexual

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train
clf = LinearSVC(class_weight='balanced')
clf.fit(X_train, y_train)

# Test
y_true = y_test
y_pred = clf.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

# How did it do?
label_names = pd.Series(['not sexual', 'sexual'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names)


# In[ ]:





# ## Section B: A custom vectorizer
# 
# Repeat the above, but **customize your `TfidfVectorizer`** in an attempt to improve your model. You can also change your classifier, if you'd like.
# 
# For the vectorizer, you can add:
# 
# |term|use|
# |---|---|
# |`vocabulary=`|a custom list of words to look at|
# |`stopwords=`|a list of words to ignore|
# |`max_features=`|the total number of features (words) to count|
# |`max_df=`|the maximum number of documents a word can appear in|
# |`min_df=`|the minimum number of documents a word must appear in|
# 
# For `max_df` and `min_df`, it can be a number or a decimal percentage. For example, `3` means 3 documents and `0.3` means 30% of documents.

# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer
import Stemmer

stemmer = Stemmer.Stemmer('en')
analyzer = TfidfVectorizer().build_analyzer()

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: stemmer.stemWords(analyzer(doc))

vectorizer = StemmedTfidfVectorizer(max_features=300, max_df=0.30)
matrix = vectorizer.fit_transform(labeled_data.Review)


# In[30]:


vectorizer.get_feature_names_out()


# In[31]:


pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())


# In[32]:


from sklearn.svm import LinearSVC

X = matrix
y = labeled_data.sexual

clf = LinearSVC(class_weight='balanced')
clf.fit(X, y)


# In[33]:


import eli5

eli5.explain_weights(clf, feature_names=vectorizer.get_feature_names_out())


# In[34]:


X = vectorizer.fit_transform(labeled_data.Review)

labeled_data['predicted'] = clf.predict(X)
labeled_data['predicted_proba'] = clf.decision_function(X)


# In[35]:


labeled_data.sort_values(by='predicted_proba', ascending=False)


# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X = vectorizer.fit_transform(labeled_data.Review)
y = labeled_data.sexual

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train
clf = LinearSVC(class_weight='balanced')
clf.fit(X_train, y_train)

# Test
y_true = y_test
y_pred = clf.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

# How did it do?
label_names = pd.Series(['not sexual', 'sexual'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names)


# In[ ]:




