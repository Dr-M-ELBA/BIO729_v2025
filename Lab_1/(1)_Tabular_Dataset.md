# Introduction

Tabular datasets are presented in a table-like format. These datasets are intuitive to interpret, with columns representing variables and rows representing samples. An example of tabular data is shown below:

| Sample | Variable 1 | Variable 2 | Variable 3 | Label |
|--------|------------|------------|------------|-------|
| 0      | 34         | 78         | 12         | 1     |
| 1      | 56         | 43         | 89         | 0     |
| 2      | 22         | 67         | 45         | 1     |
| 3      | 90         | 12         | 56         | 0     |
| 4      | 11         | 88         | 34         | 1     |
| 5      | 76         | 54         | 23         | 0     |
| 6      | 38         | 21         | 74         | 1     |
| 7      | 59         | 33         | 62         | 0     |
| 8      | 84         | 47         | 19         | 1     |
| 9     | 41         | 72         | 53         | 0     |

In this example, there are 5 columns in total. Three columns represent different variables, one column identifies the sample name, and one column contains the labels. We can also see that this dataset consists of 10 samples in total.

In the following example, we will work with a commonly used dataset: the breast cancer dataset. This dataset is widely recognised in the machine learning field, where the goal is to predict the presence of cancer based on features extracted from an image.

# Coding
As with any Python project, the first thing is to install the relevant packages. Without these, you'll be limited to simple functions, like addition and multiplication,
without extensive programming. Fortunately, some kind-hearted individuals have done the leg work so that if you wanted to upload your dataset, it can be easily done with
one line of code. So let's start by installing the relevant packages:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from google.colab import drive
```
Can you guess what each package does? Leave a comment beside each one with your guess.

Next, we need to mount our drive. This allows Colab to communicate with our directories:

```python
drive.mount('/content/drive')
```
Let's read our file. Notice how simple and interpretable the code is. Heads up, there is a bug in the following code:

```python
df = pd.read_excel('breast-cancer.csv')
```
Let's call our variable and inspect the data to make sure we are working with the right file:

```python
df
```

Clearly, there is a lot of information. Let's begin by inspecting the size of our dataframe:

```python
df.shape
```

Again, with a simple code, we can see how many columns and rows we have. As you progress with your machine learning careers you will
see the importance of knowing the number of columns and rows that you'll be working with in a given task.

We can also perform and inspect simple statistical information regarding our dataset with the following code:

```python
df.describe()
```
I always use the above code to inspect my data and look out for any anomalies. We can also beautify it if needed:

```python
df.describe().style.background_gradient(cmap='Blues')
```

Another key line of code is to inspect the datatype for each column. Again, this is to detect any errors that might be present:

```python
df.info()
```
Does the data make sense? Why are all but one column numerical?

In the following code, we will determine the percentage of outliers in each column. Outliers are known to have a negative impact, so
it is worth at the early stage of our pipeline to identify them:

```python
numerical_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                     'smoothness_mean', 'compactness_mean', 'concavity_mean',
                     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                     'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                     'smoothness_se', 'compactness_se', 'concavity_se',
                     'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                     'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                     'smoothness_worst', 'compactness_worst', 'concavity_worst',
                     'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

def calculate_outlier_percentage(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_percentage = (outliers / len(df)) * 100
    return outlier_percentage

outlier_df = pd.DataFrame({
    'Column': numerical_columns,
    'Outlier_Percentage': [calculate_outlier_percentage(col) for col in numerical_columns]
})

outlier_df

```

Up until now we have been performing EDA. The information has indeed been informative but when working with a large dataset, it can take time to interpret. 
As mentioned in the lecture, we can represent our results graphically instead of in text format. Let's begin by representing the percentage of outliers:

```python
fig = px.bar(outlier_df,
             x='Column',
             y='Outlier_Percentage',
             title='Outlier Percentage by Feature',
             color_discrete_sequence=['#12436D'])

fig.update_layout(
    title_x=0.5,
    xaxis_title='Feature',
    yaxis_title='Outlier Percentage (%)',
    template='plotly_white',
)

fig.update_xaxes(tickangle=90)

fig.show()
```

An important aspect of EDA is analysing the labels. In ML, the label distribution can significantly impact model training, with imbalanced labels having a
negative impact on model learning. So let's use EDA to examine the distribution of our labels:

```python
diagnosis_counts = df['diagnosis'].value_counts().reset_index()
diagnosis_counts.columns = ['Diagnosis', 'Count']

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Pie Chart: Cancer Type Distribution", "Bar Chart: Cancer Type Count"),
    specs=[[{"type": "domain"}, {"type": "xy"}]]
)

fig.add_trace(
    go.Pie(labels=diagnosis_counts['Diagnosis'],
           values=diagnosis_counts['Count'],
           marker=dict(colors=['#12436D', '#F46A25']),
           title=""),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=diagnosis_counts['Diagnosis'],
           y=diagnosis_counts['Count'],
           marker_color=['#12436D', '#F46A25'],
           text=diagnosis_counts['Count'],
           textposition='auto'),
    row=1, col=2
)

fig.update_layout(
    title_text="Cancer Type Distribution (Malignant vs Benign)",
    title_x=0.5,
    template='plotly_white',
    showlegend=False
)

fig.show()
```

Would you consider the dataset to be balanced or imbalanced?

Let's perform some bivariate analysis and see if we can find a pattern in our dataset. In the following code, we will plot the distribution
of each column and use colour to differentiate between the two labels:

```python
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
            'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
            'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

n_rows = 8
n_cols = 4

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=features,
    vertical_spacing=0.1,
    horizontal_spacing=0.05
)

for i, feature in enumerate(features):
    row = (i // n_cols) + 1
    col = (i % n_cols) + 1


    benign = df[df['diagnosis'] == 'B'][feature]
    malignant = df[df['diagnosis'] == 'M'][feature]


    fig.add_trace(
        go.Histogram(x=benign, nbinsx=30, name='Benign', marker_color='#12436D', opacity=0.6, showlegend=(i == 0)),
        row=row, col=col
    )


    fig.add_trace(
        go.Histogram(x=malignant, nbinsx=30, name='Malignant', marker_color='#F46A25', opacity=0.6, showlegend=(i == 0)),
        row=row, col=col
    )

fig.update_layout(
    title_text="Feature Distribution by Cancer Type (Malignant vs Benign)",
    title_x=0.5,
    template="plotly_white",
    barmode='overlay',
    height=3000,
)


fig.update_xaxes(title_text="Feature Value")
fig.update_yaxes(title_text="Count")

fig.show()
```

Let's now explore an example of multivariate analysis by plotting a correlation heatmap of all our features:

```python
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
corr = df.corr()

plt.figure(figsize=(16, 14))


sns.heatmap(corr, cmap='Blues', annot=False, fmt=".2f",
            square=False, linewidths=.5, cbar_kws={"shrink": .8},
            annot_kws={"size": 10})

plt.title('Correlation Heatmap', fontsize=20)
plt.show()
```
In the above, we see that the function sns.heatmap() has several arguments. One of which is 'annot'. Try setting this to True
and see what it does.

There are different ways of plotting a correlation heatmap:

```python
corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(16, 14))


sns.heatmap(corr, mask=mask, cmap='Blues', annot=True, fmt=".2f",
            square=False, linewidths=.5, cbar_kws={"shrink": .8},
            annot_kws={"size": 10})

plt.title('Correlation Heatmap', fontsize=20)
plt.show()
```

Which one do you prefer?
