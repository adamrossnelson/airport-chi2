```python
# Import Pandas, Seaborn, and related
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Set the Seaborn context
sns.set_context('talk')
```

# A Code Along For Chi 2

Working through the Chi 2 analysis step by step with base Python and Numpy only.

<table style="text-align: left; margin-left: 0;">
  <tr><th>Grade</th><th>9th</th><th>10th</th><th>11th</th><th>12th</th></tr>
  <tr><td>A</td><td>60</td><td>70</td><td>80</td><td>90</td></tr>
  <tr><td>B</td><td>50</td><td>60</td><td>70</td><td>80</td></tr>
  <tr><td>C</td><td>40</td><td>50</td><td>60</td><td>70</td></tr>
  <tr><td>D</td><td>30</td><td>40</td><td>50</td><td>60</td></tr>
</table>

<table style="text-align: left; margin-left: 0;">
  <tr><th>
$$
\chi^2 = \sum \frac{(\textcolor{red}{\text{Observed}} - \textcolor{blue}{\text{Expected}})^2}{\textcolor{blue}{\text{Expected}}}
= \sum \frac{(\textcolor{red}{O} - \textcolor{blue}{E})^2}{\textcolor{blue}{E}},
\quad df = (rows - 1) \times (cols - 1)
$$

</table>

Critical thinking questions:

1) Do you suppose that the number of students passing the exam may be a function of grade level?
2) Other than the data shown above, what leads you to that conclusion?
3) How can we test this? What is the corresponding null hypothesis and what is the corresponding alternate?


```python
scores = [
    [60, 70, 80, 90],
    [50, 60, 70, 80],
    [40, 50, 60, 70],
    [30, 40, 50, 60]]

scores = np.array(scores)

scores
```




    array([[60, 70, 80, 90],
           [50, 60, 70, 80],
           [40, 50, 60, 70],
           [30, 40, 50, 60]])




```python
np.zeros(16).reshape(4,4)
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
temp = np.full((4,4), np.nan)
```


```python
temp
```




    array([[nan, nan, nan, nan],
           [nan, nan, nan, nan],
           [nan, nan, nan, nan],
           [nan, nan, nan, nan]])




```python
temp[2,2] = 99
temp
```




    array([[nan, nan, nan, nan],
           [nan, nan, nan, nan],
           [nan, nan, 99., nan],
           [nan, nan, nan, nan]])




```python
scores
```




    array([[60, 70, 80, 90],
           [50, 60, 70, 80],
           [40, 50, 60, 70],
           [30, 40, 50, 60]])



# The Really Really Slow Way


```python
expected = np.zeros(16).reshape(4, 4)
expected
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
# Here is how that would work for each cell
# Row 0
expected[0,0] = (scores[0, :].sum() * scores[:, 0].sum()) / scores.sum()
expected[0,1] = (scores[0, :].sum() * scores[:, 1].sum()) / scores.sum()
expected[0,2] = (scores[0, :].sum() * scores[:, 2].sum()) / scores.sum()
expected[0,3] = (scores[0, :].sum() * scores[:, 3].sum()) / scores.sum()

# Row 1
expected[1,0] = (scores[1, :].sum() * scores[:, 0].sum()) / scores.sum()
expected[1,1] = (scores[1, :].sum() * scores[:, 1].sum()) / scores.sum()
expected[1,2] = (scores[1, :].sum() * scores[:, 2].sum()) / scores.sum()
expected[1,3] = (scores[1, :].sum() * scores[:, 3].sum()) / scores.sum()

# Row 2
expected[2,0] = (scores[2, :].sum() * scores[:, 0].sum()) / scores.sum()
expected[2,1] = (scores[2, :].sum() * scores[:, 1].sum()) / scores.sum()
expected[2,2] = (scores[2, :].sum() * scores[:, 2].sum()) / scores.sum()
expected[2,3] = (scores[2, :].sum() * scores[:, 3].sum()) / scores.sum()

# Row 3
expected[3,0] = (scores[3, :].sum() * scores[:, 0].sum()) / scores.sum()
expected[3,1] = (scores[3, :].sum() * scores[:, 1].sum()) / scores.sum()
expected[3,2] = (scores[3, :].sum() * scores[:, 2].sum()) / scores.sum()
expected[3,3] = (scores[3, :].sum() * scores[:, 3].sum()) / scores.sum()

expected
```




    array([[56.25      , 68.75      , 81.25      , 93.75      ],
           [48.75      , 59.58333333, 70.41666667, 81.25      ],
           [41.25      , 50.41666667, 59.58333333, 68.75      ],
           [33.75      , 41.25      , 48.75      , 56.25      ]])



# A Better Loopy Approach


```python
scores
```




    array([[60, 70, 80, 90],
           [50, 60, 70, 80],
           [40, 50, 60, 70],
           [30, 40, 50, 60]])




```python
expected = np.full((4,4), np.nan)
expected
```




    array([[nan, nan, nan, nan],
           [nan, nan, nan, nan],
           [nan, nan, nan, nan],
           [nan, nan, nan, nan]])




```python

```

    [300 260 220 180]
    [180 220 260 300]



```python
row_totals = scores.sum(axis=1)
col_totals = scores.sum(axis=0)

print(row_totals)
print(col_totals)

for r in range(scores.shape[0]):
    for c in range(scores.shape[1]):
        expected[r, c] = (row_totals[r] * col_totals[c]) / scores.sum()

expected
```




    array([[56.25      , 68.75      , 81.25      , 93.75      ],
           [48.75      , 59.58333333, 70.41666667, 81.25      ],
           [41.25      , 50.41666667, 59.58333333, 68.75      ],
           [33.75      , 41.25      , 48.75      , 56.25      ]])



# Finding The Errors


```python
errors = np.full((4,4), np.nan)

# Your task now alone or small groups... build a nested for loop
# that will poluate errors with the correct values

for r in range(expected.shape[0]):
    for c in range(expected.shape[1]):
        errors[r, c] = ((scores[r, c] - expected[r, c]) ** 2) / expected[r, c]

# for r in range(errors.shape[0]):
#     for c in range(scores.shape[1]):
#         errors[r, c] = (scores[r, c] - expected[r, c]) 

# for r in range(scores.shape[0]):
#     for c in range(scores.shape[1]):
#         errors[r, c] = ((scores[r, c] - expected[r, c]) ** 2) / expected[r, c]
        
# sq_errors = errors ** 2

errors
```




    array([[0.25      , 0.02272727, 0.01923077, 0.15      ],
           [0.03205128, 0.00291375, 0.00246548, 0.01923077],
           [0.03787879, 0.00344353, 0.00291375, 0.02272727],
           [0.41666667, 0.03787879, 0.03205128, 0.25      ]])



**RAW ERRORS**
```
array([[ 3.75      ,  1.25      , -1.25      , -3.75      ],
       [ 1.25      ,  0.41666667, -0.41666667, -1.25      ],
       [-1.25      , -0.41666667,  0.41666667,  1.25      ],
       [-3.75      , -1.25      ,  1.25      ,  3.75      ]])
```
**SQ ERRORS**
```
array([[14.0625    ,  1.5625    ,  1.5625    , 14.0625    ],
       [ 1.5625    ,  0.17361111,  0.17361111,  1.5625    ],
       [ 1.5625    ,  0.17361111,  0.17361111,  1.5625    ],
       [14.0625    ,  1.5625    ,  1.5625    , 14.0625    ]])
```


```python
errors.sum()
```




    1.3021794056759093




```python
from scipy import stats

stats.chi2.sf(errors.sum(), 9)
```




    0.9983655939499505



# Bestish Way


```python
scores = np.array([
    [60, 70, 80, 90],
    [50, 60, 70, 80],
    [40, 50, 60, 70],
    [30, 40, 50, 60]])

scores 
```




    array([[60, 70, 80, 90],
           [50, 60, 70, 80],
           [40, 50, 60, 70],
           [30, 40, 50, 60]])




```python
row_totals = scores.sum(axis=1, keepdims=True)
col_totals = scores.sum(axis=0, keepdims=True)

print(row_totals)
```

    [[300]
     [260]
     [220]
     [180]]



```python
print(col_totals)
```

    [[180 220 260 300]]



```python
row_totals.shape
```




    (4, 1)




```python
expected = row_totals @ col_totals / scores.sum()
expected
```




    array([[56.25      , 68.75      , 81.25      , 93.75      ],
           [48.75      , 59.58333333, 70.41666667, 81.25      ],
           [41.25      , 50.41666667, 59.58333333, 68.75      ],
           [33.75      , 41.25      , 48.75      , 56.25      ]])




```python
errors = ((scores - expected) ** 2) / expected
errors
```




    array([[0.25      , 0.02272727, 0.01923077, 0.15      ],
           [0.03205128, 0.00291375, 0.00246548, 0.01923077],
           [0.03787879, 0.00344353, 0.00291375, 0.02272727],
           [0.41666667, 0.03787879, 0.03205128, 0.25      ]])




```python
stats.chi2.sf(errors.sum(), 9)
```




    0.9983655939499505




```python
from scipy.stats import chi2_contingency
```


```python
for i in chi2_contingency(scores):
    print(i)
```

    1.3021794056759093
    0.9983655939499505
    9
    [[56.25       68.75       81.25       93.75      ]
     [48.75       59.58333333 70.41666667 81.25      ]
     [41.25       50.41666667 59.58333333 68.75      ]
     [33.75       41.25       48.75       56.25      ]]



```python
df = sns.load_dataset('titanic')
```


```python
df.sample(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>702</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>18.0</td>
      <td>0</td>
      <td>1</td>
      <td>14.4542</td>
      <td>C</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>769</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.3625</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.array(pd.crosstab(df['sex'], df['survived']))
```




    array([[ 81, 233],
           [468, 109]])




```python
# Ho : Survival is not a funciton of sex
# Ha : Survival is a function of sex
```


```python
for i in chi2_contingency(pd.crosstab(df['sex'], df['survived'])):
    print(i)
```

    260.71702016732104
    1.197357062775565e-58
    1
    [[193.47474747 120.52525253]
     [355.52525253 221.47474747]]



```python
df.sample(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>459</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>397</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>46.0</td>
      <td>0</td>
      <td>0</td>
      <td>26.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>870</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
