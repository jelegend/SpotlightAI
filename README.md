## SpotlightAI NLP Project : Location Detection

Filename : aayushmani_nlp (.py and .ipynb)

### Importing Packages and Libraries

For this task I originally planned to implent only with spaCy which is primarily designed for practical applications like this. However during implementation I realized that the results obtained were not of very high accuracy (i.e < 80-85%). Thus I decided to augment it with Geograpy (based on NLTK another famous NLP Library) which is a specifically designed for tasks pertaining to geolocation.

```python
import numpy
import csv

import spacy
nlp = spacy.load('en_core_web_sm')

import geograpy
import nltk
nltk.downloader.download('averaged_perceptron_tagger')

from time import perf_counter
```

### Importing Text

For clarity's sake and portatbility of code (work standalon) I decided to lift the sample text instead of opening it from file which may cause issues if not kept in mind

### Logic Implementation

#### (a) Location Identification

First I have used spaCy's NLP capabilties to identify all locations using the GPE (Geo-political entity) and LOC (Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains) tags.

After combining the list obtained through this,  I have further used Geograpy (running NLTK on backend) to comb the text again for any references to locations that spaCy missed out (using cities, regions and others functions)

These 2 combined give a High accuracy algorithm to identify all possible references to locations in the text as there capabilties are complementary to each other for the same.

```python
t_start = perf_counter()

gpe = []

doc = nlp(text)
places = geograpy.get_place_context(text=text)

for ent in doc.ents:
    if (ent.label_ == 'GPE'):
        gpe.append(ent.text)
    elif (ent.label_ == 'LOC'):
        gpe.append(ent.text)
    
for ent in places.cities:
    if (ent not in gpe):
        gpe.append(ent)
    
for ent in places.other:
    if (ent not in gpe):
        gpe.append(ent)
    
for ent in places.regions:
    if (ent not in gpe):
        gpe.append(ent)
```

#### (b) Validation

In an attempt to achieve High accuracy by combining 2 Libraries we tend to over count the possible locations. Since we are aiming for precise identification we need to isolate and remove false positives of the algo. For this I have went for validation via cross-referencing with a certified Database of locations worldwide.

This helps remove whatever little false identifications that could have been made in part (a).

```python
data = []
with open("cities.csv", encoding = "utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
        
arr = numpy.array(data)

final = []

for text in gpe:
    if (text in arr and (text not in final)):
        final.append(text)
        
print(final)
        
t_stop = perf_counter()
```

### Other Possible Methods  

We can use other possible methods like RNN or even NLP Packages with extra training for higher accuracy and lower falso identification rates. However the time and cost of implementation of such methods relative to improvement in accuracies does not justify their in my view for this task. For successful completion via these methods a large amount of data for training would be rquired as well which not only requires effort to acquire but also would need Extra Time to Train the models.

Thus as a compromise between speed and accuracy I have opted for off the shelf solutions and combined them. This in my POV is a healthy combination and further stretching the limits for either parameters would not be advisable.

####  Accuracy, False Positives and Time

This section is plainly designed to check the the above parameters for the given task. The results obtained are :

```bash
['Boston', 'Longmeadow', 'MA', 'Rhode Island', 'Connecticut', 'Brooklyn', 'Chicago', 'San Francisco', 'Cambridge', 'Sugar Land', 'TX', 'NYC']
```
Ideal Case:
```python
answers = ['Indo', 'Boston','Longmeadow', 'MA', 'Rhode Island', 'Connecticut', 'Sugar Land', 'TX', 'Brooklyn', 'NYC', 'Chicago', 'San Francisco', 'Cambridge']
```
Performance of NLP Code :
```bash
Correct: 12
Ideal: 13
Accuracy: 92.3076923076923
False Identification: 0
Time Elapsed during Identification:  2.3060209 sec
```



