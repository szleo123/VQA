## json
json files store the evaluation output in each epoch corresponding to different models

## txt
txt files store the evaluation analysis result in each epoch corresponding to different models

## How to run
After running the models under code directory, those results will automatically be generated.

## What to expect
The overall accuracy and Per Question Type Accuracy in txt will be reported as follows: 
```
Overall Accuracy is: 38.23

Per Question Type Accuracy is the following:
none of the above : 45.63
what are the : 16.37
what is : 12.19
what : 23.04
is this a : 67.48
...
was : 69.83
do : 68.35
how many people are in : 9.18

Per Answer Type Accuracy is the following:
other : 22.89
yes/no : 66.18
number : 15.88
```