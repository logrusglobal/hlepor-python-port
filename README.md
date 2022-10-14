Library to calculate hLEPOR score (harmonic mean of enhanced Length Penalty, Precision, n-gram Position difference Penalty and Recall) has been created as port from Perl on materials of the following atricle by Aaron Li-Feng Han, Derek F. Wong, Lidia S. Chao, Liangye He Yi Lu, Junwen Xing, and Xiaodong Zeng. 2013. ["Language-independent Model for Machine Translation Evaluation with Reinforced Factors"](https://www.researchgate.net/profile/Aaron-L-F-Han/publication/256460090_MT_SUMMIT13Language-independent_Model_for_Machine_Translation_Evaluation_with_Reinforced_Factors/links/00463522d48942210c000000/MT-SUMMIT13Language-independent-Model-for-Machine-Translation-Evaluation-with-Reinforced-Factors.pdf). In Proceedings of the XIV Machine Translation Summit.

All hLepor score calculation functions take mandatory and optional parameters for input; mandatory parameters are: ```reference``` (ideal translation), ```hypothesis``` - new translation which has to be compared with reference.

Optional parameters are: 

- ```preprocess``` is a function to preprocess strings, default is ```str.lower()```. 

- ```separate_punctuation``` allows different tokenization options: by default standard ```word_tokenize()``` function from ```nltk.tokenize``` is used, for this option you can specify the language (default is English), if  ```separate_punctuation = False```, sentence is tikenized by spaces.

Other optional parameters control hLepor algorithm:
- ```alpha``` and ```beta``` -- recall and  precision weights, respectively, to calculate weighted Harmonic mean of precision and recall;
- ```n``` -- number of words in vicinity of current word in N-gram word alignment algorithm;
- ```weight_elp, weight_pos, weight_pr``` -- weigths for enhanced length penalty, N-gram Position Difference Penalty and weighted Harmonic mean of precision and recall for   hLepor calculation.

Main functions:
1. To calculate hLepor on one pair of sentences you need to pass these strings to single_hlepor_score function:
```
reference = 'Rising urban populations around the world have ushered in the concept of Smart Cities, in which digital innovations are used to address long-standing urban challenges.'
hypothesis = 'Rising urban populations around the world introduced the concept of Smart Cities, where long-standing urban challenges are addressed with digital innovations.'
hLepor_value = single_hlepor_score(reference, hypothesis)
round(hLepor_value, 4)
```
The result is 0.7550.

2. To calculate hLepor on a set of sentences they need to be passed to hLepor as a list of strings:
```
reference = ['Rising urban populations around the world have ushered in the concept of Smart Cities, in which digital innovations are used to address long-standing urban challenges.', 'The related construction boom has put increasing demands on builders to erect structures and systems beautifully and efficiently.']
hypothesis = ['Rising urban populations around the world introduced the concept of Smart Cities, where long-standing urban challenges are addressed with digital innovations.', 'The related construction boom has put increasing demands on teams to build structures and systems beautifully and efficiently.']
hLepor_value = hlepor_score(reference, hypothesis)
round(hLepor_value, 4)
```
This code will calculate hLepor on each pair of sentences and mean value will be calculated, the result should be 0.8395.
