# Reference
https://github.com/endgameinc/ember  

H. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models”, in ArXiv e-prints. Apr. 2018.  

```
@ARTICLE{2018arXiv180404637A,  
  author = {{Anderson}, H.~S. and {Roth}, P.},  
  title = "{EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models}",  
  journal = {ArXiv e-prints},  
  archivePrefix = "arXiv",  
  eprint = {1804.04637},  
  primaryClass = "cs.CR",  
  keywords = {Computer Science - Cryptography and Security},  
  year = 2018,  
  month = apr,  
  adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180404637A},  
}  
```  

# Install
above python 3.5    
```
;install virtualenv
$ virtualenv emberenv -p python3
$ . ./emberenv/bin/activate
```
  
```
;install python modules
(emberenv)$ pip install -r requirements.txt
(emberenv)$ pip install -U scikit-learn
```

# inputfile(csv including label) structure
without column's names
![traindata_label](screenshot/traindata_label.png)

# Summary of run
```
  sript/01_extract.sh -> script/02_learn.sh -> script/03_predict.sh -> script/04_get_accuracy.sh
```  
  or    

```
  script/run.sh
```