# Intent Recognition for Doctor-Patient Interviews

In this paper we used different models based on Deep Learning and Information Retrieval
methods to classify utterances in a medical-interview. The utterance class represent an intent 
of a doctor and shall be used in a simulator. In this simulator, the patients are virtualized and 
the intent class triggers a video clip which contains a prerecorded response of the patient.

The Paper can be found in the Proceedings of LREC 2020 or 
[here](/paper/Intent_Recognition_in_Doctor_Patient_Interviews.pdf)

## Setup

### Setup environment

```bash
conda create -n pytorch python=3.6.7
conda activate pytorch
```

### Install pytorch
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Run experiments
```bash
cd experiments
python neural_classifier_networks.py --model bert --bert_style
python neural_ranker_networks.py --model cnn --bert_style
```

## Results
The results are based on a test set, which is evaluated against after tuning parameters 
and training weights:

Model | MRR | Accuracy
----- | --- | -------
TF/IDF | 66.39 | 54.24
BM25 | 73.25 | 62.15 
TF/IDF+TProbs | 66.37 | 53.67
BM25+TProbs | __73.88__ | 62.15
--- | --- | ---
BERT | 67.08 | 64.08
+Previous Utterance | 65.87 | __71.35__
+Transition Probabilities | 66.02 | 70.31
+Both | 65.29 | 64.58
For each column, the bold percentage is the maximum Accuracy and MRR respectively.

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
