# SECAF: A Span-level Emotion Cause Analysis Framework
This repo implements the framework for span-level emotion cause analysis.

## Project Setup
### Conda environment installation
```
conda env create -f environment.yml
conda activate eca
cd model
```

## Run experiment
Adjust the experiment settings and hyper-parameters in `config.py` file. Choose `ECSE`, `EESE` or `ECSP` as the evaluation task.
### Bert-based Emotion Cause Tagging - BECT
- BECT from a document (BECT-doc)
##### Training:
```
python main.py --model_class bert --data_dir ../data --do_train --evaluation_metrics ECSE
```
##### Inference: in `config.py`, set `OUTPUT_DIR` as the model path
```
python main.py --model_class bert --data_dir ../data --do_eval --evaluation_metrics ECSE
```

- BECT from a clause (BECT-clause)
##### Training:
```
python main.py --model_class bert-clause --data_dir ../data --do_train --evaluation_metrics ECSE --task_name eca-clause
```
##### Inference:
```
python main.py --model_class bert-clause --data_dir ../data --do_eval --evaluation_metrics ECSE
```

- Commensense knowledge encoded BECT (Comet-BECT)
##### Training: change `COMET_FILE` in `config.py` to specify the target COMET relations (i.e. *HasSubEvent*, *Causes*, *xReason*, *xEffect* or *xReact*)
```
python main.py --model_class comet-bert --data_dir ../data --do_train --evaluation_metrics ECSE --task_name eca-comet
```
##### Inference: 
```
python main.py --model_class comet-bert --data_dir ../data --do_eval --evaluation_metrics ECSE --task_name eca-comet
```
