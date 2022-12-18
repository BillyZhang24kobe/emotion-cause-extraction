# Generating Commonsense Enhanced Natural Language Explanations for Span-level Emotion Cause Analysis
This repo implements the framework for span-level emotion cause analysis, along with the implementations to generate commonsense enhanced explanations with COMET-based, GLUCOSE-based and GPT-3 prompting methods.

## Project Setup
### Conda environment installation
```
conda env create -f environment.yml
conda activate eca
cd model
```

## Data Preprocessing
The raw data is stored in the directory `raw-data/NTCIR-ECA13-3000`. For the scope of our project, we mainly use the English data. The data preprocessing is separated into two stages. The first stage convert the raw data from `.xml` format to `.tsv` format. To run the first stage, please refer to the Jupyter Notebook `preprocess_stage1.ipynb`. In the second stage, the generated train and test `.tsv` files are cleaned so that all noises (e.g. duplicated data entries) are removed. Relevant codes for cleaning are in `preprocess_stage2.ipynb`.

## COMET-based Inference
We employ COMET-ATOMIC-BART model introduced in [(Comet-) Atomic 2020: On Symbolic and Neural Commonsense Knowledge Graphs](https://www.semanticscholar.org/paper/COMET-ATOMIC-2020%3A-On-Symbolic-and-Neural-Knowledge-Hwang-Bhagavatula/e39503e01ebb108c6773948a24ca798cd444eb62) to generate commonsense explanations. To run an example demo, please follow the steps below: 
```
cd model/COMET
python3 demo_example.py
```
To generate explanations for data in train, dev and test set, please run the following command in the same directory (i.e. `model/COMET`):
```
python3 generation_comet.py
```
The resulting generation results are saved in `data` directory, which will be used as augmented inputs to downstream tasks.

## GLUCOSE-based Inference
We adopted the T5 model fine-tuned on [GLUCOSE](https://arxiv.org/abs/2009.07758) dataset (denoted as `t5-glucose`) to generate commonsense inferences on unseen stories that match human's mental models. To directly run inferences, the original dataset needs to be processed in a way that `t5-glucose` model accepts. Please refer to `preprocess-glucose.ipynb` in `model/GLUCOSE/t5-glucose` directory.

After preprocessing, run the following script in the same directory to obtain GLUCOSE-based generations for each clause in the dataset:
```
bash predict.sh
```
With the predicted outputs, refer to `concat_generations.ipynb` to properly structure the augmented data that will be used for downstream tasks. The resulting data will be stored in `model/GLUCOSE/t5-glucose/data` directory.

## GPT-3 Prompting
To prompt GPT-3 with our curated few-shot examples, we refer to OpenAI [API](https://openai.com/api/). We used `text-davinci-002` as our main model. To generate the dataset with GPT-3 annotations, refer to the following steps:
```
cd model/GPT3/
python3 gpt3_prompting.py
```
The resulting annotations are saved in `data` directory in `model/GPT3`.

## Downstream Tasks Experiments
Make sure you are in the `model` directory before proceeding to other sections.
Adjust the experiment settings and hyper-parameters in `config.py` file. Choose `ECSE`, `EESE` or `ECSP` as the evaluation task. 
### Vanila BECT model 
#### Training:
```
python main.py --model_class bert-clause --data_dir ../data --do_train --evaluation_metrics ECSE --task_name eca-clause
```
#### Inference:
```
python main.py --model_class bert-clause --data_dir ../data --do_eval --evaluation_metrics ECSE
```

### COMET-based BECT model
#### Training: change `COMET_FILE` in `config.py` to specify the target COMET relations (i.e. *HasSubEvent*, *Causes*, *xReason*, *xEffect* or *xReact*)
```
python main.py --model_class comet-bert --data_dir ../data --do_train --evaluation_metrics ECSE --task_name eca-comet
```
#### Inference: 
```
python main.py --model_class comet-bert --data_dir ../data --do_eval --evaluation_metrics ECSE --task_name eca-comet
```

### GLUCOSE-based BECT model
#### Training: change `GLUCOSE_FILE` in `config.py` to specify the target GLUCOSE dimensions (i.e. *dim1*, *dim2*, *dim6*, *dim7*).
```
python main.py --model_class glucose-bert --data_dir ./GLUCOSE/t5-glucose/data --do_train --evaluation_metrics ECSE --task_name eca-comet
```
#### Inference: 
```
python main.py --model_class glucose-bert --data_dir ./GLUCOSE/t5-glucose/data --do_eval --evaluation_metrics ECSE --task_name eca-comet
```

### GPT-3 based BECT model
#### Training: change `GPT3_SHOT_TYPE` in `config.py` to specify the target set of examples for prompting (i.e. *TRS-2*, *TRS-4*).
```
python main.py --model_class bert-gpt3 --data_dir ./GPT3/data --do_train --evaluation_metrics ECSE --task_name eca-comet
```
#### Inference: 
```
python main.py --model_class bert-gpt3 --data_dir ./GPT3/data --do_eval --evaluation_metrics ECSE --task_name eca-comet
```