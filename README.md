# Augment data used in NER (Named Entity Recognition) task.

## Table of Contents

[comment]: <> (- [Security]&#40;#security&#41;)
* [Functions](#Functions)
* [Usage](#Usage)
* [Notes](#Notes)


[comment]: <> (- [Usage]&#40;#usage&#41;)
[comment]: <> (- [API]&#40;#api&#41;)
[comment]: <> (## Security)

## Functions
Currently, we provide pre-processing steps including:
1. replace medical entities with synonyms
2. replace adjectives of medical entities with synonyms
3. replace medical entities with English / Chinese
4. replace medical entities with paraphase 
5. randomly delete words
6. randomly swap words

## Usage
1. Put your own data files in Data folder 
2. Put source files in corresponding folders
3. Specify parameters like seed... 
4. Execute command like:
python DataProcessor.py --data_file "Data/..." --save_path "Data/..."
5. You can add your own augmentation steps into this project

## Notes
1. This script is used only in NER task, to preprocess the train/dev/test data.
2. The data files should be in json format, an example is as:
[
  {
    "text": "（5）房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。",
    "entities": [
      {
        "start_idx": 3,
        "end_idx": 7,
        "type": "pro",
        "entity": "房室结消融"
      },
      {
        "start_idx": 9,
        "end_idx": 13,
        "type": "pro",
        "entity": "起搏器植入"
      },
      {
        "start_idx": 16,
        "end_idx": 33,
        "type": "dis",
        "entity": "反复发作或难治性心房内折返性心动过速"
      }
    ]
  },
  {
    "text": "（6）发作一次伴血流动力学损害的室性心动过速（ventriculartachycardia），可接受导管消融者。",
    "entities": [
      {
        "start_idx": 8,
        "end_idx": 21,
        "type": "dis",
        "entity": "血流动力学损害的室性心动过速"
      },
      {
        "start_idx": 23,
        "end_idx": 44,
        "type": "dis",
        "entity": "ventriculartachycardia"
      },
      {
        "start_idx": 50,
        "end_idx": 53,
        "type": "pro",
        "entity": "导管消融"
      }
    ]
  }
  ]
You need to convert the format firstly to follow the above examples.
