## The official implementation of the paper OAT: Object-Level Attention Transformer for Gaze Scanpath Prediction (accepted in ECCV 2024)

Visual search is important in our daily life. The efficient allocation of visual attention is critical to effectively complete visual search tasks. Prior research has predominantly modelled the spatial allocation of visual attention in images at the pixel level, e.g. using a saliency map. However, emerging evidence shows that visual attention is guided by objects rather than pixel intensities. This paper introduces the Object-level Attention Transformer (OAT), which predicts human scanpaths as they search for a target object within a cluttered scene of distractors. OAT uses an encoder-decoder architecture. The encoder captures information about the position and appearance of the objects within an image and about the target. The decoder predicts the gaze scanpath as a sequence of object fixations, by integrating output features from both the encoder and decoder. We also propose a new positional encoding that better reflects spatial relationships between objects. We evaluated OAT on the Amazon book cover dataset and a new dataset for visual search that we collected. OAT's predicted gaze scanpaths align more closely with human gaze patterns, compared to predictions by algorithms based on spatial attention on both established metrics and a novel behavioural-based metric. Our results demonstrate the generalization ability of OAT, as it accurately predicts human scanpaths for unseen layouts and target objects.

![main2](../../../../../../../Desktop/pics_update/main2.jpeg)

### Download datasets

We used two datasets in our paper, amazon book cover and our collected dataset. You can download both datasets from: https://drive.google.com/drive/folders/1WZnjHhtF15mJ1nuJfwo4DALryaWiFcvJ?usp=sharing. Amazon book cover is also available here: https://github.com/DFKI-Interactive-Machine-Learning/STI-Dataset.

Put the data under ./dataset/

### Preprocess the data

1. Generate random index file by running ./src/preprocess/split.py
2. Generate dataset file by running ./src/preprocess/img_feature.py for our collected dataset or img_feature_amazon.py for amazon book cover dataset.

### Training and testing

Run src/run.py to train the model. Remember to change input argument like the file path. The testing outcome will be printed in the end of the training. Alternatively, you can run ./src/evaluation/evaluation_full.py or ./src/evaluation/MM/calculate_mm.py using the saved outcome csv files.

### Citation

Will update later.

