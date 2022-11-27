# FeatureExtraction Model for Highlight Classification

## Installation

- conda environment setting
```
conda create -n LSMMA python=3.7 -y
conda activate LSMMA
```
- nessary packages
```
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn
```


## Feature Extraction

```
python HighlightClassification/FeatureExtraction/main_featureExtract.py
```
It saves the extracted features from pre-trained model.

By default, extracted features are stored under `data/model_features/`.


## Model Training

```
python HighlightClassification/FeatureExtraction/main.py
```
It train the highlightclassification model and save the result.

By default, model are stored under `models/`.


## Code Reference

It is based on(https://github.com/SilvioGiancola/SoccerNetv2-DevKit)

## Paper citation

```
@InProceedings{Deliège2020SoccerNetv2,
      title={SoccerNet-v2 : A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos}, 
      author={Adrien Deliège and Anthony Cioppa and Silvio Giancola and Meisam J. Seikavandi and Jacob V. Dueholm and Kamal Nasrollahi and Bernard Ghanem and Thomas B. Moeslund and Marc Van Droogenbroeck},
      year={2021},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      month = {June},
}
```
