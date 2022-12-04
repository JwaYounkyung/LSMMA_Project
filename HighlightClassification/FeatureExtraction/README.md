# FeatureExtraction Model for Highlight Classification

## Installation

- conda environment setting
```
conda create -n LSMMA python=3.7 -y
conda activate LSMMA
```
- nessary packages
```
conda install pytorch torchvision torchaudio -c pytorch -y (install pytorch 1.13.0 with your condition)
pip install tensorflow==2.3.0
pip install scikit-video imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn torchsummary wandb
```
## Data Download
```
python Feature_Download.py --dir data/features # non pca version
python Feature_Download.py --dir data/features --pca # pca version
```
It saves the feature extracted from ResNet50.

By default, extracted features are stored under `data/features/`.

None pca version is about 40GB then, we only use pca version(10GB).


## Feature Extraction
```
python HighlightClassification/FeatureExtraction/main_featureExtract.py --extract_mode train
```
It saves the extracted features from pre-trained model.

You need to adjust the extract mode to train or val or test.

By default, extracted features are stored under `data/model_features/`.


## Model Training

```
python HighlightClassification/FeatureExtraction/main.py
```
It train the highlightclassification model and save the result.

You can adjust the hyperparameter in the main.py.

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
