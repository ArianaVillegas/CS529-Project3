# CS529-Project3

Integrantes 

- Ariana Villegas
- Enrique Sobrados


## Models Implemented

- VGG-16
- ResNet50
- InceptionV3 
- EfficientNetV2




## Usage ##

Running the tool:

    python3 main.py [options]

Typical Usage Example:

    python3 main.py --model resnet --mode train --window 224

Options:

  -h, --help            show this help message and exit
  --train-folder TRAIN_FOLDER
                        The relative path to the training dataset folder
  --test-folder TEST_FOLDER
                        The relative path to the testting dataset folder
  --val-prop VAL_PROP   The validation proportion to split train and
                        validation sets
  --model {dummy,resnet,vgg16,inceptionV3,efficientnetV2}
                        Model name
  --mode {opt,train,aug,test}
                        Execution mode: optmization (opt) | training (train) |
                        augmentation (aug) | testing (test)
  --window WINDOW       Window size
  --augmentation AUGMENTATION
                        Data augmentation size per class

	


