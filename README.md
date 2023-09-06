# face_detection

This project was made to learn and practice some machine learning, data procesing and data pipelines. 

What was used:
- model:
  - keras
  - VGG16 (pre trained nn model to detect objects)
- dataset:
  - 100 images from the internet (manually classified with 'labelme')
  - data pipeline to create X times more input images (albumentations was used)

## Acknowledgements

 - [VGG16 with keras](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c)
 - [labelme](https://www.v7labs.com/blog/labelme-guide)
 - [albumentations documentation](https://albumentations.ai/docs/)

## Setup

You can create new virtual environment in folder with all project files by typing in terminal

```
python -m venv venv
```

and activate it using command:

- for Windows

```
venv\Scripts\actibate
```

- for other OSes

```
source venv/bin/activate
```

To run the project you need to install required packages, which are included in requirements.txt file

```
pip install -r requirements.txt
```
## Launch

To launch the project type command

```
python main.py
```

If there is a problem with opencv lib, try deactivate your antivirus (and run everything once again)

## Controls

To shut down webcam just press 'q'.

## Screenshots

https://github.com/mkaniow/face_detection/assets/117663140/59280080-1ec3-4a41-8aab-0fb51a244285
