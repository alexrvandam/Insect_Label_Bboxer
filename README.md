# Insect_Label_Bboxer
# Insect_Label_Bboxer (ILBb) a custom Detectron2 model followed by bounding box collector and tsv out, video coming soon!

# install miniconda
## follow the instructions to install miniconda on your system here
https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links

## create a conda environment
conda create -n Detectron2_ILBb python=3.9
## activate the environment
conda activate Detectron2_ILBb

# So far only tested on a cpu, but it should work on gpu as well
# There are ton of dependencies but don't worry between pip and conda you shoud be able to install them, I will try to list some of them here:
pip install opencv-python

pip install opencv-python-headless

pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html

pip install torch torchvision torchaudio

pip install spacy

pip install googlemaps

pip install scikit-learn

pip install mmcv

pip install mmocr

pip install Pillow

pip install pytesseract

pip install easydict

pip install scipy

pip install pandas

pip install filterpy

# Many of these are not necessary but may be needed for later implementations of this code as I plan to keep working on it and build upon it, the key ones to install are Detectron2 and cv2 you can also follow there instructions here on how to install Detectron2 if the pip install above did not work:
https://detectron2.readthedocs.io/en/latest/

# Then once you have them installed you need to download the Git or at least the individual files in the same structure so that the program can talk to all of its members

# Then cd into the Insect_label folder
cd Insect_label

# Then you will need to put the images you want to detect, instance segment, and then extract bbox's for in the test folder replace my_images with yours below, cp is a suggestion you could also use mv or just drag and drop them
cp my_images.jpeg test/
# then you can run the code and if all is set up it should work

python Get_label_bbox_plus_IDs.py

## Then checkout your output folder
### if it does now work it is probably an install issue with Detectron2 similar, however there are several file paths within the script to double check
##
### Pleas keep in  mind that this is the first implementation and I will keep upgrading the model and release more robust and general versions at the moment this modle is very preliminary and will likely need a much larger training data set, but it has been tested and works about 83% of the time, so good luck!










