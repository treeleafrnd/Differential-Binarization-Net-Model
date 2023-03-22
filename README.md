# DIFFERENTIAL BINARIZATION NETWORK MODEL
Differential Binarization Network (DBNet) Model for scene text detection.

https://github.com/MhLiao/DB

Differential Binarization Network uses known convolutional networks such as ResNet-18 or ResNet-50 to extract text location. This extraction is a prediction made by the network. In order to encompass all the texts in the scene regardless of their size, a Adaptive Scale Fusion (ASF) is used. Finally a binarization is used which improves the accuracy and performance. 

# INTRODUCTION
This is a PyToch implementation of DBNet(arxiv) and DBNet++(TPAMI, arxiv). It presents a real-time arbitrary-shape scene text detector, achieving the state-of-the-art performance on standard benchmarks.

Part of the code is inherited from MegReader.

# Installation:

# Requirements:

    Python3
    PyTorch == 1.2
    GCC >= 4.9 (This is important for PyTorch)
    CUDA >= 9.0 (10.1 is recommended)

# first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name DB -y
  conda activate DB

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install -r requirement.txt

  # install PyTorch with cuda-10.1
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

  # clone repo
  git clone https://github.com/MhLiao/DB.git
  cd DB/

  # build deformable convolution opertor
  # make sure your cuda path of $CUDA_HOME is the same version as your cuda in PyTorch
  # make sure GCC >= 4.9
  # you need to delete the build directory before you re-build it.
  echo $CUDA_HOME
  cd assets/ops/dcn/
  python setup.py build_ext --inplace
  
# ALL OF THE PREREQUISITS ARE IMPLEMENTED IN DBNet.ipynb

# MODEL
  totaltext_resnet50
  
# RESULTS

  Dataset: Good Notes Handwritten Kollection (GNHK)
  
  ![gnhk5](https://user-images.githubusercontent.com/99968233/226842188-9e1c3e84-e30f-489a-ad53-71097fc0760d.jpg)
![gnhk14](https://user-images.githubusercontent.com/99968233/226842206-dee3c323-e7ec-477b-b817-dbef3a53d9c1.jpg)
![gnhk18](https://user-images.githubusercontent.com/99968233/226842226-3930a530-8618-42d9-921e-051ab96a530c.jpg)
![gnhk22](https://user-images.githubusercontent.com/99968233/226842237-40eb94f4-ad0a-4070-9e2b-10eee41c3bfc.jpg)
![gnhk24](https://user-images.githubusercontent.com/99968233/226842243-c12fb918-9955-4b14-83c7-2b1382f192c8.jpg)
![gnhk25](https://user-images.githubusercontent.com/99968233/226842249-3903c835-137e-4127-973c-e7cca2e3afb7.jpg)
![gnhk30](https://user-images.githubusercontent.com/99968233/226842257-890ecc13-e88d-4f7e-9783-a805a15428b8.jpg)
![gnhk31](https://user-images.githubusercontent.com/99968233/226842265-3ce43832-dafd-41f6-bb95-9395a58df5e8.jpg)
![gnhk38](https://user-images.githubusercontent.com/99968233/226842272-8d58ed1a-db73-48c6-9339-997cbeaff079.jpg)
![gnhk56](https://user-images.githubusercontent.com/99968233/226842276-434e10d4-a2fd-41d3-80ea-5d1c5063442e.jpg)
![gnhk57](https://user-images.githubusercontent.com/99968233/226842283-cc0f5437-fc7d-453f-8d8f-4bad00509594.jpg)
![gnhk65](https://user-images.githubusercontent.com/99968233/226842293-f5e89ff2-7705-4ac7-9acc-830b9d4398ca.jpg)
![gnhk70](https://user-images.githubusercontent.com/99968233/226842299-5567a18f-9533-442c-a7f1-2f3fb7e20056.jpg)
![gnhk81](https://user-images.githubusercontent.com/99968233/226842306-15b0d4ee-b05c-4747-9ee1-2b6ccc6a813c.jpg)
![gnhk84](https://user-images.githubusercontent.com/99968233/226842309-8b9c2ccd-ee52-42e8-bf6c-b17641a55567.jpg)
![gnhk87](https://user-images.githubusercontent.com/99968233/226842315-847e8d91-8c6e-478b-b63a-26640b306f6f.jpg)
![gnhk94](https://user-images.githubusercontent.com/99968233/226842321-ae8fe8c6-859f-46bd-92ad-375929ae7cdc.jpg)

