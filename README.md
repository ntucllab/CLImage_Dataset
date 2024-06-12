# CLImage Dataset

The dataset repo of "CLImage: Human-Annotated Datasets for Complementary-Label Learning"

## Abstract
This repo contains four datasets: CLCIFAR10, CLCIFAR20, CLMicroImageNet10, and CLMicroImageNet20 with human annotated complementary labels for complementary label learning tasks.

TL;DR: the download links to CLCIFAR and CLMicroImageNet dataset
* CLCIFAR10: [clcifar10.pkl](https://drive.google.com/file/d/1uNLqmRUkHzZGiSsCtV2-fHoDbtKPnVt2/view?usp=sharing) (148MB)
* CLCIFAR20: [clcifar20.pkl](https://drive.google.com/file/d/1PhZsyoi1dAHDGlmB4QIJvDHLf_JBsFeP/view?usp=sharing) (151MB)
* CLMicroImageNet10 Train: [clmicro_imagenet10_train.pkl](https://drive.google.com/file/d/1k02mwMpnBUM9de7TiJLBaCuS8myGuYFx/view?usp=sharing) (55MB)
* CLMicroImageNet10 Test: [clmicro_imagenet10_test.pkl](https://drive.google.com/file/d/1e8fZN8swbg9wc6BSOC0A5KHIqCY2C7me/view?usp=sharing) (6MB)
* CLMicroImageNet20 Train: [clmicro_imagenet20_train.pkl](https://drive.google.com/file/d/1Urdxs_QTxbb1gDBpmjP09Q35btckI3_d/view?usp=sharing) (119MB)
* CLMicroImageNet20 Test: [clmicro_imagenet20_test.pkl](https://drive.google.com/file/d/1EdBCrifSrIIUg1ioPWA-ZLEHO53P4NPl/view?usp=sharing) (11MB)

## Reproduce Code

The python version should be 3.8.10 or above.

```bash
pip3 install -r requirement.txt
bash run.sh
```

## CLCIFAR10

This Complementary labeled CIFAR10 dataset contains 3 human annotated complementary labels for all 50000 images in the training split of CIFAR10. The workers are from Amazon Mechanical Turk(https://www.mturk.com). We randomly sampled 4 different labels for 3 different annotators, so each image would have 3 (probably repeated) complementary labels.

For more details, please visit our paper at link.

### Dataset Structure

Dataset download link: [clcifar10.pkl](https://clcifar.s3.us-west-2.amazonaws.com/clcifar10.pkl) (148MB)

We use `pickle` package to save and load the dataset objects. Use the function `pickle.load` to load the dataset dictionary object `data` in Python.

```python
data = pickle.load(open("clcifar10.pkl", "rb"))
# keys of data: 'names', 'images', 'ord_labels', 'cl_labels'
```

`data` would be a dictionary object with four keys: `names`, `images`, `ord_labels`, `cl_labels`.

* `names`: The list of filenames strings. This filenames are same as the ones in CIFAR10

* `images`: A `numpy.ndarray` of size (32, 32, 3) representing the image data with 3 channels, 32*32 resolution.

* `ord_labels`: The ordinary labels of the images, and they are labeled from 0 to 9 as follows:

  0: airplane
  1: automobile
  2: bird
  3: cat
  4: deer
  5: dog
  6: frog
  7: horse
  8: ship
  9: truck

* `cl_labels`: Three complementary labels for each image from three different workers.

### HIT Design

Human Intelligence Task (HIT) is the unit of works in Amazon mTurk. We have several designs to make the submission page friendly:

* Enlarge the tiny 32\*32 pixels images to 200\*200 pixels for clarity.

![](https://i.imgur.com/SGVCVXV.mp4)

## CLCIFAR20

This Complementary labeled CIFAR100 dataset contains 3 human annotated complementary labels for all 50000 images in the training split of CIFAR100. We group 4-6 categories as a superclass according to [[1]](https://arxiv.org/abs/2110.12088) and collect the complementary labels of these 20 superclasses. The workers are from Amazon Mechanical Turk(https://www.mturk.com). We randomly sampled 4 different labels for 3 different annotators, so each image would have 3 (probably repeated) complementary labels.

### Dataset Structure

Dataset download link: [clcifar20.pkl](https://clcifar.s3.us-west-2.amazonaws.com/clcifar20.pkl) (151MB)

We use `pickle` package to save and load the dataset objects. Use the function `pickle.load` to load the dataset dictionary object `data` in Python.

```python
data = pickle.load(open("clcifar20.pkl", "rb"))
# keys of data: 'names', 'images', 'ord_labels', 'cl_labels'
```

`data` would be a dictionary object with four keys: `names`, `images`, `ord_labels`, `cl_labels`.

* `names`: The list of filenames strings. This filenames are same as the ones in CIFAR20

* `images`: A `numpy.ndarray` of size (32, 32, 3) representing the image data with 3 channels, 32*32 resolution.

* `ord_labels`: The ordinary labels of the images, and they are labeled from 0 to 19 as follows:

  0: aquatic_mammals
  1: fish
  2: flowers
  3: food_containers
  4: fruit, vegetables and mushrooms
  5: household electrical devices
  6: household furniture
  7: insects
  8: large carnivores and bear
  9: large man-made outdoor things
  10: large natural outdoor scenes
  11: large omnivores and herbivores
  12: medium-sized mammals
  13: non-insect invertebrates
  14: people
  15: reptiles
  16: small mammals
  17: trees
  18: transportation vehicles
  19: non-transportation vehicles

* `cl_labels`: Three complementary labels for each image from three different workers.

### HIT Design

Human Intelligence Task (HIT) is the unit of works in Amazon mTurk. We have several designs to make the submission page friendly:

* Hyperlink to all the 10 problems that decrease the scrolling time
* Example images of the superclasses for better understanding of the categories
* Enlarge the tiny 32\*32 pixels images to 200\*200 pixels for clarity.

![](https://i.imgur.com/wg5pV2S.mp4)

## CLMicroImageNet10

This Complementary labeled MicroImageNet10 dataset contains 3 human annotated complementary labels for all 5000 images in the training split of TinyImageNet200. The workers are from Amazon Mechanical Turk(https://www.mturk.com). We randomly sampled 4 different labels for 3 different annotators, so each image would have 3 (probably repeated) complementary labels.

For more details, please visit our paper at link.

### Dataset Structure

Training set download link: [clmicro_imagenet10_train.pkl](https://drive.google.com/file/d/1k02mwMpnBUM9de7TiJLBaCuS8myGuYFx/view?usp=sharing) (55MB)

Testing set download link: [clmicro_imagenet10_test.pkl](https://drive.google.com/file/d/1e8fZN8swbg9wc6BSOC0A5KHIqCY2C7me/view?usp=sharing) (6MB)

We use `pickle` package to save and load the dataset objects. Use the function `pickle.load` to load the dataset dictionary object `data` in Python.

```python
data = pickle.load(open("clmicro_imagenet10_train.pkl", "rb"))
# keys of data: 'names', 'images', 'ord_labels', 'cl_labels'
```

`data` would be a dictionary object with four keys: `names`, `images`, `ord_labels`, `cl_labels`.

* `names`: The list of filenames strings. This filenames are same as the ones in MicroImageNet10

* `images`: A `numpy.ndarray` of size (32, 32, 3) representing the image data with 3 channels, 32*32 resolution.

* `ord_labels`: The ordinary labels of the images, and they are labeled from 0 to 9 as follows:

  0: sulphur-butterfly
  1: backpack
  2: cardigan
  3: kimono
  4: magnetic-compass
  5: oboe
  6: scandal
  7: torch
  8: pizza
  9: alp

* `cl_labels`: Three complementary labels for each image from three different workers.

### HIT Design

Human Intelligence Task (HIT) is the unit of works in Amazon mTurk. We have several designs to make the submission page friendly:

* Enlarge the tiny 32\*32 pixels images to 200\*200 pixels for clarity.

## CLMicroImageNet20

This Complementary labeled MicroImageNet20 dataset contains 3 human annotated complementary labels for all 10000 images in the training split of TinyImageNet200. The workers are from Amazon Mechanical Turk(https://www.mturk.com). We randomly sampled 4 different labels for 3 different annotators, so each image would have 3 (probably repeated) complementary labels.

For more details, please visit our paper at link.

### Dataset Structure

Training set download link: [clmicro_imagenet20_train.pkl](https://drive.google.com/file/d/1Urdxs_QTxbb1gDBpmjP09Q35btckI3_d/view?usp=sharing) (119MB)

Testing set download link: [clmicro_imagenet20_test.pkl](https://drive.google.com/file/d/1EdBCrifSrIIUg1ioPWA-ZLEHO53P4NPl/view?usp=sharing) (11MB)

We use `pickle` package to save and load the dataset objects. Use the function `pickle.load` to load the dataset dictionary object `data` in Python.

```python
data = pickle.load(open("clmicro_imagenet20_train.pkl", "rb"))
# keys of data: 'names', 'images', 'ord_labels', 'cl_labels'
```

`data` would be a dictionary object with four keys: `names`, `images`, `ord_labels`, `cl_labels`.

* `names`: The list of filenames strings. This filenames are same as the ones in MicroImageNet20

* `images`: A `numpy.ndarray` of size (32, 32, 3) representing the image data with 3 channels, 32*32 resolution.

* `ord_labels`: The ordinary labels of the images, and they are labeled from 0 to 19 as follows:

  0: tailed frog
  1: scorpion
  2: snail
  3: american lobster
  4: tabby
  5: persian cat
  6: gazelle
  7: chimpanzee
  8: bannister
  9: barrel
  10: christmas stocking
  11: gasmask
  12: hourglass
  13: iPod
  14: scoreboard
  15: snorkel
  16: suspension bridge
  17: torch
  18: tractor
  19: triumphal arch

* `cl_labels`: Three complementary labels for each image from three different workers.

### HIT Design

Human Intelligence Task (HIT) is the unit of works in Amazon mTurk. We have several designs to make the submission page friendly:

* Enlarge the tiny 32\*32 pixels images to 200\*200 pixels for clarity.

### Reference

[[1]](https://arxiv.org/abs/2110.12088) Jiaheng Wei, Zhaowei Zhu, and Hao Cheng. Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations. arXiv preprint arXiv:2110.12088, 2021.
