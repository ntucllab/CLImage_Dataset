# CLCIFAR Dataset

The dataset repo of "CLCIFAR: CIFAR-Derived Benchmark Datasets with Human Annotated Complementary Labels"

## Abstract
This repo contains two datasets: CLCIFAR10 and CLCIFAR20 with human annotated complementary labels for complementary label learning tasks.

TL;DR: the download links to CLCIFAR dataset
* CLCIFAR10: [clcifar10.pkl](https://clcifar.s3.us-west-2.amazonaws.com/clcifar10.pkl) (148MB)
* CLCIFAR20: [clcifar20.pkl](https://clcifar.s3.us-west-2.amazonaws.com/clcifar20.pkl) (151MB)

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

* `names`: The list of filenames strings. This filenames are same as the ones in CIFAR10

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

### Reference

[[1]](https://arxiv.org/abs/2110.12088) Jiaheng Wei, Zhaowei Zhu, and Hao Cheng. Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations. arXiv preprint arXiv:2110.12088, 2021.
