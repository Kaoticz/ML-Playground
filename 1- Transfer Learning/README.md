# Transfer Learning

## Summary

In this project, a neural network is trained from scratch with a dataset comprised of pictures of cats and dogs.

Then a pre-trained VGG16 neural network is downloaded and fine-tuned with the same dataset.

Then the performance of both models is compared with each other.

## Running it on Google Colab

Upload the [Transfer_Learning.ipynb](./Transfer_Learning.ipynb) to Google Colab, then click `Runtime > Run all` or press `Ctrl + F9` on your keyboard.

Don't forget to set your runtime to a machine with a GPU, so training doesn't take too long.

## Running it locally

### Prerequisites

- A CUDA-compatible GPU/TPU/NPU card.
- The driver for your respective card.
- Docker or Podman
- Nvidia Container Toolkit ([setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

### Setup

Create the folder where the dataset is going to be stored at:

On Windows (Command Prompt):
```cmd
mkdir data && icacls data /grant:r Everyone:(OI)(CI)F
```

On Unix systems:
```bash
mkdir data && chmod 777 ./data
```

Build the Docker image with the following command:

```bash
docker build -t ml-transfer-learning ./
```

Then create and run a container from this image with the following command:

```bash
docker run --rm --gpus all -v ./data:/home/appuser/app/data ml-transfer-learning
```

If your card has enough VRAM, the models will be trained and used to perform a prediction at the end.

## Reference

https://github.com/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb
