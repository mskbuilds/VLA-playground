# VLA-playground

## Running the container with GPU support

To run the Docker image with GPU access enabled, use:

```bash
docker run --gpus all -it myimage
```

## To get the pytorch in the docker

To get the pytorch in the docker, build the pytorch docker
```bash
sudo docker build -f pytorch_dockerfile -t pytorch-build .
```

Then extract wheel

```bash
docker create --name tmp pytorch-build
sudo docker cp tmp:/output ./
docker rm tmp
```

Expect something like torch-2.2.0-cp310-cp310-linux_aarch64.whl

To install in system, just use pip

```bash
pip3 install torch-2.2.0-cp310-cp310-linux_aarch64.whl
```

##Points to remember

These are points to remember especially around the environment variables
disable NCCL + MKLDNN (Set them to 0)
Cuda versioin for jetson orin = 8.7