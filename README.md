# A Learning-based Quadcopter Controller with Extreme Adaptation
This repo contains the code associated with the paper *A Learning-based Quadcopter Controller with Extreme Adaptation*

[![Cover](media/Cover.jpg)](https://youtu.be/kZEU8lxMZug?si=Y8grEiGLXqEeb2c6)

#### Paper and Video

If you use this code in an academic context, please cite the following publication:

Paper: [ A Learning-based Quadcopter Controller with Extreme Adaptation](https://arxiv.org/abs/2409.12949) 

Video: [YouTube](https://youtu.be/kZEU8lxMZug?si=Y8grEiGLXqEeb2c6)


```
@misc{zhang2024learningbasedquadcoptercontrollerextreme,
      title={A Learning-based Quadcopter Controller with Extreme Adaptation}, 
      author={Dingqi Zhang and Antonio Loquercio and Jerry Tang and Ting-Hao Wang and Jitendra Malik and Mark W. Mueller},
      year={2024},
      eprint={2409.12949},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.12949}, 
}
```


### Usage

We can run the following scipts to see the simulation of trajectory tracking with our controller.
```
bash setup.bash
```

### Docker Setup

If your operating system doesn't support environment of this project, docker is a great alternative.

First of all, you have to build the project and create an  image like so:

```bash
## Assuimg you are in the correct project directory
docker build -t xadaptctrl .
```
To use a shortcut, you may use the following command:

```bash
## Assuimg you are in the correct project directory
make docker_build
```


After the image is created, copy and paste the following command to the terminal to run the image:

```bash
## Assuimg you are in the correct project directory
xhost +
docker run -it --network host   --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --device=/dev/video0:/dev/video0 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --entrypoint /bin/bash xadaptctrl
```
> **_NOTE:_**  You only have to run xhost + once each time you log into the machine. These settings persist per login session.
To use a shortcut, you may use following command:

```bash
make docker_run
```
#### Running the package at docker image

If you are in the docker image , this project is already sourced and the project can be run as the following command;

```bash
python simulate.py 
```