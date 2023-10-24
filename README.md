# RVA Project 1 - Augmented Sword

The main objective of this project is to develop an application that augments an image of a person wielding a reference marker by rendering a virtual sword associated with the marker. The marker may be a planar piece of wood or hard cardboard with two patterns, one on each face of the planar object. Alternatively, you may develop an improved version of the application, using a cube with patterns on five of its faces (on all faces except the one that is facing the user’s hand).



<table>
   <tr>
    <th>Planar Marker Virtual Sword</th>
    <th><img src="https://github.com/FabioMiguel2000/Augmented-Sword/blob/main/img/PlanarMarker.png" alt="Planar Marker"></th>
  </tr>
  <tr>
    <th>Cube Marker Virtual Sword</th>
    <th><img src="https://github.com/FabioMiguel2000/Augmented-Sword/blob/main/img/CubeMarker.png" alt="Planar Marker" ></th>
  </tr>
</table>

## Installation and prerequisites

- Install Python3, see [official website](https://www.python.org/downloads/)
- It is recommended to run in a `conda environment`, our advice is to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- To setup the project, you need the following packages:
    - opencv-python
    - numpy
    - opencv-contrib-python

### Conda environment setup
1. After installing Miniconda, create a conda environment:
```shell
conda create -n augmented_sword python=3.11
```
2. Activate the environment:
```shell
conda activate augmented_sword
```
3. Install the necessary libraries: 
```shell
pip install -r ../requirements.txt
```

## Usage

### How to run
1. Make sure the `augmented_sword` environment is activated and you're under the `src/` directory:
    ```shell
    conda activate augmented_sword

    cd src/
    ```
2.
    **Windows**: Using the Command Line, inside the project directory:
    ```shell
    python main.py
    ```

    **Linux/MacOS**: Using the Command Line, inside the project directory:

    ```shell
    python3 main.py 
    ```
3. The program will wait for `q` to be pressed, to close the window and exit.

### How to remove
1. Remove the created environment:
```shell
conda remove -n ENV_NAME --all
```
2. Delete the local repository.

## Virtual Sword Display Modes
After running the program, the device's default webcam will be used to capture the image. The image will be processed and the result will be displayed in a new window.
The user can then display a virtual sword using one of two methods:

**Note:** In this project we use ArUcO markers that can be found [here](https://chev.me/arucogen/).

### Simple Mode - Flat Cardboard Cutout (2 ArUcO Markers)
- A planar object with the markers 0 and 1.
<p align="center">
   <img width="412" alt="image" src="https://github.com/FabioMiguel2000/Augmented-Sword/assets/100025288/4c706893-c2cf-4c8f-81e2-4ed528dfca9d">
</p>

### Advanced Mode - Handheld Cube (5 ArUco Markers)
- A cube with 5 faces, where the markers from 1 to 4 are placed on the side faces without any rotation and the marker 0 is placed on the top face, like in the following picture:
<p align="center">
   <img width="412" alt="image" src="https://github.com/FabioMiguel2000/Augmented-Sword/assets/100025288/fabefdfd-310c-464a-9203-801eff754391"
</p>

## Group Members

- [Fabio Huang](https://github.com/FabioMiguel2000) / up201806829@fe.up.pt

- [Luis Guimaraes](https://github.com/luismrguimaraes) / up202204188@edu.fe.up.pt

- [Ricardo Gonçalves Pinto](https://github.com/ricas00) / up201806849@edu.fe.up.pt
