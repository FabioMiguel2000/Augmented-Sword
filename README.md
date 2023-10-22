# RVA-proj1

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

### Simple Mode - Flat Cardboard Cutout (2 Artoolkit Markers)
- [!] ToDo: Insert Instructions here on how to print the cardboard cutout, and which markers and their IDs to put on each cube face.

### Advanced Mode - Handheld Cube (5 ArUco Markers)
- [!] ToDo: Insert Instructions here on how to print the cube, and which markers and their IDs to put on each cube face.


## Group Members

- [Fabio Huang](https://github.com/FabioMiguel2000) / up201806829@fe.up.pt

- [Luis Guimaraes](https://github.com/luismrguimaraes) / up202204188@edu.fe.up.pt

- [Ricardo Gonçalves Pinto](https://github.com/ricas00) / up201806849@edu.fe.up.pt
