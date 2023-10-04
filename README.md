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
conda create -n ENV_NAME python=3.11
```
2. Activate the environment:
```shell
conda activate ENV_NAME
```
3. Install the necessary libraries: 
```shell
pip install -r ../requirements.txt
```

## Usage
After running the program, the default webcam will be used to capture the image. The image will be processed and the result will be displayed in a new window.
In order to display the AR virtual sword, the user simply needs to hold a cube, with the following faces:
- **ToDO: Insert faces here and maybe images of them**
- **Right now it detects `4x4` markers generated in `https://chev.me/arucogen/`**

### How to run
1. Make sure the `ENV_NAME` environment is activated.
2.
    **Windows**: Using the Command Line, inside the project directory:
    ```shell
    python main.py
    ```

    **Linux/MacOS**: Using the Command Line, inside the project directory:

    ```shell
    python3 main.py 
    ```
3. The program will wait for a key press, 'q', to close the window and exit.

### How to remove
1. Remove the created environment:
```shell
conda remove -n ENV_NAME --all
```
2. Delete the local repository.


## Group Members

- [Fabio Huang](https://github.com/FabioMiguel2000) / up201806829@fe.up.pt

- [Luis Guimaraes](https://github.com/luismrguimaraes) / up202204188@edu.fe.up.pt

- [Ricardo Gonçalves Pinto](https://github.com/ricas00) / up201806849@edu.fe.up.pt
