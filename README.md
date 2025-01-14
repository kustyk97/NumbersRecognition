# NumbersRecognition

This repository containst two modules:
- Number Classifier 

    This is an implementation of CNN model trained on MNIST dataset, to classifire images of numbers (0-9)

    ![](figures/Acc.png)

- Application 

    This is an application whitch use Number Classifier module, to recognition number on image, whitch is written by user inside application. Belowe is an example screenshot of application window 

    ![Example screen from application](figures/AppScreenshot.png)

## Requirements
- Python 3.x
- NumPy
- OpenCV
- PyTorch
- PyQt5
- Seaborn
- Matplotlib
- Pandas
- tqdm


## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/kustyk97/NumbersRecognition.git
    ```
2. Navigate to the project directory:
    ```bash
    cd NumbersRecognition
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```


## Usage
### Application
If you want to run main aplication, you can do this by run the following command:
```bash
python main.py
```
### Train 
You can train your own model, by run the following command:
```bash
python numberClassifier/train.py 
```
### Plots
You can plot results of training, by run the following command:
```bash
python plots.py 
```
## License
This project is licensed under the MIT License.