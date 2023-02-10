## CS8903 (Guided study and research)-Spring 22'

### ML Surrogates for characterizing fracture toughness

Faculty mentored research project aiming to provide ML alternative to computationally intensive finite-element modelling for characterizing specimen’s fracture toughness.

#### Professors:

- [Christos Athanasiou](https://www.ceathanasiou.com/)
- [Elizabeth Qian](https://www.elizabethqian.com/)

#### Relevant research papers:

- https://www.sciencedirect.com/science/article/abs/pii/S1359645420302032
- https://www.pnas.org/doi/abs/10.1073/pnas.2104765118

#### Methodology

> Figure work in progress

#### File organization
|-main.py (Main file with the core function calls and hyperparameter definition)  
|-- data_func.py (File with functions/classes relevant to pre-processing of the data)  
|-- epoch_func.py (File with functions/classes relevant to epoch-specific steps, example: training, testing, early-stopping, etc.)  
|-- loss_func.py (File with functions/classes relevant to calculating different loss/accuracy criteria)  
|-- model_func.py (File with functions/classes relevant to pre-defined neural network structures)  
|-- post_func.py (File with functions/classes relevant to post-processing of data, example: plotting loss curves, accuracy curves, etc.)  
|-- setup.py (File with relevant info for running the *.py files)
