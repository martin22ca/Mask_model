# Mask_Recognition

Masks play a crucial role in protecting the health of individuals against respiratory diseases, as is one of the few precautions available for COVID-19 in the absence of immunization. Nowadays most hospitals, shops, offices, etc, often require the use of a mask to access the premises due to restrictions  to prevent the spread of the COVID-19 Virus. Normally there's someone in charge of enforcing these rules but with the help of AI we can automate this task, with high accuracy. 

## About the problem

We started with [This dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) from kaggle which contained 853 images belonging to the 3 classes, as well as their bounding boxes. Those classes being:

	With mask;
	Without mask;
	Mask worn incorrectly.

We later realized that the daset quality was quite poor and we decided to add some data from other sources like the dataset from [faces on the wild](http://vis-www.cs.umass.edu/lfw/) and this dataset from a [hackathon](https://github.com/hydramst/hackathon_2021) 

## Architecture used
```python
BootstrapCCpy(cluster, K, B, n_cores)
```
Parameters
- cluster

    The class of a clustering algorithm implementation (Mandatory)

    For example, you could head to [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster) to pick the one of your preference. Let's use KMeans and do it properly

     	cluster=KMeans().__class__

- K

	Positive Integer (Mandatory)
	
	Refers to the maximum number of clusters to try

	For example, if it's set to 4, the algorithm will process the data in 2, 3, and 4 clusters. 

- B 

	Positive Integer (Mandatory)

	Amount of bootstrap samples to be performed by the algorithm for each cluster number.

- n_cores

	Integer (Optional, default: -1)

	The number of CPU cores to be used by the algorithm to fit the data. If it's set to -1, all available cores will be used.

## Getting started

Download this repository
```bash
https://github.com/martin22ca/Mask_model.git
```
_Please check out dependencies section in case you are having trouble._
```python
fit(data, verbose)
```

Trains the algorithm with the provided data to discover the optimal number of clusters. This function can be called just once per object instance.


:warning: Take into account that this method is CPU and memory intensive, it may take a long time to be completed. :warning:

Parameters
- data
	
	ndarray (Mandatory)

- :construction: verbose

	boolean (Optional, default: False)

	Determines if it should print messages when fitting


## Usage
```python
get_best_k()
```
## Results

## Authors

* [Caceres Martin](https://github.com/martin22ca) - Faculty of Engineering, Catholic University of Córdoba (UCC) *
* [Paschini Catalina](https://github.com/cata99) - Faculty of Engineering, Catholic University of Córdoba (UCC) *
* Ing. Pablo Pastore - DeepVisionAi, inc.
* [Bioing. PhD Elmer Fernández](https://github.com/elmerfer) - CIDIE-CONICET-UCC

*both authors must be considered as the first author