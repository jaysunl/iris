# iris

## Machine Learning Models
To run: ```python main.py [-h] [-g] [-m MODEL]``` 
where

```-h``` help option

```-g``` graph the dataset

```-m``` train using the model specified by ```MODEL``` string

## Data Infrastructure
Splunk dashboards are used to visualize and automate and compare various machine learning workloads. For more complex models, hyperparameters are reconfigurable. We use Splunk Machine Learning Toolkit (MLTK) for visualizing basic machine learning workloads. For complex models, we may use Splunk Data Science and Deep Learning Toolkit in which we set up a Docker endpoint for our workloads to run on. This provides more detailed metrics if we choose to use neural networks.
