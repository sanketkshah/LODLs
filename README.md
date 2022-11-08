# Code for "Decision-Focused Learning without Decision-Making: Learning Locally Optimized Decision Losses"

## Set up environment

To run the code, first you have to set up a conda environment. Once you have [Anaconda installed](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), run the following command:
```
conda env create --name lodlenv --file=environment.yml
```
Once the environment has been created, load it using the command
```
conda activate lodlenv
```

You'll also need to manually install one of the libraries using the following command
```
conda install metis
```

## Running Different Domains

We now present the default parameters required to run our DirectedMSE approach on the different domains from the paper below.

### Linear Model

```
python main.py --problem=cubic --loss=weightedmse++ --seed=1 --instances=400 --testinstances=400 --valfrac=0.5 --numitems=50 --budget=1 --lr=0.1 --layers=1 --sampling=random_hessian --samplingstd=1 --numsamples=50000 --losslr=0.01 --serial=False
```

### Web Advertising

```
python main.py --problem=budgetalloc --loss=weightedmse++ --seed=1 --instances=100 --testinstances=500 --valfrac=0.2 --numitems=5 --budget=1 --sampling=random_hessian --numsamples=5000 --losslr=0.1 --serial=False
```

### Portfolio Optimization

```
python main.py --problem=portfolio --loss=weightedmse++ --seed=1 --instances=400 --testinstances=400 --valfrac=0.5 --stocks=50 --stockalpha=0.1 --lr=0.01 --sampling=random_hessian --samplingstd=0.1 --numsamples=50000 --losslr=0.001 --serial=False
```

_Note: (A) Sampling points can take a **long** time. (B) In the event that you receive an error along the lines of `Thread creation failed: Resource temporarily unavailable`, you should run the following command:_

```
export OMP_NUM_THREADS=1
```

## Running Different Approaches

In Table 1 of our paper, we highlight 7 different approaches. To run the domains above with a specific approach, set the `--loss` parameter to the input corresponding to that approach. _(Note: Obtaining the results in the paper requires method-specific hyperparameter tuning.)_

| Method      | Corresponding Input |
| ----------- | ----------- |
| 2-Stage     | mse         |
| DFL         | dfl         |
| WeightedMSE | weightedmse |
| DirectedWeightedMSE | weightedmse++ |
| DirectedWeightedMSE | weightedmse++ |
| Quadratic   | quad |
| DirectedQuadratic | quad++ |
| NN          | dense       |

Similarly, to run different sampling strategies, set the `--sampling` parameter according to the following table:

| Method      | Corresponding Input |
| ----------- | ----------- |
| 1-perturbed     | random_jacobian |
| 2-perturbed     | random_hessian |
| All-perturbed     | random |
