
# pyperch

## Getting Started

### About
Pyperch is a neural network weight optimization package developed to support students taking Georgia Tech’s graduate machine learning course CS7641. 
Three random optimization algorithms - randomized hill climbing, simulated annealing, and a genetic algorithm - can be used as drop-in replacements for traditional gradient-based optimizers using PyTorch.
### Install

```
pip install pyperch
```
## Examples

### PyTorch-Only (No Skorch Required)

- [`rhc_adam_hybrid.ipynb`](notebooks/hybrid_rhc_network.ipynb)  
  **New notebook** demonstrating PyTorch training using pyperch's RHC optimizer and Adam together on different layers - no Skorch dependency required. Ideal for hybrid workflows and experimentation.
  
---

### Classic Skorch Examples

- [`backprop_network.ipynb`](notebooks/backprop_network.ipynb)  
  Standard neural network training using backpropagation with Skorch.

- [`rhc_opt_network.ipynb`](notebooks/rhc_opt_network.ipynb)  
  Neural network trained using Randomized Hill Climbing (RHC) via Skorch.

- [`sa_opt_network.ipynb`](notebooks/sa_opt_network.ipynb)  
  Neural network trained using Simulated Annealing (SA) via Skorch.

- [`ga_opt_network.ipynb`](notebooks/ga_opt_network.ipynb)  
  Neural network trained using a Genetic Algorithm (GA) via Skorch.

- [`regression_examples.ipynb`](notebooks/regression_examples.ipynb)  
  Regression tasks using randomized optimization.


## Contributing

Pull requests are welcome.  

* Fork pyperch.
* Create a branch (`git checkout -b branch_name`)
* Commit changes (`git commit -m "Comments"`)
* Push to branch (`git push origin branch_name`)
* Open a pull request
