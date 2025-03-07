# State of Health Estimation of Second Life Batteries using Approximate Message Passing with Shallow Neural Networks

## Abstract
Batteries as stationary storage or in other applications such as electric vehicles are
an essential part of the transition to a sustainable future. Due to calendric and cyclic
aging, the capacity of a battery degrades over time. Measuring this degradation in
terms of the State of Health (SoH) of a battery can be a time-consuming and costly
endeavor. In addition, data about battery degradation is scarce due to the high costs
of (dis-)charging a battery up until failure. Consequently, a precise estimation of
the SoH is not only essential but also safety-critical, particularly when batteries are
utilized as stationary storage. That is why we propose the usage of factor graphs.
Factor graphs offer an efficient framework for probabilistic inference through mes-
sage passing. However, one key limitation of the applicability of factor graphs are
the message update functions for given factors. Those update functions may not
have analytical solutions and can require the computation of intractable integrals.
We propose a novel approximation for those factor update functions using shallow
neural networks that were trained on factor specific sampled data. By replacing
analytical functions with neural networks, we expand the applications that can
leverage the probabilistic framework of factor graphs. We introduce our sampling
framework as well as assess the sampling process and the approximated factors
against two factors whose analytical solutions are already known. Afterwards, we
present a factor graph that models part of the electrical model of a battery cell
as a proof-of-concept graph to estimate the SoH of a battery cell. We compare
the converged posterior distributions of the SoH at different time steps against
simulated battery data.

Our experiments show that a sample-based framework in combination with
shallow neural networks that are able to approximate a factor’s message update
function is a functioning and robust approach. We obtained promising results with
a Root Mean Square Error (RMSE) value of 0.0075 and a Mean Absolute Percentage
Error (MAPE) value of 1.07% for the state of health estimation across five charge
cycles using the introduced factor graph. Moreover, we achieved the results with
an acceptable runtime of 32.76 seconds.

## Steps to Reproduce Results

### Sample Data for the Weighted Sum, Gaussian Mean and Electric Model factor
In `SoHEstimation/approximate_message_passing/weighted_sum_factor`, `SoHEstimation/approximate_message_passing/gaussian_mean_factor` and `SoHEstimation/approximate_message_passing/em_factor` you find the `script_execution.jl` file. This file is the entry point for the data generation. You can go ahead and just execute it or adjust some of the distribution parameters.

### Train Neural Networks
`SoHEstimation/approximate_message_passing/evaluation/train_nets.jl`is the script that trains neural networks on the previously generated data according to the factors one adds to the list at the beginning of the file. Those networks will be trained using the hyperparameters found and described in the thesis. All networks, including the normalization parameters, are saved in `SoHEstimation/approximate_message_passing/models`.

### Adapt Factors to Use NNs instead of Analytical Solution
`lib/factors.jl` contains all implemented factors. Some of them were already implemented by the chair of AI & Sustainability of the Hasso-Plattner Institute. However, in each update_msg function inside the approximated factors, we substituted the analytical formulas with our neural networks. In addition, the EMFactor at the end of the file, is a completely new one, used in the factor graph to estiamte the SoH of a battery cell. This factor is also capable of using our sampling approach instead of neural networks in order to update the outgoing message.

### Run the Factor Graph for the SoH Estimation
`SoHEstimation/approximate_message_passing/em_factor/test_em_factorgraph.jl` can be executed to load the simulated data generated by the battery simulator, build the factor graph, start the convergence, and finally do the evaluation. The battery data can previously be generated with `SoHEstimation/simulator/test.jl`
