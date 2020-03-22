# Coronavirus spread prediction

Python script, based on SIR model + astroABC library + Wikipedia data, to predict Coronavirus spread.

## Implementation details

Simple, one-dimensional [SIR](https://www.maa.org/press/periodicals/loci/joma/the-sir-model-for-spread-of-disease-the-differential-equation-model) model has been used to simulate the number of people affected. It consists of three dynamic variables, each as a function of time:
```
 S(t) - susceptible
 I(t) - infected
 R(t) - recovered and dead (S - I)
```
Max-squared-error cost function is responsible for calculating the difference between real and simulated data. 

To minimize the cost function, the Approximate Bayesian Computation Sequential Monte Carlo (ABC SMC) algorithm implementation ([astroABC](https://github.com/pedrycz/astroABC)) was introduced with four parameters:
```
 s% - total affected population divided by total population
 i% - initially affected pupulation divided by total affected population
 b - illness rate from SIR model
 k - healing rate from SIR model
```
Parameter distribution and other simulation properties can be adjusted in the code.

## Requirements

 * [astroABC](https://github.com/pedrycz/astroABC) (forked from EliseJ and modified by me)
 * python, numpy, matplotlib

## Examples

