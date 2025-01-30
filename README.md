# AutoScatter

AutoScatter is an efficient tool to automatically discover coupled mode setups with desired scattering properties, e.g. to realize an isolator, a circulator or an amplifier. Our tool optimises the discrete and continuous properties of the setups and provides an exhaustive list of all possible setups. The user can then select the setups best suited for the hardware.

This repository allows you to design your own coupled mode setups with your desired scattering behaviour and to reproduce the examples discussed in our publication: \
[Automated Discovery of Coupled Mode Setups](https://arxiv.org/abs/2404.14887) \
*Jonas Landgraf, Vittorio Peano, and Florian Marquardt* (2024) 

## Overview over available examples:
1_isolator.ipynb: Design isolators. You can try out the code [on Google Colab](https://colab.research.google.com/github/jlandgr/autoscatter/blob/main/1_isolator.ipynb) \
2_circulator.ipynb: Design ciculators. You can try out the code [on Google Colab](https://colab.research.google.com/github/jlandgr/autoscatter/blob/main/2_circulator.ipynb) \
3_directional_coupler.ipynb: Discovers the graphs to realise a directional coupler. You can try out the code [on Google Colab](https://colab.research.google.com/github/jlandgr/autoscatter/blob/main/3_directional_coupler.ipynb) \
4_directional_coupler.ipynb: Analyses and Generalises the discovered directional coupler graphs. You can try out the code [on Google Colab](https://colab.research.google.com/github/jlandgr/autoscatter/blob/main/4_directional_coupler_generalisation.ipynb) \
5_directional_quantum_limited_amplifier.ipynb: Discovers and generalises a fully-directional quantum-limited amplifer. You can try out the code [on Google Colab](https://colab.research.google.com/github/jlandgr/autoscatter/blob/main/5_directional_quantum_limited_amplifier.ipynb) \
6_optomechanical_circulator.ipynb:  Design an optomechanical circulator and introduces the far-detuned mode. You can try out the code [on Google Colab](https://colab.research.google.com/github/jlandgr/autoscatter/blob/main/6_optomechanical_circulator.ipynb) 

## Installation
We recommend create a new environment using venv or conda. The package can be installed with:
```
git clone https://github.com/jlandgr/autoscatter.git
cd autoscatter
python setup.py install
```
Alternatively use the example Google Colab notebooks to test our code.

## Cite us

Are you using AutoScattering in your project or research or do some related research? Then, please cite us!
```
@misc{landgraf2024automateddiscoverycoupledmode,
      title={Automated Discovery of Coupled Mode Setups}, 
      author={Jonas Landgraf and Vittorio Peano and Florian Marquardt},
      year={2024},
      eprint={2404.14887},
      archivePrefix={arXiv},
      primaryClass={physics.optics},
      url={https://arxiv.org/abs/2404.14887}, 
}
```
