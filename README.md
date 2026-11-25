# Formally Verified LTL<sub>_f_</sub> for PyTorch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24531331.svg)](https://doi.org/10.5281/zenodo.24531331)

This repository accompanies the paper titled *Formally Verified Neurosymbolic Trajectory Learning via Tensor-based Linear Temporal Logic on Finite Traces*.
It contains OCaml code generated from the Isabelle/HOL formalisation, which is used to inject LTL<sub>_f_</sub> constraints into PyTorch learners. This is required to run the case studies in the paper, which are also included.

Note that the case study code does not fix the randomness seed for Python, NumPy or PyTorch.
As such, there can be small differences in the learned trajectories.

## Licence

This software is released under an MIT license (details in the accompanying LICENSE file).

## Installation

Requirements to build the LTL<sub>_f_</sub> loss executable:
- `opam` to manage OCaml packages, with `ocaml` version 5 (tested with `5.1.1` and `5.3.0`), `zarith` (for arbitrary precisiion integers) and `angstrom` (for LTL<sub>_f_</sub> parsing) packages installed
- `dune` to build the OCaml project

Once opam is installed and initialised, you can create a new switch and install the packages as follows:
```
$ opam switch create NAME 5.3.0
$ eval $(opam env --switch=NAME)
$ opam install zarith angstrom
```

To build the executable, run the `loss_setup.sh` script in this directory.
This checks that the above requirements are met, builds the executable and makes it available to the Python interface.
(To test that this step was successful, you can navigate into the `python` directory and run `ltl_loss_test.py`.)

The case studies are in the `case_studies` directory in Jupyter notebooks and the `self_contained.py` script.
The latter script executes the configured number of repetitions for all the case studies without the need for a Jupyter notebook, producing total timings and figures for each.

Python libraries required are:
- `jupyter` or `jupyterlab` to open the notebooks with the case studies
- `pytorch`, `matplotlib`, `seaborn`, `scipy` for the case study setup

Using for instance [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html), you can execute the following to create an environment, install the dependencies and start the notebook:
```
$ micromamba create --name NAME
$ micromamba activate NAME
$ micromamba install pytorch matplotlib scipy jupyterlab -c defaults
$ jupyter lab
```

The specific requirements of the Micromamba environment above made at the time of writing have been exported to `case_studies/requirements.yaml`.
To rebuild an environment from that, run the following:
```
$ micromamba create --name NAME --file case_studies/requirements.yaml
```

### Static Linking

There are situations when the loss executable needs to be compiled on a different machine than where it will run.
For instance, running the case studies on a cluster that does not have `ocaml`, `opam` or `dune` available to compile the loss executable.
In such cases it is possible that the dynamically linked libraries on the target machine differ from the compilation machine.
To solve this, the loss executable can instead be linked statically by adding the following line to the relevant [dune file](ocaml/conse/dune) in the executable definition:
```
  (flags (:standard -ccopt -static))
```

Note that this requires the static version of the gmp library, which may not be present on all machines and may need to be installed.
For instance, the relevant package on the Arch linux User Repository is [libgmp-static](https://aur.archlinux.org/packages/libgmp-static).

## Usage

The main entry-point to using the formally verified constraints is the `case_studies/ltl/constraint.py` file and its class `Constraint`.
This takes four parameters, with only the first being mandatory:
- `ltl`: a string representing the LTL constraint (see [Constraint Language](#constraint-language))
- `transform`: a function applied to the input trajectory tensor before the loss is calculated, such as computing a distance from a point as an additional state variable (default: identity)
- `consts`: constants to append to the state (after `transform` has been applied), such as a distance threshold (default: none)
- `gamma`: smoothing factor to use (default: 0.05)

(There is a fifth optional parameter `exectype` to determine the LTL loss executable to use, which currently only supports `ocaml`.)

For instance consider a constraint for avoiding a zone around a point.
The constraint instance transforms a trace with 2D coordinates into distance from a point, appends a threshold constant and sets the smoothing factor to 0.005.
The LTL<sub>_f_</sub> formula is `G (0 >= 1)` to say that the distance from the point must always be greater than or equal to the threshold.
In Python:
```
avoid_pt = torch.Tensor([0.4,0.4]).to(DEVICE)
dist_f = lambda x : torch.sqrt(torch.sum((x-avoid_pt)**2,dim=-1)).unsqueeze(-1)
c = Constraint("G (0 >= 1)", dist_f, [0.1], 0.005)
```

To use such a constraint, use its `loss` method with the trajectory tensor (tensor of order at least 2: time, state and then possibly batches).
The constraint will apply the relevant transform, append the relevant constants and apply the loss function using the LTL loss executable.

The constraint loss has the forward and backward functions defined for PyTorch's Autograd, so it can be included as part of a larger framework as either the single loss or as part of a combined loss.

### Constraint Language

The constraint language is based on LTL<sub>_f_</sub> with atoms formed by a comparison language between elements of the state.
Its string representation is parsed in [OCaml](ocaml/conse/ltl_loss.ml) into the relevant formal objects.

The basic atoms of the language are natural numbers representing indices into the trace (possibly transformed and extended as part of the `Constraint` object).

These indices can be paired in six ways, representing comparisons of their values at a certain index in the trace: `<=`, `<`, `==`, `!=`, `>=`, `>`.

These comparisons form the atoms of an expression language that represents LTL<sub>_f_</sub> constraints.
It has the following operators:
- `!` - logical negation
- `N` - strong next
- `X` - weak next
- `G` - always
- `F` - eventually
- `U` - weak until
- `R` - strong release
- `&` - logical conjunction
- `|` - logical disjunction
- `<->` - logical equivalence
- `->` - logical implication

## Citation

If you use this software in your research, please cite the associated paper, accepted for publication in the Journal of Artificial Intelligence Research (JAIR).

The paper is currently in press. The citation will be updated with the final DOI and publication details.

```bibtex
@article{chevallier2026formalltl,
  author  = {Mark Chevallier and Filip Smola and Richard Schmoetten and Jacques D. Fleuriot},
  title   = {Formally Verified Neurosymbolic Trajectory Learning via Tensor-based Linear Temporal Logic on Finite Traces},
  journal = {Journal of Artificial Intelligence Research},
  year    = {forthcoming},
  note    = {In press}
}
```
