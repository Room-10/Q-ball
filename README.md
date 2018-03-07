
Q-Ball
======

Reconstruction of Q-ball images using a confidence-interval-based fidelity and
an optimal transport-based (spatial) TV-regularizer.
On the sphere, orientation distribution functions are regularized using spherical
harmonics.

Abbreviations:
* bases: sh (spherical harmonics), srk (sparse reproducing kernels), n (none)
* fidelity: l (L2), w (Wasserstein), bnd (confidence intervals/bounds)
* regularizers: tvo (ODF pointwise), tvw (Wasserstein), tvc (coefficients)

Implemented models:
* n_w_tvw (SSVM)
* sh_w_tvw (modified SSVM)
* sh_l_tvc (Ouyang '14)
* sh_bndl1_tvc (modified Ouyang '14)
* sh_bndl2_tvc (modified Ouyang '14)
* sh_l_tvo
* sh_l_tvw (proposed)
* sh_bndl2_tvw (proposed)

Not yet implemented:
* sh_bndl1_tvw (proposed)
* srk_l_tvc
* srk_l_tvw

Setup
-----

The recommended (and tested) setup is based on Ubuntu 16.04 with CUDA 8.0 or
newer. In that case, the following lines will do:

    sudo apt install -y python3 python3-venv python3.5-dev llvm-dev g++

    # set up a virtual environment
    python3 -m venv env
    source ./env/bin/activate
    pip install --upgrade pip
    pip install wheel numpy
    pip install -r requirements.0.txt
    pip install -r requirements.1.txt

    # install VTK 7.x with Python 3.x bindings
    sudo add-apt-repository -y ppa:elvstone/vtk7
    sudo apt update
    sudo apt install -y vtk7
    echo "/opt/VTK-7.0.0/lib/python3.5/site-packages" > env/lib/python3.5/site-packages/vtk7.pth

It is possible to run the code without CUDA using the solver parameter
"use_gpu=False", but CUDA is recommended and enabled by default.

Optionally, parts of the code can be run using PyCVX which will profit a lot
from using the (commercial) MOSEK solver:

    # install mosek (you need a license!)
    pip install git+https://github.com/MOSEK/Mosek.pip

Run
---

Run the script `demo.py` without arguments to show available experiments and models:

    python demo.py

After that, specify the experiment and model you want to test:

    python demo.py cross sh_l_tvw

More options are documented when using

    python demo.py cross sh_l_tvw --help

All logs, plots, parameters, results etc. are automatically stored in subdirectories
of the `results` directory.

Outlook
-------

* The confidence-interval-based fidelity still has to be implemented.
* Evaluate results wrt. quality of fiber tractography (feasible?)
