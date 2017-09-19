
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

You will need VTK 7.x with Python 3.x bindings, llvm-config (included in llvm
devel packages) and gcc-c++.

    # setting up a virtual environment
    python3 -m venv env
    source ./env/bin/activate
    pip install --upgrade pip
    pip install wheel numpy
    pip install -r requirements.0.txt
    pip install -r requirements.1.txt

    # include VTK7 from special location
    echo "/opt/VTK-7.0.0/lib/python3.5/site-packages" > env/lib/python3.5/site-packages/vtk7.pth

    # install mosek (you need a license!)
    pip install git+http://github.com/MOSEK/Mosek.pip

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
