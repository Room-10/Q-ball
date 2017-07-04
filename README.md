
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
* sh_l_tvo
* sh_l_tvw (proposed)

Not yet implemented:
* srk_l_tvc
* srk_l_tvw

Setup
-----

    # setting up a virtual environment
    python3 -m venv env
    source ./env/bin/activate
    pip install --upgrade pip
    pip install wheel
    pip install -r requirements.0.txt
    pip install -r requirements.1.txt

    # fix bug in dipy-0.11.0
    cd env/lib/python3.5
    patch -p0 < ../../../overrides/dipy-int.patch
    cd ../../../

    # include VTK7 from special location
    cd env/lib/python3.5
    echo "/opt/VTK-7.0.0/lib/python3.5/site-packages" > site-packages/vtk7.pth
    cd ../../../

    # install mosek (you need a license!)
    pip install git+http://github.com/MOSEK/Mosek.pip

Run
---

Run the scripts in the `demos` subdirectory, e.g.,

    python demos/cross.py

Outlook
-------

* The confidence-interval-based fidelity still has to be implemented.
* Evaluate results wrt. quality of fiber tractography.
