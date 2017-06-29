
Q-Ball
======

Reconstruction of Q-ball images using a confidence-interval-based fidelity and
an optimal transport-based (spatial) TV-regularizer.
On the sphere, orientation distribution functions are regularized using spherical
harmonics.

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

    python demos/cross_hardi.py

Outlook
-------

* The confidence-interval-based fidelity still has to be implemented.
* Evaluate results wrt. quality of fiber tractography.
