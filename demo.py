
import sys, pkgutil, importlib
import logging
import matplotlib
matplotlib.use("Agg")

# Import util for propper logging format.
import qball.util

if __name__ == "__main__":
    import qball.experiments
    pth = qball.experiments.__path__
    exp_names = [name for _,name,_ in pkgutil.iter_modules(pth)]
    logging.info("Available experiments: %s" % ", ".join(exp_names))

    import qball.solvers
    pth = qball.solvers.__path__
    model_names = [name for _,name,_ in pkgutil.iter_modules(pth)]
    logging.info("Available models: %s" % ", ".join(model_names))

    if len(sys.argv) < 3:
        sys.exit("Error: Please specify an EXPERIMENT and a MODEL.")

    if sys.argv[1] not in exp_names:
        sys.exit("Error: Unknown experiment '%s'" % sys.argv[1])

    if sys.argv[2] not in model_names:
        sys.exit("Error: Unknown model '%s'" % sys.argv[2])

    logging.info("Running experiment '%s' from command line." % sys.argv[1])
    exp_modulename = "qball.experiments.%s" % (sys.argv[1],)
    exp_module = importlib.import_module(exp_modulename)
    exp = exp_module.MyExperiment(sys.argv[2:])
    exp.run()
    exp.plot()
