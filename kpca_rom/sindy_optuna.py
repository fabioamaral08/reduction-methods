import sys 
sys.path.append('../src') 
from pathlib import Path
from utils import log_exp
from create_modes import load_modes
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pysindy as ps
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse

import multiprocessing as mp
import time
class Objective:
    def __init__(self, Phi, bias):
        self.Phi = Phi
        self.bias = bias
    def __call__(self, trial):
        """Objective function for Optuna optimization."""
        
        # Define hyperparameters to optimize
        threshold = trial.suggest_float('relax_coeff_nu', 1e-6, 1, log=True)
        nu = trial.suggest_float('reg_weight_lam', 1e-4, 1.5, log=True)
        max_iter = 2000
        
        # Initialize SR3 optimizer
        optimizer = ps.SR3(reg_weight_lam=threshold, relax_coeff_nu=nu, max_iter=max_iter)
        
        # Initialize SINDy model
        model = ps.SINDy(
            optimizer=optimizer,
            feature_library=ps.PolynomialLibrary(degree=3, include_bias=self.bias)
        )
        
        # Fit model (use your training data)
        model.fit(self.Phi, t=1)
        
        # Evaluate on validation set
        t = np.arange(0, self.Phi.shape[0])
        mse = self.solve_with_timeout(model, t, trial, timeout_seconds=30)
        return mse
    
    def run_solver(self, model, t, trial, queue):
        try:
            predictions = model.simulate(self.Phi[0], t)
            mse = mean_squared_error(self.Phi, predictions)
        except (RuntimeError, ValueError, FloatingPointError, TimeoutError) as e:
            # Marca inviável + registra motivo (muito útil no diagnóstico)
            trial.set_user_attr("constraints", (1.0,))
            trial.set_user_attr("fail_reason", repr(e))
            queue.put(float("inf"))
            return
        
        queue.put(mse)
        return

    def solve_with_timeout(self, model, t, trial, timeout_seconds):
        queue = mp.Queue()
        p = mp.Process(target=self.run_solver, args=(model, t, trial, queue))
        p.start()
        p.join(timeout_seconds)

        if p.is_alive():
            p.terminate()
            p.join()
            return float("inf")

        return queue.get()

def constraints_func(trial: optuna.trial.FrozenTrial):
    # deve retornar uma sequência de floats
    # > 0  => viola constraint (inviável)
    return trial.user_attrs.get("constraints", (0.0,))

def optimize(Re, Wi, beta, kernel_type, n_modes_file = 10, n_modes_model=3,  bias=True, n_trials=100,verbose=True):
    """Run Optuna optimization."""
    
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    exp_id = f'optuna_cavity_oldroyd_Re_{Re:g}_Wi_{Wi:g}_beta_{beta:g}_kernel_{kernel_type}_nmodes_{n_modes_model}'
    storage_name = "sqlite:///runs/{}.db".format(exp_id)
    # Optuna optimization
    Phi = load_modes(Re, Wi, beta, kernel_type, n_modes_file)[:,:n_modes_model]
    sampler = TPESampler(seed=42, constraints_func=constraints_func)
    pruner = MedianPruner()
    objective = Objective(Phi, bias)
    study = optuna.create_study(study_name = exp_id, storage=storage_name,
                                sampler=sampler, pruner=pruner,load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError, ValueError, FloatingPointError))
    
    #Save figures
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"runs/{exp_id}_param_importances.html")
    fig2 = optuna.visualization.plot_optimization_history(study)
    fig2.write_html(f"runs/{exp_id}_optimization_history.html")

    # save study
    study.trials_dataframe().to_csv(f"runs/{exp_id}_trials.csv", index=False)
    # Log test
    MANIFEST = Path("summary/manifest.csv")
    rec = log_exp.RunRecord(
        exp_id=exp_id,
        geom="cavity",
        constitutive_model="Oldroyd-B",
        kernel=kernel_type,
        task="create_modes",
        Wi=Wi,
        beta=beta,
        r=n_modes_model,
        notes="Run Optuna for hyperparameter tuning"
    )
    log_exp.upsert_manifest(rec, MANIFEST)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optuna tunning for SR3')
    parser.add_argument('-Re', type=float, help='Reynolds number')
    parser.add_argument('-Wi', type=float, help='Weissenberg number')
    parser.add_argument('-beta', type=float, help='Beta parameter')
    parser.add_argument('-kernel_type', type=str, help='Kernel type')
    parser.add_argument('--n_modes_file', type=int, default=10, help='Number of modes in file')
    parser.add_argument('--n_modes_model', type=int, default=3, help='Number of modes for model')
    parser.add_argument('--bias', type=bool, default=True, help='Include bias term')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--verbose', type=bool, default=False, help='Show Info during optimization', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    Re = args.Re
    Wi = args.Wi
    beta = args.beta
    kernel_type = args.kernel_type
    n_modes_file = args.n_modes_file
    n_modes_model = args.n_modes_model
    bias = args.bias
    n_trials = args.n_trials
    verbose = args.verbose
    start_time = time.time()
    optimize(Re, Wi, beta, kernel_type, n_modes_file=n_modes_file, n_modes_model=n_modes_model, bias=bias, n_trials=n_trials, verbose=verbose)
    total_time = time.time() - start_time
    print(f'Exec time: {total_time:g}s')