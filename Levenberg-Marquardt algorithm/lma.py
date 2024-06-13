import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def breit_wigner(x: np.ndarray,
                 a: float, b: float, c: float) -> np.ndarray:
    return a / ((b - x)**2 + c)


def eval_jacobian(x: np.ndarray, beta: np.ndarray,
                  function: callable, delta: float = 1e-4) -> np.ndarray:
    # initialize jacobian matrix with zeros
    jacobian = np.zeros((len(x), len(beta)))
    # compute entries in jacobian matrix
    for i in range(len(x)): 
        for j in range(len(beta)):
            plus_delta_beta = beta.copy()
            minus_delta_beta = beta.copy()
            
            plus_delta_beta[j] += delta
            minus_delta_beta[j] -= delta
            
            jacobian[i,j] = (function(x[i], *plus_delta_beta) - function(x[i], *minus_delta_beta)) / (2*delta)
        
    return jacobian


def eval_error(x: np.ndarray, y: np.ndarray,
               beta: np.ndarray, function: callable) -> np.ndarray:
    # compute the error of f(x) with parameters beta
    
    error = y - function(x, *beta)
    return error


def gauss_newton_update(jacobian: np.ndarray, error: np.ndarray) -> np.ndarray:
    # compute the parameter update delta_beta
    # ...
    
    delta_beta = np.zeros(jacobian.shape[1])
    
    A = jacobian.T @ jacobian
    g = jacobian.T @ error
    delta_beta = np.linalg.solve(A,g)
    
    return delta_beta


def gauss_newton(x: np.ndarray,
                 y: np.ndarray,
                 beta: np.ndarray,
                 function: callable,
                 max_iter: int = 100,
                 threshold: float = 1e-3) -> tuple[np.ndarray, int]:
    for i in range(1, max_iter + 1):
        jac = eval_jacobian(x, beta, function)
        err = eval_error(x, y, beta, function)
        delta_beta = gauss_newton_update(jac, err)
        beta += delta_beta
        if np.linalg.norm(delta_beta) < threshold:
            break
    if i == max_iter:
        raise UserWarning(f"Gauss-Newton method did not converge in {i} iterations.")
    return beta, i


## Calculate LM delta_beta
def lm_update(jacobian: np.ndarray,
              error: np.ndarray,
              lm_lambda: float) -> np.ndarray:
    # compute the parameter update delta_beta
    n = jacobian.shape[1]
    I = np.eye(n)
    delta_beta = np.zeros(jacobian.shape[1])
    A = ((jacobian.T @ jacobian) + (lm_lambda * I)) #Vectoris lm_lambda
    g = jacobian.T @ error  
    
    delta_beta = np.linalg.solve(A,g)
    
    return delta_beta



def levenberg_marquardt(x: np.ndarray,
                        y: np.ndarray,
                        beta: np.ndarray,
                        function: callable,
                        max_iter: int = 1000,
                        threshold: float = 1e-3) -> tuple[np.ndarray, int]:
    
    # compute initial parameter lambda
    lambda_i = 1
  
    for i in range(1, max_iter + 1):
        # compute jacobian, error, and delta_beta
        
        jacobian = eval_jacobian(x,beta,function)
        
        error = eval_error(x,y,beta,function)
        delta_beta = lm_update(jacobian, error, lambda_i) 
        
        new_beta = beta + delta_beta
        
        new_error = eval_error(x, y, new_beta, function)
    
        # compute quality measure rho
        rho = (np.linalg.norm(error)**2 - np.linalg.norm(new_error)**2) / (delta_beta.T @ (lambda_i * delta_beta + jacobian.T @ error)) #lambda_i is not a vector
        
        # update lambda according to quality measur                                                                 
        if rho > 0:
            beta = new_beta 
            lambda_i = max(lambda_i / 9, 10**-7)
            
        # update beta if quality measure is positive
        
        else:
            lambda_i = min(lambda_i * 11, 10**7)
            
        # check for early convergence
        if np.linalg.norm(delta_beta) < threshold: 
            print(f'Convergence at the iteration i: {i}') # Found the optimal delta_beta before final rounds
            return beta, i 

    if i == max_iter:
        raise UserWarning(f"Levenberg-Marquardt method did not converge in {i} iterations.")
    return beta, i


if __name__ == "__main__":
    data = pd.read_csv("breit_wigner.csv")
    beta_guess_list = [
        [100000, 100, 1000],
        [80000, 100, 700],
        [50000, 100, 700],
        [80000, 150, 1000],
        [80000, 70, 700],
        [10000, 50, 500],
        [1000, 10, 100],
        [1, 1, 1]
    ]

    print("Testing Gauss-Newton method:")
    for beta_guess in beta_guess_list:
        print("initial guess", beta_guess)
        try:
            beta, iterations = gauss_newton(data["x"],
                                            data["y"],
                                            beta_guess,
                                            breit_wigner,
                                            max_iter=500)
            print(f"-> converged in {iterations:3d} iterations")
            print("->", beta)
        except Exception:
            print("-> did not converge")

    print("")
    print("Testing Levenberg-Marquardt method:")
    for beta_guess in beta_guess_list:
        print("initial guess", beta_guess)
        try:
            beta, iterations = levenberg_marquardt(data["x"],
                                                   data["y"],
                                                   beta_guess,
                                                   breit_wigner)
            print(f"-> converged in {iterations:3d} iterations")
            print("->", beta)
        except UserWarning:
            print("-> did not converge")

    x_range = np.linspace(-50, 250, 1000)
    y_range = breit_wigner(x_range, *beta)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data["x"], data["y"], label="data")
    ax.plot(x_range, y_range, ":r", label="fit")
    ax.legend()
    plt.show()
