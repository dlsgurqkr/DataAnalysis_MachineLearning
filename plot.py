import pandas as pd
import matplotlib.pyplot as plt


def create_contour_plot():
    # read and transform your data here
    data = pd.read_csv("/Users/parkinhyuk/Desktop/TUM/Python for Engineering Data Analysis - From Machine Learning to Visualization/fes.csv")
    # ...
    X = data['CV1'].values
    Y = data['CV2'].values
    Z = data['free energy (kJ/mol)'].values

    valid_mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[valid_mask], Y[valid_mask], Z[valid_mask]
    
    #Grid reshape 1D to 2D. (grid_szie, grid_size) <- turple 
    grid_size = int(np.sqrt(len(X)))
    X = X[:grid_size*grid_size].reshape((grid_size,grid_size)) 
    Y = Y[:grid_size*grid_size].reshape((grid_size,grid_size)) 
    Z = Z[:grid_size*grid_size].reshape((grid_size,grid_size)) 
    levels = np.linspace(Z.min(),Z.max(),num=20)
    
    # create your contour plot here
    fig = plt.figure()
    contour = fig.contour(X, Y, Z, levels = levels, cmap='CMRmap')
    
    fig.set_xlabel('CV1 (units)')
    fig.set_ylabel('CV2 (units)')
    fig.set_title('2D Contour Plot of Free Energy Surface')
    
    cbar1 = plt.colorbar(contour)
    cbar1.set_label('colorbar')
    
    # save your plot
    plt.savefig("contour.png")


def create_surface_plot():
    # read and transform your data here
    data = pd.read_csv("fes.csv")
    # ...
    X = data['CV1'].values
    Y = data['CV2'].values
    Z = data['free energy (kJ/mol)'].values

    valid_mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[valid_mask], Y[valid_mask], Z[valid_mask]
    
    #Grid reshape 1D to 2D. (grid_szie, grid_size) <- turple 
    grid_size = int(np.sqrt(len(X)))
    X = X[:grid_size*grid_size].reshape((grid_size,grid_size)) 
    Y = Y[:grid_size*grid_size].reshape((grid_size,grid_size)) 
    Z = Z[:grid_size*grid_size].reshape((grid_size,grid_size)) 
    
    # create your 3D surface plot here
    fig = plt.figure()
    # ...
    contour = fig.plot_surface(X, Y, Z, vmin=Z.min(), vmax=Z.max(), cmap='CMRmap')
    
    fig.set_xlabel('CV1')
    fig.set_ylabel('CV2')
    fig.set_zlabel('Free Energy')
    fig.set_title('3D Contour Plot of Free Energy Surface')
    
    cbar2 = plt.colorbar(contour)
    cbar2.set_label('colorbar')
    # save your plot
    plt.savefig("surface.png")


if __name__ == "__main__":
    create_contour_plot()
    create_surface_plot()
