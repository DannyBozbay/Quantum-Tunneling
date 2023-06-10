import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
plt.style.use('seaborn')
plt.style.use("dark_background")


class Gaussian_Wave:

    def __init__(self, N_grid, L, a, V0, w, x0, k0, sigma, t):
        self.t = t  # time values
        self.L = L  # length of grid
        self.N_grid = N_grid  # number of grid points

        self.x = np.linspace(-self.L, self.L, self.N_grid + 1)  # grid of points
        self.dx = self.x[1] - self.x[0]  # grid point spacing or 'discrete' analogue of the differential length

        def integral(f, axis=0):
            """This function allows us to approximate integrals in discrete space"""
            return np.sum(f * self.dx, axis=axis)

        self.Psi0 = np.exp(-1 / 2 * (self.x[1:-1] - x0) ** 2 / sigma ** 2) * np.exp(1j * k0 * self.x[1:-1])
        # use this range for x because as mentionned, we need the wavefunction to be 0 at the endpoints of the grid.

        # normalise the initial state
        norm = integral(np.abs(self.Psi0) ** 2)
        self.Psi0 = self.Psi0 / np.sqrt(norm)

        # kinetic energy
        self.T = -1 / 2 * 1 / self.dx ** 2 * (
                np.diag(-2 * np.ones(self.N_grid - 1)) + np.diag(np.ones(self.N_grid - 2), 1) +
                np.diag(np.ones(self.N_grid - 2), -1))

        # potential as a flat array
        self.V_flat = np.array([V0 if a < pos < a + w else 0 for pos in self.x[1:-1]])

        # potential energy as a diagonal matrix
        self.V = np.diag(self.V_flat)

        # Hamiltonian
        self.H = self.T + self.V

    # solve the eigenvalue problem and get the time-dependent wavefunction
    def animation(self):
        def integral(f, axis=0):
            """This function allows us to approximate integrals in discrete space"""
            return np.sum(f * self.dx, axis=axis)

        # get eigenvalues and eigenvectors and normalise
        E, psi = np.linalg.eigh(self.H)
        psi = psi.T
        norm = integral(np.abs(psi) ** 2)
        psi = psi / np.sqrt(norm)

        # get expansion coeffs
        c_n = np.zeros_like(psi[0], dtype=complex)

        for j in range(0, self.N_grid - 1):
            c_n[j] = integral(np.conj(psi[j]) * self.Psi0)  # for each eigenvector, compute the inner product

        # check that the probabilities sum to 1:
        sum_prob = np.linalg.norm(c_n)
        sum_prob = round(sum_prob, 3)  # round to 3 decimal places
        print("Total probability = ", sum_prob)

        # get a function that returns the time dependent wavefunction
        def Psi(t):
            return psi.T @ (c_n * np.exp(-1j * E * t))

        # get a function that returns the time dependent probabilities
        def Prob_Psi(t):
            return np.abs(Psi(t)) ** 2

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 5), facecolor=(1, 1, 1))

        # Lines we wish to animate
        line1, = ax.plot([], [], lw=1, color='blue', label=r'$\Re(\Psi)$') # Real part of Psi
        line2, = ax.plot([], [], lw=1, color='green', label=r'$\Im(\Psi)$') # Imag part of Psi
        line3, = ax.plot([], [], lw=1, color='white', label=r'$\sqrt{P(\Psi)}$') # Square root of probability amplitude

        ax.plot(self.x[1:-1], self.V_flat, lw=.5, color='red', label=r'$V(x)$')  # Plot potential
        ax.fill_between(self.x[1:-1], self.V_flat, color='red', alpha=.5)  # Shade under potential
        ax.set_xlim([-self.L * 0.70, self.L * 0.70]) # set limits of x-axis
        ax.set_ylim([-0.3, .7]) # set limits of y-axis
        ax.set_xlabel(r'$x$', fontsize=15) # label of x-axis
        ax.legend(loc='best') # plot legend
        ax.grid(False) # Hide grid lines
        ax.axis('off')

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            return line1, line2, line3,

        def animate(t):
            """This is the function that we will call to animate"""

            y1 = np.real(Psi(t)) # real part of Psi
            y2 = np.imag(Psi(t)) # imaginary part of Psi
            y3 = np.sqrt(Prob_Psi(t)) # Envelope of Psi

            line1.set_data(self.x[1:-1], y1)
            line2.set_data(self.x[1:-1], y2)
            line3.set_data(self.x[1:-1], y3)

            return line1, line2, line3,


        ani = FuncAnimation(fig, animate, frames=self.t, init_func=init, blit=True, repeat=True)

        plt.close()

        return ani


wavepacket = Gaussian_Wave(N_grid=5000, L=350, a=0, V0=.5,
                           w=5, x0=-200, k0=1, sigma=13, t=np.linspace(0, 600, 600))

Psi = wavepacket.animation()

Writer = writers['ffmpeg']
Writer = Writer(fps=60)
Psi.save("Psi.mp4", Writer)
