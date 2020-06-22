""" Lorenz 63 tangent Integrator classes.
----------
Contents
----------
- Integrator, class for integrating L63 equations and corresponding tangent
dynamics simultaneously. Uses RK4.

- TrajectoryObserver, class for observing the trajectory of the L63 integration.

- make_observations, function that makes many observations L96 tangent
integration.
"""
import numpy as np
import xarray as xr
from tqdm.notebook import tqdm

class Integrator:

    """Integrates the L63 ODEs and it's tangent dynamics simultaneously."""
    def __init__(self, a=10, b=8/3, c=28.0, dt = 0.001,
                 X_init=None, Y_init=None, Z_init=None, dx_init=None, dy_init=None, dz_init=None):

        # Model parameters
        self.a, self.b, self.c, self.dt = a, b, c, dt
        self.size = 3

        # Step counts
        self.step_count = 0 # Number of integration steps

        # Non-linear Variables
        self.X = np.random.rand() if X_init is None else X_init # Random IC if none given
        self.Y = np.random.rand() if Y_init is None else Y_init
        self.Z = np.random.rand() if Z_init is None else Z_init

        # TLE Variables
        self.dx = np.random.rand() if dx_init is None else dx_init # Random IC if none given
        self.dy = np.random.rand() if dy_init is None else dy_init
        self.dz = np.random.rand() if dz_init is None else dz_init

    def _rhs_X_dt(self, p_state):
        """Compute the right hand side of nonlinear variables.
        param, state: where we are in phase space"""
        
        [X, Y, Z] = p_state

        dXdt = self.a * (Y - X)
        dYdt = (self.c * X) - Y - (X * Z)
        dZdt = (X * Y) - (self.b * Z)
        return np.array([self.dt * dXdt, self.dt * dYdt, self.dt * dZdt])

    def _rhs_TX_dt(self, p_state, t_state):
        """Compute the right hand side of the linearised equation."""
        [X, Y, Z] = p_state
        [dx, dy, dz] = t_state
        
        dxdt = self.a * (dy - dx)
        dydt = (self.c - Z) * dx - dy - (X * dz)
        dzdt = (Y * dx) + (X * dy) - (self.b * dz)
        return np.array([self.dt * dxdt, self.dt * dydt, self.dt * dzdt])

    def _rhs_dt(self, p_state, t_state):
        return self._rhs_X_dt(p_state), self._rhs_TX_dt(p_state, t_state)

    def _step(self):
        """Integrate one time step"""

        # RK Coefficients
        k1_X, k1_TX = self._rhs_dt(self.p_state, self.t_state)
        k2_X, k2_TX = self._rhs_dt(self.p_state + (self.dt * k1_X)/2, self.t_state + (self.dt * k1_TX)/2)
        k3_X, k3_TX = self._rhs_dt(self.p_state + (self.dt * k2_X)/2, self.t_state + (self.dt * k2_TX)/2)
        k4_X, k4_TX = self._rhs_dt(self.p_state + (self.dt * k3_X), self.t_state + (self.dt * k3_TX))

        # Update State
        
        [del_X, del_Y, del_Z] =  1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        [del_dx, del_dy, del_dz] = 1 / 6 * (k1_TX + 2 * k2_TX + 2 * k3_TX + k4_TX)
        self.X += del_X
        self.Y += del_Y
        self.Z += del_Z
        self.dx += del_dx
        self.dy += del_dy
        self.dy += del_dz
        self.step_count += 1

    def integrate(self, time, noprog=True):
        """time: how long we integrate for in adimensional time."""
        steps = int(time / self.dt)
        for n in tqdm(range(steps), disable=noprog):
            self._step()

    def set_state(self, p_state, t_state):
        """x is [X, Y]. tangent_x is [dx, dy]"""
        [self.X, self.Y, self.Z] = p_state
        [self.dx, self.dy, self.dz] = t_state

    @property
    def p_state(self):
        """Where we are in phase space"""
        return np.array([self.X, self.Y, self.Z])

    @property
    def t_state(self):
        """Where we are in tangent space"""
        return np.array([self.dx, self.dy, self.dz])
    @property
    def state(self):
        return np.concatenate((self.p_state, self.t_state))
    
    @property
    def time(self):
        """a-dimensional time"""
        return self.dt * self.step_count

    @property
    def parameter_dict(self):
        param = {
        'a': self.a, 
        'b': self.b,
        'c': self.c,
        'dt': self.dt
        }
        return param

    def reset_count(self):
        """Reset Step count"""
        self.step_count = 0
        
#-----------------------------------------------------------------------------------------   
# Pertubed Inegrator Class
#-----------------------------------------------------------------------------------------   
        
class PerturbedIntegrator:
    """Integrates the perturbed L63 ODEs and it's tangent dynamics simultaneously.
    param, perturbation_field, function of phase space and time that perturbs oridnary dynamics.
    param, t_perturbation_field, function of phase space, tangent space and time that perturbs tangent dynamics.
    """
    def __init__(self, a=10, b=8/3, c=28.0, dt = 0.001,
                 perturbation_field=None, p_args=[], t_perturbation_field=None, tp_args=[], perturbation_on=False,
                 X_init=None, Y_init=None, Z_init=None, dx_init=None, dy_init=None, dz_init=None):

        # Model parameters
        self.a, self.b, self.c, self.dt = a, b, c, dt
        self.size = 3
        
        # Perturbation Variables
        self.perturbation_field = perturbation_field # arguments should always look like (p_state, time, *p_args)
        self.p_args = p_args
        self.t_perturbation_field = t_perturbation_field # arguments should always look like (p_state, t_state, time, *tp_args)
        self.tp_args = tp_args
        self.perturbation_on = perturbation_on

        # Step counts
        self.step_count = 0 

        # Non-linear Variables
        self.X = np.random.rand() if X_init is None else X_init # Random IC if none given
        self.Y = np.random.rand() if Y_init is None else Y_init
        self.Z = np.random.rand() if Z_init is None else Z_init

        # TLE Variables
        self.dx = np.random.rand() if dx_init is None else dx_init 
        self.dy = np.random.rand() if dy_init is None else dy_init
        self.dz = np.random.rand() if dz_init is None else dz_init

    def _rhs_X_dt(self, p_state, time):
            """Compute the right hand side of nonlinear variables.
            param, state: where we are in phase space
            param, time: perturbation field may by non-autonomous"""

            [X, Y, Z] = p_state

            dXdt = self.a * (Y - X)
            dYdt = (self.c * X) - Y - (X * Z)
            dZdt = (X * Y) - (self.b * Z)
            rhs = np.array([dXdt, dYdt, dZdt])

            if (self.perturbation_on):
                rhs += self.perturbation_field(p_state, time, *self.p_args)  

            return self.dt * rhs # Notice dt multiplication

    def _rhs_TX_dt(self, p_state, t_state, time):
        """Compute the right hand side of the linearised equation."""
        [X, Y, Z] = p_state
        [dx, dy, dz] = t_state
        
        dxdt = self.a * (dy - dx)
        dydt = (self.c - Z) * dx - dy - (X * dz)
        dzdt = (Y * dx) + (X * dy) - (self.b * dz)
        rhs = np.array([dxdt, dydt, dzdt])
        
        if (self.perturbation_on):
                rhs += self.t_perturbation_field(p_state, t_state, time, *self.tp_args)
        return self.dt * rhs

    def _rhs_dt(self, p_state, t_state, time):
        return self._rhs_X_dt(p_state, time), self._rhs_TX_dt(p_state, t_state, time)

    def _step(self):
        """Integrate one time step"""

        # RK Coefficients
        k1_X, k1_TX = self._rhs_dt(self.p_state, self.t_state, self.time)
        k2_X, k2_TX = self._rhs_dt(self.p_state + (self.dt * k1_X)/2, self.t_state + (self.dt * k1_TX)/2,
                                   self.time + self.dt/2)
        k3_X, k3_TX = self._rhs_dt(self.p_state + (self.dt * k2_X)/2, self.t_state + (self.dt * k2_TX)/2,
                                   self.time + self.dt/2)
        k4_X, k4_TX = self._rhs_dt(self.p_state + (self.dt * k3_X), self.t_state + (self.dt * k3_TX), 
                                   self.time + self.dt)

        # Update State
        
        [del_X, del_Y, del_Z] =  1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        [del_dx, del_dy, del_dz] = 1 / 6 * (k1_TX + 2 * k2_TX + 2 * k3_TX + k4_TX)
        self.X += del_X
        self.Y += del_Y
        self.Z += del_Z
        self.dx += del_dx
        self.dy += del_dy
        self.dy += del_dz
        self.step_count += 1

    def integrate(self, time, noprog=True):
        """time: how long we integrate for in adimensional time."""
        steps = int(time / self.dt)
        for n in tqdm(range(steps), disable=noprog):
            self._step()

    def set_state(self, p_state, t_state):
        """x is [X, Y]. tangent_x is [dx, dy]"""
        [self.X, self.Y, self.Z] = p_state
        [self.dx, self.dy, self.dz] = t_state

    @property
    def p_state(self):
        """Where we are in phase space"""
        return np.array([self.X, self.Y, self.Z])

    @property
    def t_state(self):
        """Where we are in tangent space"""
        return np.array([self.dx, self.dy, self.dz])
    @property
    def state(self):
        return np.concatenate((self.p_state, self.t_state))
    
    @property
    def time(self):
        """a-dimensional time"""
        return self.dt * self.step_count

    @property
    def parameter_dict(self):
        param = {
        'a': self.a, 
        'b': self.b,
        'c': self.c,
        'dt': self.dt
        }
        return param

    def reset_count(self):
        """Reset Step count"""
        self.step_count = 0

#-----------------------------------------------------------------------------------------   
# Trajectory Observer Class
#-----------------------------------------------------------------------------------------  
        
class TrajectoryObserver():
    """Observes the trajectory of L63 ODE integrator. Dumps to netcdf."""

    def __init__(self, integrator, name='L63 Trajectory'):
        """param, integrator: integrator being observed."""

        # Need knowledge of the integrator
        self._parameters = integrator.parameter_dict

        # Trajectory Observation logs
        self.time_obs = [] # Times we've made observations
        self.state_obs = []

    def look(self, integrator):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self.time_obs.append(integrator.time)

        # Making Observations
        self.state_obs.append(integrator.p_state.copy())
        
    def wipe(self):
        self.time_obs = [] # Times we've made observations
        self.state_obs = []

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.state_obs) == 0):
            print('I have no observations! :(')
            return

        _time = self.time_obs
        x = xr.DataArray(np.array(self.state_obs)[:, 0], dims=['Time'], name = 'X', coords = {'Time': _time})
        y = xr.DataArray(np.array(self.state_obs)[:, 1], dims=['Time'], name = 'Y', coords = {'Time': _time})
        z = xr.DataArray(np.array(self.state_obs)[:, 2], dims=['Time'], name = 'Z', coords = {'Time': _time})
        trajectory = xr.Dataset({'X': x, 'Y': y, 'Z': z})

        return trajectory

def make_observations(runner, looker, obs_num, obs_freq, noprog=False):
    """Makes observations given runner and looker.
    runner, integrator object.
    looker, observer object.
    obs_num, how many observations you want.
    obs_freq, adimensional time between observations"""
    for step in tqdm(np.repeat(obs_freq, obs_num), disable=noprog):
        runner.integrate(obs_freq)
        looker.look(runner)