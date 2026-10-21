import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import control as ct

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


# ============================================================
# Utility Helpers
# ============================================================
def build_mixer_matrix(L, mu):
    """Build the 4x4 mixer matrix M."""
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [L, 0.0, -L, 0.0],
            [0.0, L, 0.0, -L],
            [mu, -mu, mu, -mu],
        ],
        dtype=np.float64,
    )


def make_env():
    landing_params = dict(
    z_max=0.05,
    l_max=0.25,
    phi_max=np.deg2rad(10.0),
    theta_max=np.deg2rad(10.0),
    v_l_max=0.5,
    v_z_max=0.10,
    T_max=10.0,
)
    quad_params = dict(
        m=1.2,
        Ix=0.22,
        Iy=0.22,
        Iz=0.44,
        g=9.81,
        M=build_mixer_matrix(L=0.3, mu=1.0),
        L=0.3,
        K=np.eye(4),
    )
    env = QuadLandingEnv(quad_params, landing_params)
    return QuadLandingSB3Wrapper(env)


# ============================================================
# Dynamics
# ============================================================
def lin_sys_matrices(quad_params):
    A = np.zeros([12, 12])
    A[[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]] = 1
    A[3, 7], A[4, 6] = quad_params["g"], -quad_params["g"]

    B_tilda = np.zeros([12, 4])
    B_tilda[5, 0] = 1 / quad_params["m"]
    B_tilda[9, 1] = quad_params["L"] / quad_params["Ix"]
    B_tilda[10, 2] = quad_params["L"] / quad_params["Iy"]
    B_tilda[11, 3] = 1 / quad_params["Iz"]

    #B = B_tilda @ quad_params["M"] @ quad_params["K"]
    M=quad_params["M"]
    K=quad_params["K"]
    #desired_eig = np.full(12, -6)
    #theta_d = ct.place_varga(A, -B, desired_eig)
    #a_h = A + B @ theta_d
    #_p = ct.lyap(a_h, 2 * np.eye(12))

    return dict(A=A, B_tilda=B_tilda,M=M,K=K)


def quad_dynamics_52(x, u, params):
    """
    Nonlinear quadrotor dynamics.
    State x in R^12, input u in R^4.
    """
    m = params["m"]
    Ix = params["Ix"]
    Iy = params["Iy"]
    Iz = params["Iz"]
    g = params.get("g", 9.81)

    M = params["M"]  # 4x4
    K = params["K"]  # 4x4
    L = 0.3

    px, py, pz, vx, vy, vz, phi, theta, psi, phi_d, theta_d, psi_d = x

    fz, tau_phi, tau_theta, tau_psi = M @ K @ u

    ax = (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * fz / m
    ay = (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * fz / m
    az = (np.cos(phi) * np.cos(theta) * fz / m) - g

    phi_ddot = theta_d * psi_d * (Iy - Iz) / Ix + tau_phi * L / Ix
    theta_ddot = phi_d * psi_d * (Iz - Ix) / Iy + tau_theta * L / Iy
    psi_ddot = phi_d * theta_d * (Ix - Iy) / Iz + tau_psi / Iz

    xdot = np.zeros(12)
    xdot[0:3] = [vx, vy, vz]
    xdot[3:6] = [ax, ay, az]
    xdot[6:9] = [phi_d, theta_d, psi_d]
    xdot[9:12] = [phi_ddot, theta_ddot, psi_ddot]
    return xdot


def lin_dynamics(x, u, lin_params):
    A, B_tilda,M,K= lin_params["A"], lin_params["B_tilda"],lin_params["M"],lin_params['K']
    u_in=M@K@u -(np.array([1.2*9.81,0,0,0])).T
    xdot = A @ x + B_tilda @ u_in
    return xdot


def rk4_step(f, x, u, dt, params):
    k1 = f(x, u, params)
    k2 = f(x + 0.5 * dt * k1, u, params)
    k3 = f(x + 0.5 * dt * k2, u, params)
    k4 = f(x + dt * k3, u, params)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================
# Platform + Landing Logic
# ============================================================
class Platform:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)

    def step(self, dt):
        self.pos += self.vel * dt

    def state(self):
        return {"pos": self.pos.copy(), "vel": self.vel.copy()}


def in_landing_box(x, platform_state, p):
    px, py, pz, vx, vy, vz, phi, theta, *_ = x
    Xp, Yp, Zp = platform_state["pos"]

    dx = px - Xp
    dy = py - Yp
    dz = pz - Zp
    dxy = np.sqrt(dx * dx + dy * dy)
    v_l = np.sqrt(vx * vx + vy * vy)

    conds = {
        "dz": dz <= p["z_max"] and dz >= 0.0,
        "dxy": dxy <= p["l_max"],
        "phi": abs(phi) <= p["phi_max"],
        "theta": abs(theta) <= p["theta_max"],
        "v_l": v_l <= p["v_l_max"],
        "v_z": abs(vz) <= p["v_z_max"],
    }

    return all(conds.values()), conds, {
        "dz": dz,
        "dxy": dxy,
        "phi": phi,
        "theta": theta,
        "v_l": v_l,
        "v_z": vz,
    }


def landing_cost_and_done(x, platform_state, t, params, delta_u, crashed=False):
    box, conds, vals = in_landing_box(x, platform_state, params)

    dxy = vals["dxy"]
    dz = vals["dz"]
    phi_a = abs(vals["phi"])
    theta_a = abs(vals["theta"])
    v_l = vals["v_l"]
    v_z = abs(vals["v_z"])

    dz_abs = abs(dz)
    dxy_n = dxy / (4 * params["l_max"])
    dz_n = dz_abs / (10 * params["z_max"])

    c_dist = -5 + (2.5 * dxy + 10 * (phi_a + theta_a))
    d = dxy_n**2 + 4* dz_n**2
    c_land = -15 / (1 + d) 

    c_u = 0.25 * np.linalg.norm(delta_u)
    cost = c_dist + c_u + c_land

    if (
        crashed
        or dxy > 4
        or dz > 4
        or theta_a > 0.8
        or phi_a > 0.8
        or dz <= -0.02
        or t >= params["T_max"]
    ):
        """
        if dxy>4 or dz>4 or dz<=-0.02:
            print(f"failed due to dxy: {dxy} or dz :{dz}")
        elif theta_a >0.8 or phi_a>0.8:
            print(f"failed due to angels, theta :{theta_a} phi {phi_a}")
        elif t>=params['T_max']:
            print(f"failed due to time {t}")
        """
        cost = 50
        return cost, True, False

    if box:
        cost = -5000 +2000*(t/params['T_max'])
        print("Success")
        return cost, True, True

    return cost, False, False


# ============================================================
# Environment
# ============================================================
class QuadLandingEnv:
    def __init__(self, quad_params, landing_params, dt_sim=0.001, dt_ctrl=0.05):
        self.params = quad_params
        self.lin_params = lin_sys_matrices(self.params)
        self.lp = landing_params

        self.dt_sim = dt_sim
        self.dt_ctrl = dt_ctrl
        self.N = int(dt_ctrl / dt_sim)

        self.u_hold = np.zeros(4)
        self.t = 0.0
        self.platform = Platform(pos=[0.0, 0.0, 0.0], vel=[0.1, 0.0, 0.0])

        self.w_hov = np.ones(4)*(quad_params["m"] * 9.81) / 4
        self.w_max = 2.5 * self.w_hov
        self.d_w_max = self.w_max - self.w_hov

        self.success = 0
        self.ep = 0
        self.reset()

    def reset(self):

        self.max_phi=0
        self.max_theta=0,
        self.max_psi=0
        self.platform.pos[0:2] = np.random.uniform([-1,-1],[1,1])
        self.platform.vel[0:2] =np.random.uniform([0.05,0.05],[0.2,0.2])

        self.x,self.x_hov = np.zeros(12),np.zeros(12)
        self.x[0:3] = np.random.uniform([-2, -2, 0.35], [2, 2, 3])
        self.x_hov[0:3]=self.x[0:3]

        #self.x_inital=self.x[0:3].copy()
        #self.pv_inital=self.platform.vel[0:2].copy()
        #self.px_inital=self.platform.pos[0:2].copy()

        #self.dx_inital= np.linalg.norm(self.px_inital-self.x_inital[0:2].copy())

        self.delta_x=self.x-self.x_hov
        self.x_ref=self.delta_x + self.x_hov

        self.t = 0.0
        self.prev_u = self.w_hov

        self.ep += 1
        if self.ep % 100 == 0:
            print(f" Episode: {self.ep},Successs over last 100 epsidoes {self.success}/100")
            self.success = 0

        return self._get_obs()

    def _get_obs(self):
        px, py, pz, vx, vy, vz = self.x[0:6]
        phi, theta, psi, phi_d, theta_d, psi_d = self.x[6:12]

        Xp, Yp, Zp = self.platform.pos
        Vxp, Vyp, Vzp = self.platform.vel

        obs = np.array(
            [
                px,
                py,
                pz,
                vx,
                vy,
                vz,
                phi,
                theta,
                psi,
                phi_d,
                theta_d,
                psi_d,
                Xp,
                Yp,
                Zp,
                Vxp,
                Vyp,
                Vzp,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, u):
        #refrence input 
        u = np.asarray(u, dtype=float).reshape(4,)
        u=np.clip(u,-1.0,1.0)
        self.curr_u=u
        omega= self.w_hov +self.d_w_max*u  #map action from [-1,1] to w_hover + delta
        #self.delta_u=omega

        self.u_hold[:] = omega

        self.crashed = False

        for _ in range(self.N):
            self.platform.step(self.dt_sim)
            self.x = rk4_step(
                quad_dynamics_52,
                self.x,
                self.u_hold,
                self.dt_sim,
                self.params,
            )
            self.delta_x = rk4_step(
                lin_dynamics,
                self.delta_x,
                self.u_hold,
                self.dt_sim,
                self.lin_params,
            )
            x_lin_full = self.x_hov + self.delta_x
            e=self.x[:3]-x_lin_full[:3]
            pos_err = self.x[:3] - x_lin_full[:3]
            vel_err = self.x[3:6] - x_lin_full[3:6]
            ang_err = self.x[6:9] - x_lin_full[6:9]
            rate_err = self.x[9:12] - x_lin_full[9:12]

            self.max_phi=max(self.max_phi,self.x[6])
            self.max_theta=max(self.max_theta,self.x[7])
            self.max_psi=max(self.max_psi,self.x[8])


            #print("||pos_err|| =", np.linalg.norm(pos_err))
            #print("vel err=", np.linalg.norm(vel_err))
            #print("ang err  =", np.linalg.norm(ang_err))
            #print("rate_err     =", np.linalg.norm(rate_err))
            #print("delta_u     =", np.linalg.norm(self.delta_u))

            if (self.x[2] <= -0.02):
                self.crashed = True
                #print("Failed due to crash")
                break

            self.t += self.dt_sim
            if self.t >= self.lp["T_max"]:
                break

        du = self.curr_u - self.prev_u
        cost, done, success = landing_cost_and_done(
            self.x,
            self.platform.state(),
            self.t,
            self.lp,
            delta_u=du,
            crashed=self.crashed,
        )

        reward = -cost
        info = {"success": success, "time": self.t}

        self.prev_u = self.curr_u
        if success:
            self.success += 1
            print(f"e: {np.linalg.norm(e)}, max phi: {self.max_phi},max psi: {self.max_psi}, max theta :{self.max_theta}")
        #elif done and not success :
            #print(f'Inital pos: {self.x_inital}\n Inital platform pos: {self.px_inital}\n Inital dx: {self.dx_inital}\n Inital platform vel: {self.pv_inital}\n')

        return self._get_obs(), reward, done, info

    def controller_dynamics(self, theta_b, v_b, ea, e, du, Phi_reg, params):
        # b: bar, d: dot
        gamma, beta, mue = params["gamma"], params["beta"], params["mue"]
        B, P, A_H = params["B"], params["P"], params["A_H"]

        K_a = theta_b[:, 6:]
        Phi_bar = [Phi_reg.T, du.T].T
        N = 1 + mue * Phi_bar.T @ Phi_bar
        eu = e - ea

        ea_d = A_H @ ea + B @ K_a @ du
        v_d = -gamma * B.T @ P @ eu @ Phi_bar.T
        theta_d = -beta * (theta_b - v_b) @ N

        return v_d, theta_d, ea_d

    def rk_4_cd(self, theta_b, v_b, ea, e, du, Phi_reg, params, dt):
        k1_v, k1_theta, k1_ea = self.controller_dynamics(theta_b, v_b, ea, e, du, Phi_reg, params)
        k2_v, k2_theta, k2_ea = self.controller_dynamics(self, theta_b + 0.5 * k1_theta * dt, v_b + 0.5 * k1_v * dt, du, e, Phi_reg, params)
        k3_v, k3_theta, k3_ea = self.controller_dynamics(self, theta_b + 0.5 * k2_theta * dt, v_b + 0.5 * k2_v * dt, du, e, Phi_reg, params)
        k4_v, k4_theta, k4_ea = self.controller_dynamics(self, theta_b + k3_theta * dt, v_b + k3_v * dt, du, e, Phi_reg, params)

        v_next = v_b + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        theta_next = theta_b + dt * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6
        ea_next = ea + dt * (k1_ea + 2 * k2_ea + 2*k3_ea + k4_ea) / 6
        return v_next, theta_next, ea_next

    def AC(theta, Phi_reg):
        return theta @ Phi_reg


# ============================================================
# SB3 Wrapper
# ============================================================
class QuadLandingSB3Wrapper(gym.Env):
    def __init__(self, base_env):
        super().__init__()
        self.env = base_env

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return obs.astype(np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return obs.astype(np.float32), reward, terminated, truncated, info


if __name__ == "__main__":
    quad_params = dict(
        m=1.2,
        Ix=0.22,
        Iy=0.22,
        Iz=0.44,
        g=9.81,
        M=build_mixer_matrix(L=0.3, mu=1.0),
        L=0.3,
        K=np.eye(4),
    )

    venv = make_vec_env(make_env, n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    venv = VecMonitor(venv)

    Resume = True

    if Resume:
        venv = VecNormalize.load("vecnormalize_30_m_new.pkl", venv)
        venv.training = True
        venv.norm_reward = True

        model = PPO.load(
            "ppo_quad_landing_30_m_new",
            venv,
            device="cpu",
        )
    else:
        venv = VecNormalize(
            venv,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=0.99,
        )

        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=7,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=1e-4,
            target_kl=0.02,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=torch.nn.Tanh,
            ),
            device="cpu",
            tensorboard_log="./tb_logs/quad_read/",
            verbose=2,
        )

    model.learn(total_timesteps=10_000_000, tb_log_name="quad_landing", reset_num_timesteps=False)
    model.save("ppo_quad_landing_40_m_new")
    venv.save("vecnormalize_40_m_new.pkl")
