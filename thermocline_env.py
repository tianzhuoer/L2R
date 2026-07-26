import math
import time
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from scipy.interpolate import interp1d
import pygame
from scipy.stats import norm
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from stable_baselines3.common.env_checker import check_env
from sklearn.decomposition import PCA
from utils.ctd_data import CTDDataProcessor
from config import *




class ThermoTrackEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20 }    

    def __init__(self,render_mode=None):
        self.ctd_processor = CTDDataProcessor(folder_path=CTD_FOLDER_PATH)  # Initialize the CTDDataProcessor

        self.istraining=True        

        # window
        self.win_width,self.win_height=1000, 600  
        # color
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255) 


        # state bound
        self.min_depth=DEPTH_MIN
        self.max_depth=DEPTH_MAX


        self.min_T=0
        self.max_T=40

        self.min_S=0.0
        self.max_S=40.0


        self.grid_num = 50

        self.sample_window_size = SAMPLE_WINDOW_SIZE

        self.belief_window_size = BELIEF_WINDOW_SIZE

        self.depth_grid = np.linspace(self.min_depth, self.max_depth, self.grid_num).reshape(-1, 1)


        self.obs_dim_per_step = 1 + self.sample_window_size * 2

        lower_single = np.array([self.min_depth] + [self.min_T]*self.sample_window_size + [self.min_S]*self.sample_window_size)
        upper_single = np.array([self.max_depth] + [self.max_T]*self.sample_window_size + [self.max_S]*self.sample_window_size)


        self.lower=np.tile(lower_single, (self.belief_window_size, 1))
        self.upper= np.tile(upper_single, (self.belief_window_size, 1))


        self.v=STEP_SIZE


        self.observation_space = spaces.Box(low=self.lower, high=self.upper, shape=(self.belief_window_size, self.obs_dim_per_step), dtype=np.float64)




        self.action_space = spaces.Discrete(2)
        self.action_pre=None


        self.episode_length =0
        self.max_episode_length = MAX_EPISODES_LEN

        filename, sampled_T_interp, sampled_S_interp, sampled_time_grid_datetime, depth_grid = self.ctd_processor.random_sample()
        self.T_all= self.interp_data(sampled_T_interp.T)
        self.T=self.T_all[:,self.episode_length]
        self.d=-depth_grid.reshape(-1,1)



        self.render_final_only = True

        self.belief_history = []

        self.consecutive_high_grad_count = 0

        self.max_obs_grad=0

        self.episode_return = 0


        self.llm_depth: float = float('nan')
        self._prev_dist_to_llm: float = float('nan')
        self._cached_max_grad_depth: float = float('nan')
        self._grad_cache_step: int = 0
        self._grad_cache_interval: int = 20

        # render settings
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None


    def reset(self, seed=None, options=None):

        self.episode_length = 0
        super().reset(seed=seed)

        filename, sampled_T_interp, sampled_S_interp, sampled_time_grid_datetime, depth_grid = \
            self.ctd_processor.random_sample()
        if sampled_T_interp is None:
            from utils.ctd_data import CTDDataProcessor as _CDP
            filename, sampled_T_interp, sampled_S_interp, sampled_time_grid_datetime, depth_grid = \
                _CDP.random_sample(self.ctd_processor)

        self.current_filename = filename

        self.T_all = self.interp_data(sampled_T_interp.T)
        self.S_all = self.interp_data(sampled_S_interp.T)
        self.T = self.T_all[:, self.episode_length]
        self.S = self.S_all[:, self.episode_length]
        self.d = -depth_grid.reshape(-1, 1)
        self.episode_start_datetime = sampled_time_grid_datetime[0]
        self.true_thermo_depth_seq = self._compute_thermo_depth_seq()


        if self.istraining:
            try:

                depth_grid_flat = self.depth_grid.flatten()
                # range1 = depth_grid_flat[depth_grid_flat >= -15 & (depth_grid_flat <= -5)]
                # range2 = depth_grid_flat[(depth_grid_flat >= -245) & (depth_grid_flat <= -235)]
                # depth_options = np.concatenate((range1, range2))


                depth0 = random.choice(depth_grid_flat)   
                # print("Initial depth selected during training:", depth0)
                # depth0 = random.choice(self.depth_grid.flatten())
            except ValueError as e:
                print(self.T_all.shape)
                print(self.episode_length)
                print(f"ValueError in file '{filename}': {e}")

                depth0 = float(self.depth_grid[len(self.depth_grid)//2])
        else:
            if options is not None and 'initial_depth' in options:
                depth0 = float(options['initial_depth'])
            else:
                depth_grid_flat = self.depth_grid.flatten()
                depth0 = random.choice(depth_grid_flat)


        self.T_observed_mask = np.zeros(self.grid_num + 1, dtype=bool)
        self.T_observed_mask[0] = True
        self.T_observed = []
        self.S_observed = []
        self.d_observed = []

        PO0, T_observed_mask, max_grad_depth0, max_grad0, new_belief = self.POcalc(depth0)

        self.T_observed_mask=T_observed_mask
        # self.max_grad_depth = max_grad_depth0
        # self.max_grad = max_grad0


        self.consecutive_high_grad_count = 0
        self._prev_dist_to_llm = float('nan')
        self._cached_max_grad_depth = float('nan')
        self._grad_cache_step = 0

        self.episode_return = 0


        self.state_belief_queue = []

        self.state_belief_queue.append((depth0, PO0.copy()))

        if len(self.state_belief_queue) > self.belief_window_size:
            self.state_belief_queue.pop(0)



        state_flattened = np.zeros((BELIEF_WINDOW_SIZE, self.obs_dim_per_step), dtype=np.float64)
        for i in range(len(self.state_belief_queue)):
            depth_val  = self.state_belief_queue[i][0]
            data_vector = self.state_belief_queue[i][1]   # T(15)+S(15)=30
            state_flattened[i, :] = np.concatenate([[depth_val], data_vector])
        self.state = state_flattened


        self.depth_history = []
        self.feature_history = []

        Observe_last_idx=np.where(self.state.sum(axis=1)!=0)[0]
        self.feature_history.append(self.state[Observe_last_idx,:])

        self.mask_history = []
        self.mask_history.append(self.T_observed_mask.copy())
        

        self.belief_history = []

        self.belief_history.append(self.state_belief_queue.copy())

        # info
        info = {'mask': self.T_observed_mask.copy()}


        if self.render_mode == "human":
            self._render_frame()
        return self.state, info

    
    def step(self, action):
        

        assert self.action_space.contains(action), f"Invalid action {action}"
        Observe_last_idx=np.where(self.state.sum(axis=1)!=0)[0]
        depth= self.state[Observe_last_idx[-1],0]
        # a=0, down
        # a=1, keep
        # a=2, up

        # if action in [0, 1]:
        #     depth_new_temp=depth+self.v*(action-0.5)*2
        # elif action in [2, 3]:
        #     depth_new_temp=depth+self.v*(action-2.5)*2*1.5

        depth_new_temp=depth+self.v*(action-0.5)*2

        depth_new=np.clip(depth_new_temp, self.d.min(), self.d.max(), out=None)
        # print(f'depth_new: {depth_new},action: {action}')
        # if depth_new==depth:

        #     action = 1 - action
        #     depth_new_temp = depth + self.v * (action - 0.5) * 2
        #     depth_new = np.clip(depth_new_temp, self.d.min(), self.d.max(), out=None)
            

        # t1=bool(depth_new_temp>=self.max_depth or depth_new_temp<=self.min_depth) 
        t1=False
        t2=bool(self.episode_length>=self.max_episode_length)
        terminated =t2 or t1
        truncated = False



        # R
        reward = self.reward(action)
        self.action_pre=action
        self.episode_return += reward

        info = {'mask': self.T_observed_mask.copy()}
        if terminated:
            return self.state, reward, terminated, truncated, info



        if depth_new==depth:

            self.state_belief_queue.append((depth, self.state_belief_queue[-1][1].copy()))
            self.state_belief_queue.pop(0)
            self.belief_history.append(self.state_belief_queue.copy())

            state_flattened = np.zeros((BELIEF_WINDOW_SIZE, self.obs_dim_per_step), dtype=np.float64)
            for i in range(len(self.state_belief_queue)):
                depth_val   = self.state_belief_queue[i][0]
                data_vector = self.state_belief_queue[i][1]
                state_flattened[i, :] = np.concatenate([[depth_val], data_vector])
            self.state = state_flattened

        else:
            PO, T_observed_mask, max_grad_depth, max_grad, new_belief = self.POcalc(depth_new, depth_last=depth)
            self.T_observed_mask = T_observed_mask

            self.state_belief_queue.append((depth_new, PO.copy()))
            if len(self.state_belief_queue) > self.belief_window_size:
                self.state_belief_queue.pop(0)
            self.belief_history.append(self.state_belief_queue.copy())
            self.max_obs_grad = max_grad

            state_flattened = np.zeros((BELIEF_WINDOW_SIZE, self.obs_dim_per_step), dtype=np.float64)
            for i in range(len(self.state_belief_queue)):
                depth_val   = self.state_belief_queue[i][0]
                data_vector = self.state_belief_queue[i][1]
                state_flattened[i, :] = np.concatenate([[depth_val], data_vector])
            self.state = state_flattened





        Observe_last_idx=np.where(self.state.sum(axis=1)!=0)[0]

        if len(Observe_last_idx) > 0:
            self.depth_history.append(self.state[Observe_last_idx[-1],0])
        else:
            self.depth_history.append(0.0)
        self.feature_history.append(self.state.copy())

        self.mask_history.append(self.T_observed_mask.copy())
        



        self.episode_length += 1


        if self.episode_length < self.T_all.shape[1]:
            self.T = self.T_all[:, self.episode_length]
            self.S = self.S_all[:, self.episode_length]


        if self.render_mode == "human":
            self._render_frame()

        return self.state, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):

        if self.render_final_only and self.episode_length < self.max_episode_length - 2:
            return
        
        width = self.win_width
        height = self.win_height
        WHITE = self.WHITE

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Temperature Profile Evolution')

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((width, height))
        canvas.fill(WHITE)


        depths = self.d.flatten()
        ep_len = np.arange(self.T_all.shape[1])
        ep_grid, dep_grid = np.meshgrid(ep_len, depths)




        fig, (ax, ax2) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(width / 100, height / 100), dpi=100,
            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.35}
        )
        

        temp_field_gradients = np.zeros_like(self.T_all)
        for ep in range(self.T_all.shape[1]):
            temp_profile = self.T_all[:, ep]



            interp_func = interp1d(depths, temp_profile, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            

            all_depth_segments = []
            all_temp_segments = []
            
            for i in range(len(depths)-1):
                depth_grid_seg = np.linspace(depths[i], depths[i+1], self.sample_window_size)
                temp_seg = interp_func(depth_grid_seg)
                all_depth_segments.append(depth_grid_seg)
                all_temp_segments.append(temp_seg)
            

            fine_depth_grid = np.concatenate(all_depth_segments)
            fine_temp_values = np.concatenate(all_temp_segments)
            

            unique_depths, unique_indices = np.unique(fine_depth_grid, return_index=True)

            unique_temps = fine_temp_values[unique_indices]
            

            temp_gradients_fine = np.abs(np.diff(unique_temps))
            


            if len(temp_gradients_fine) > 0:
                grad_interp = interp1d(unique_depths[:-1], temp_gradients_fine, kind='nearest',
                                      bounds_error=False, fill_value=0)
                temp_field_gradients[:, ep] = grad_interp(depths)
            else:
                temp_field_gradients[:, ep] = 0
        


        gradient_threshold = np.percentile(temp_field_gradients, PLOT_GRD_PERCENT)
        low_gradient_mask = temp_field_gradients < gradient_threshold
        

        c = ax.pcolormesh(ep_grid, dep_grid, self.T_all, cmap='RdBu', shading='auto')
        

        masked_data = np.ma.masked_where(~low_gradient_mask, np.ones_like(self.T_all))
        ax.pcolormesh(ep_grid, dep_grid, masked_data, cmap='Greys', alpha=0.4, shading='auto', vmin=0, vmax=1)
        
        fig.colorbar(c, ax=ax, label='Temperature')


        ax.plot(np.arange(len(self.depth_history)), np.array(self.depth_history), color='orange', marker='o', label='AUV Track')
        # ax.scatter(np.arange(len(self.depth_history)), -np.array(self.depth_history), color='grey', marker='o', label='AUV Position')

        if hasattr(self, 'current_filename'):
            title = f'Temperature Profile Evolution - File: {os.path.basename(self.current_filename)}'
        else:
            title = 'Temperature Profile Evolution'
        
        ax.set_xlabel('Episode Length')
        ax.set_ylabel('Depth')
        ax.set_title(title)

        ax.set_xlim(0, self.max_episode_length - 1)
        ax.set_ylim(self.min_depth, self.max_depth)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        

        if hasattr(self, 'episode_length'):
            ax.axvline(x=self.episode_length, color='red', linestyle='--', alpha=0.7, linewidth=1)  


            


        if hasattr(self, 'belief_history') and len(self.belief_history) > 0:

            max_time_steps = self.max_episode_length
            current_time_steps = len(self.belief_history)
            depth_grid_vis = np.linspace(self.min_depth, self.max_depth, 50)
            

            heatmap_data = np.full((len(depth_grid_vis), max_time_steps), np.nan)
            

            for t, belief_queue in enumerate(self.belief_history):
                if t < max_time_steps and belief_queue:

                    depths_t = []
                    gradient_values_t = []
                    
                    for depth, po_features in belief_queue:

                        if len(po_features) > 1:

                            T_part = po_features[:self.sample_window_size]
                            temp_gradients = np.abs(np.diff(T_part))
                            max_gradient = np.max(temp_gradients) if len(temp_gradients) > 0 else 0.0
                        else:
                            max_gradient = 0.0
                        
                        depths_t.append(depth)
                        gradient_values_t.append(max_gradient)
                    

                    for depth, gradient_value in zip(depths_t, gradient_values_t):

                        depth_idx = np.argmin(np.abs(depth_grid_vis - depth))
                        if 0 <= depth_idx < len(depth_grid_vis):

                            if np.isnan(heatmap_data[depth_idx, t]):
                                heatmap_data[depth_idx, t] = gradient_value
                            else:
                                heatmap_data[depth_idx, t] = max(heatmap_data[depth_idx, t], gradient_value)
            

            time_grid = np.arange(max_time_steps)
            time_mesh, depth_mesh = np.meshgrid(time_grid, depth_grid_vis)
            

            masked_data = np.ma.masked_invalid(heatmap_data)

            grad_vmin = 0
            grad_vmax = np.nanmax(heatmap_data) if not np.all(np.isnan(heatmap_data)) else 1.0
            im = ax2.pcolormesh(time_mesh, depth_mesh, masked_data, 
                               cmap='YlOrRd', shading='auto', alpha=0.8, 
                               vmin=grad_vmin, vmax=grad_vmax)
            

            nan_mask = np.isnan(heatmap_data)
            ax2.pcolormesh(time_mesh, depth_mesh, nan_mask.astype(float), 
                          cmap='Greys', alpha=0, shading='auto', vmin=0, vmax=1)
            

            if not np.all(np.isnan(heatmap_data)):
                cbar = fig.colorbar(im, ax=ax2, label='Temperature Gradient (°C/5m)')
            

            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Depth (m)')
            ax2.set_title('Temperature Gradient Evolution (5m intervals)')
            

            ax2.set_ylim(self.min_depth, self.max_depth)
            ax2.set_xlim(0, max_time_steps - 1)
            

            if hasattr(self, 'episode_length'):
                ax2.axvline(x=self.episode_length, color='red', linestyle='--', alpha=0.7, linewidth=1)


        fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.35)

        fig.canvas.draw()


        if self.episode_length > self.max_episode_length - 1:

            if hasattr(self, 'current_filename') and self.current_filename:
                base_name = os.path.basename(self.current_filename).split('.')[0]
                current_time = time.strftime("%Y%m%d%H%M")
                save_path = f"render_{base_name}_{current_time}.png"
            

            output_dir = "figs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            full_save_path = os.path.join(output_dir, save_path)
            fig.savefig(full_save_path, dpi=150)
            print(f"Final frame saved to {full_save_path}")

        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        img_array = np.transpose(img_array, (1, 0, 2))
        img_surface = pygame.surfarray.make_surface(img_array[:, :, :3].astype(np.uint8))

        canvas.blit(img_surface, (0, 0))
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        plt.close(fig)

    def get_episode_return(self):
        
        return self.episode_return

    def get_VLMdata(self):
        
        max_time_steps = self.max_episode_length
        current_time_steps = len(self.belief_history)

        stage=current_time_steps/max_time_steps

        history = self.state_belief_queue
        max_obs_grad = self.max_obs_grad

        if self.state_belief_queue:
            depths = [item[0] for item in self.state_belief_queue]
            coverage = (min(depths), max(depths))
        else:
            coverage = (0, 0)

        
        return history, max_obs_grad, coverage,stage


    def _compute_thermo_depth_seq(self):
        
        d_flat = self.d.flatten()
        z_seq = []
        for t in range(self.T_all.shape[1]):
            T_t = self.T_all[:, t]
            grads = np.abs(np.gradient(T_t, d_flat))
            z_seq.append(float(d_flat[np.argmax(grads)]))
        return z_seq

    def get_true_thermo_depth_seq(self):
        
        current_z = self.true_thermo_depth_seq[min(self.episode_length, len(self.true_thermo_depth_seq) - 1)]
        return current_z, self.true_thermo_depth_seq, self.episode_start_datetime

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def reward(self, action):
        # a=0, down  a=1, up
        Observe_last_idx = np.where(self.state.sum(axis=1) != 0)[0]
        depth = self.state[Observe_last_idx[-1], 0]

        next_depth = depth + self.v * (action - 0.5) * 2
        next_depth_clipped = np.clip(next_depth, self.d.min(), self.d.max())

        if next_depth_clipped == depth:
            return -20.0

        if Observe_last_idx.shape[0] > 1:
            depth_prev = self.state[Observe_last_idx[-2], 0]
            if abs(depth - depth_prev) < self.v:
                return -20.0


        import math as _math
        if not _math.isnan(self.llm_depth):
            dist = abs(next_depth_clipped - self.llm_depth)
            return R_LLM_DENSE_MAX * _math.exp(-dist / R_LLM_SIGMA)


        return 1.0

    def get_max_grad_depth(self) -> float | None:
        
        if len(self.d_observed) < 4:
            return None
        d = np.array(self.d_observed)
        T = np.array(self.T_observed)
        S = np.array(self.S_observed)


        _THERMO_D_MIN, _THERMO_D_MAX = -150.0, -25.0
        mask = (d >= _THERMO_D_MIN) & (d <= _THERMO_D_MAX)
        if mask.sum() < 4:
            return None
        d, T, S = d[mask], T[mask], S[mask]


        order = np.argsort(d)
        d, T, S = d[order], T[order], S[order]
        unique_d, idx = np.unique(d, return_index=True)
        if len(unique_d) < 4:
            return None

        unique_T = np.array([T[d == ud].mean() for ud in unique_d])
        unique_S = np.array([S[d == ud].mean() for ud in unique_d])


        d_min, d_max = unique_d[0], unique_d[-1]
        if d_max - d_min < self.v * 2:
            return None
        n_pts = max(int((d_max - d_min) / self.v) + 1, 4)
        d_grid = np.linspace(d_min, d_max, n_pts)
        from scipy.interpolate import interp1d
        T_interp = interp1d(unique_d, unique_T, kind='linear', bounds_error=False, fill_value='extrapolate')(d_grid)
        S_interp = interp1d(unique_d, unique_S, kind='linear', bounds_error=False, fill_value='extrapolate')(d_grid)

        grad_T = np.abs(np.gradient(T_interp, d_grid))
        grad_S = np.abs(np.gradient(S_interp, d_grid))
        combined = grad_T + 0.5 * grad_S
        return float(d_grid[np.argmax(combined)])
    

    def return_Tgradient(self, d):
        
        depth_array=self.d
        temp_array=self.T

        depth_array=depth_array.reshape(-1)
        temp_array=temp_array.reshape(-1)
    
        if depth_array[0]>depth_array[-1]:
            depth_array=-depth_array
            temp_array=-temp_array

        if d < depth_array[0] or d > depth_array[-1]:
            print(d)
            print("深度范围:", depth_array[0],depth_array[-1])
            raise ValueError("Depth d is out of the range of the provided depths")


        idx_below = np.searchsorted(depth_array, d) - 1
        idx_above = idx_below + 1


        depth_below = depth_array[idx_below]
        temp_below = temp_array[idx_below]
        depth_above = depth_array[idx_above]
        temp_above = temp_array[idx_above]


        depth_diff = depth_above - depth_below
        if abs(depth_diff) < 1e-10:
            return 0.0
    
        gradient = (temp_above - temp_below) / depth_diff
        return gradient

    def POcalc(self, depth0, depth_last=None):
        


        T_observed_mask = self.T_observed_mask.copy()
        

        d_flat = self.d.reshape(-1)
        T_flat = self.T.reshape(-1)
        S_flat = self.S.reshape(-1)
        if d_flat[0] > d_flat[-1]:
            order = np.argsort(d_flat)
            d_flat = d_flat[order]
            T_flat = T_flat[order]
            S_flat = S_flat[order]

        d_min = float(d_flat[0])
        d_max = float(d_flat[-1])


        if depth_last is None:
            if len(self.d_observed) > 0:
                depth_start = self.d_observed[-1]
            else:
                depth_start = depth0
        else:
            depth_start = depth_last


        num_samples = max(2, int(abs(depth0 - depth_start) / 5) + 1)
        if depth_start == depth0:
            sample_depths = [depth0]
        else:
            sample_depths = np.linspace(depth_start, depth0, num_samples)


        interp_T = interp1d(d_flat, T_flat, kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_S = interp1d(d_flat, S_flat, kind='linear', bounds_error=False, fill_value='extrapolate')
        sample_temperatures = interp_T(sample_depths)
        sample_salinities   = interp_S(sample_depths)


        for i, (depth, temp, sal) in enumerate(zip(sample_depths, sample_temperatures, sample_salinities)):
            if i == 0 and len(self.d_observed) > 0:
                continue
            self.d_observed.append(depth)
            self.T_observed.append(temp)
            self.S_observed.append(sal)


        bin_edges = np.linspace(d_max, d_min, self.grid_num + 1)
        for depth in sample_depths:
            bin_idx = np.digitize(depth, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, self.grid_num - 1)
            T_observed_mask[bin_idx + 1] = True




        w = self.sample_window_size
        if len(self.T_observed) >= 2:
            d_obs = np.array(self.d_observed)
            T_obs = np.array(self.T_observed)
            S_obs = np.array(self.S_observed)
            d_lo, d_hi = d_obs.min(), d_obs.max()
            if d_hi - d_lo < self.v:
                PO_T = np.full(w, T_obs[-1])
                PO_S = np.full(w, S_obs[-1])
            else:
                bin_edges = np.linspace(d_lo, d_hi, w + 1)
                PO_T = np.full(w, T_obs[-1])
                PO_S = np.full(w, S_obs[-1])
                for i in range(w):
                    mask = np.where((d_obs >= bin_edges[i]) & (d_obs < bin_edges[i + 1]))[0]
                    if len(mask) > 0:
                        latest = mask[-1]
                        PO_T[i] = T_obs[latest]
                        PO_S[i] = S_obs[latest]
        else:
            PO_T = np.full(w, self.T_observed[-1] if self.T_observed else 0.0)
            PO_S = np.full(w, self.S_observed[-1] if self.S_observed else 0.0)
        PO = np.concatenate([PO_T, PO_S])   # (30,)


        max_grad = 0.0
        max_grad_depth = depth0
        if len(sample_depths) > 1:
            temp_gradients = np.gradient(sample_temperatures, sample_depths)
            max_grad_idx = np.argmax(np.abs(temp_gradients))
            max_grad = abs(temp_gradients[max_grad_idx])
            max_grad_depth = sample_depths[max_grad_idx]



        new_belief = {}
        
        return PO, T_observed_mask, max_grad_depth, max_grad, new_belief

    def interp_data(self,sampled_T_interp):
        

        max_episode_length = self.max_episode_length
        num_depths = sampled_T_interp.shape[1]
        interpolated_T = np.zeros((num_depths, max_episode_length))


        # new_time_grid = np.linspace(sampled_time_grid_datetime[0], sampled_time_grid_datetime[-1], max_episode_length)

        old_time_grid = np.linspace(0, sampled_T_interp.shape[0] - 1,  sampled_T_interp.shape[0])
        new_time_grid = np.linspace(0, sampled_T_interp.shape[0] - 1, max_episode_length)

        for i in range(num_depths):

            interp_func = interp1d(old_time_grid, sampled_T_interp[:, i], kind='linear', fill_value="extrapolate")


            interpolated_T[i, :] = interp_func(new_time_grid)

        return interpolated_T

# # # check
# env = ThermoTrackEnv(render_mode="human")
# seed=1111
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)




# for i in range(2):
#     # print(f"Step {i+1}")
#     state, info = env.reset()
#     for j in range(5):
#         action = env.action_space.sample()
#         print(f"Action taken: {action}")
#         state, reward, terminated, truncated, info = env.step(action)
#         if terminated or truncated:
#             break
#     state, info = env.reset()
#     for k in range(5):
#         action = env.action_space.sample()
#         print(f"Action taken: {action}")
#         state, reward, terminated, truncated, info = env.step(action)
#         if terminated or truncated:
#             break
#     state, info = env.reset()
#     for l in range(5):
#         action = env.action_space.sample()
#         print(f"Action taken: {action}")
#         state, reward, terminated, truncated, info = env.step(action)
#         if terminated or truncated:
#             break

#     # if np.isnan(state).any():
#     #     print(f"Warning: NaN found in initial state: {state}")
# # # # # It will check your custom environment and output additional warnings if needed
# # check_env(env)
# env = ThermoTrackEnv(render_mode="human")
# state, info = env.reset()
# for i in range(2):
#     action = env.action_space.sample()
#     state, reward, terminated, truncated, info = env.step(action)
#     history, max_obs_grad, coverage,stage=env.get_VLMdata()
#     # print(f"history: {history}")


gym.register(
    id=ENV_ID,
    entry_point=ThermoTrackEnv,
)
