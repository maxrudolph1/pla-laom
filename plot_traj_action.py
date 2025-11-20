import h5py
import numpy as np
import matplotlib.pyplot as plt


def hdf5_to_dict(hdf5_file):
    """
    Recursively convert an HDF5 file/group to a dictionary.
    
    Args:
        hdf5_file: h5py File or Group object
        
    Returns:
        Dictionary containing all datasets and groups from the HDF5 file
    """
    result = {}
    
    for key in hdf5_file.keys():
        item = hdf5_file[key]
        
        if isinstance(item, h5py.Group):
            # Recursively convert groups
            result[key] = hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            # Convert datasets to numpy arrays
            result[key] = item[()]
            
    return result


# # Open HDF5 file and convert to dictionary
# data_path = "/home1/09312/rudolph/documents/visual_dm_control/data/pm_clean/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/all_episodes/episode_0.hdf5"  # Replace with your actual path

# with h5py.File(data_path, 'r') as f:
#     data_dict = hdf5_to_dict(f)

# pos = data_dict['state'][:, :2]
# vel = data_dict['state'][:, 2:]
# act = data_dict['action']

# plt.subplot(3, 1, 1)
# plt.plot(pos[:, 0], pos[:, 1], 'r-')
# plt.title('Position')
# plt.subplot(3, 1, 2)
# plt.plot(vel[:, 0], vel[:, 1], 'b-')
# plt.title('Velocity')
# plt.subplot(3, 1, 3)
# plt.plot(act[:, 0], act[:, 1], 'g-')
# plt.title('Action')
# plt.savefig('traj_action.png')


# print("Keys in the dictionary:", list(data_dict.keys()))

# data_path = "/home1/09312/rudolph/documents/visual_dm_control/data/pm_clean/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/500_episodes/500_episodes.hdf5"
# data_path = '/home1/09312/rudolph/documents/visual_dm_control/data/test/policy_rollouts/undistracted/cheetah/run_forward/sac/expert/easy/84x84/action_repeats_2/train/500_episodes/500_episodes.hdf5'
data_path = 'data/pm_tanh_actions/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/64_episodes/64_episodes.hdf5'
with h5py.File(data_path, 'r') as f:
    data_dict = hdf5_to_dict(f)

print(data_dict.keys())

act = data_dict['action'].squeeze()
act -= act.mean(axis=0)
act /= act.std(axis=0)
# Get number of dimensions
n_dims = act.shape[1]

# Create subplots
fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3*n_dims))

# Handle case where n_dims = 1 (axes won't be an array)
if n_dims == 1:
    axes = [axes]

# Plot histogram for each dimension
for i in range(n_dims):
    axes[i].hist(act[:, i], bins=50, alpha=0.7, edgecolor='black')
    axes[i].set_xlabel(f'Action Dimension {i}')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Histogram of Action Dimension {i}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('action_histograms_pm.png', dpi=150)
plt.show()
