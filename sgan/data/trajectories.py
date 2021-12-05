import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset
from sgan.utils import debug

logger = logging.getLogger(__name__)


def seq_collate(data):
    """
    Input data:
        obs_seq_list: list of arrays, len = bs, dim = (pedestrians, 2, obs_len)
        pred_seq_list: list of arrays, len = bs, dim = (pedestrians, 2, pred_len)
        obs_seq_rel_list: list of arrays, len = bs, dim = (pedestrians, 2, obs_len)
        pred_seq_rel_list: list of arrays, len = bs, dim = (pedestrians, 2, pred_len)
        non_linear_ped_list: list of arrays, len = bs, dim = (pedestrians,)
        loss_mask_list: list of arrays, len = bs, dim = (pedestrians, seq_len)
        ped_key_list: list of arrays, len = bs, dim = (pedestrians,)

    Output:
        obs_traj: (obs_len, pedestrains in batch, 2)
        pred_traj: (pred_len, pedestrains in batch, 2)
        obs_traj_rel: (obs_len, pedestrains in batch, 2)
        pred_traj_rel: (pred_len, pedestrains in batch, 2)
        non_linear_ped: (pedestrains in batch,)
        loss_mask: (pedestrains in batch, seq_len)
        seq_start_end: (bs, 2)
        ped_keys: (pedestrains in batch,)
    """
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, ped_key_list) = zip(*data)

    # debug('Before collate:\n'
    #       f'obs_seq_list shape {len(obs_seq_list)}, {obs_seq_list[0].shape}\n')

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    ped_keys = np.concatenate(ped_key_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, ped_keys
    ]

    # debug('Processed data batch:\n'
    #       f'obs_traj shape {out[0].shape}\n'
    #       f'pred_traj shape {out[1].shape}\n'
    #       f'obs_traj_rel shape {out[2].shape}\n'
    #       f'pred_traj_rel shape {out[3].shape}\n'
    #       f'non_linear_ped shape {out[4].shape}\n'
    #       f'loss_mask shape {out[5].shape}\n'
    #       f'seq_start_end shape {out[6].shape}\n'
    #       f'ped_keys shapes {out[7].shape}')

    # debug(f'ped keys {out[7]}')
    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        ped_key_list = []
        for path in all_files:
            data = read_file(path, delim)
            '''
            frames: time frames
            list
            '''
            frames = np.unique(data[:, 0]).tolist()
            '''
            frame_data: scene data index by time frames
            list of list of list. 
            dim 0: time frame
            dim 1: agent
            dim 2: data fields
            '''
            np.set_printoptions(precision=2)
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])

            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                '''
                idx: the start frame of a scene clip of duration seq_len
                '''
                '''
                curr_seq_data: scene data of later seq_len time frames starting from idx
                list of list
                dim0: collection of agents in all seq_len frames
                dim1: data fields
                '''
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                '''
                peds_in_curr_seq: ped id included in the clipped time frames
                list
                '''
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                ped_keys = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    ''' curr_ped_seq: sequence of the current pedestrian
                    list of list
                    dim0: time frames
                    dom1: data fields
                    '''
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    # ped_keys.append('tf_' + str(int(curr_ped_seq[0, 0])) + '_a_' + str(int(curr_ped_seq[0, 1])))
                    ped_keys.append(str(int(curr_ped_seq[0, 1])))
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    '''
                    pad_front: relative start frame of the pedestrian's traj w.r.t the clip
                    pad_end: relative end frame of the pedestrian's traj w.r.t the clip
                    '''
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    ''' transformed curr_ped_seq: trajecory of the current pedestrian
                    dim = (2, time_frames)
                    '''
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    '''
                    rel_curr_ped_seq: displacements across time steps
                    dim = (2, time_frames)
                    '''
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    '''
                    _idx: a contiguous list of ids for considered pedestrains
                    '''
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    '''
                    _non_linear_ped: classification result of whether the trajectories are linear
                    list of int
                    len = num_peds_considered
                    '''
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    '''
                    non_linear_ped (list of int): linearity of all considered trajectories from the data set 
                    num_peds_in_seq (list of int, len = num_clips): number of peds considered in each clip
                    seq_list (list of nparray, len = num_clips, array dim (num_peds_considered, 2, seq_len))
                    '''
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    ped_key_list.append(ped_keys[:num_peds_considered])

        '''
        non_linear_ped (list of int, len = all_trajs_considered): linearity of all considered trajectories from the data set 
        seq_list and seq_list_rel (list of nparray, dim = (all_trajs_considered, 2, seq_len))
        '''
        self.num_seq = len(seq_list)
        ped_key_list = np.concatenate(ped_key_list, axis=0)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.ped_keys = ped_key_list
        '''
        cum_start_idx: start index of trajectories for each scene clip
        seq_start_end: start and end indices of trajectories for each scene clip
        '''
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        """
        get trajectories from a scene clip
        obs_traj:  dim = (pedestrians, 2, obs_len)
        pred_traj:  dim = (pedestrians, 2, pred_len)
        obs_traj_rel:  dim = (pedestrians, 2, obs_len)
        pred_traj_rel:  dim = (pedestrians, 2, pred_len)
        non_linear_ped: dim = (pedestrians,)
        loss_mask: dim = (pedestrians, seq_len)
        ped_ids: dim = (pedestrians,)
        """
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.ped_keys[start:end]
        ]

        # debug('Raw data batch:\n'
        #       f'obs_traj shape {out[0].shape}\n'
        #       f'pred_traj shape {out[1].shape}\n'
        #       f'obs_traj_rel shape {out[2].shape}\n'
        #       f'pred_traj_rel shape {out[3].shape}\n'
        #       f'non_linear_ped shape {out[4].shape}\n'
        #       f'loss_mask shape {out[5].shape}\n'
        #       f'ped_ids shape {out[6].shape}')

        return out
