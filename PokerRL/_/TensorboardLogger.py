# Copyright (c) 2019 Eric Steinberger

from os.path import join as ospj

from torch.utils.tensorboard import SummaryWriter

from PokerRL.rl.MaybeRay import MaybeRay
from PokerRL.util.file_util import create_dir_if_not_exist, write_dict_to_file_json


class TensorboardLogger:
    """Log training metrics to TensorBoard using :class:`SummaryWriter`.

    ``TensorboardLogger`` pulls log data from the Chief worker and writes it to
    TensorBoard event files. Optionally, logs can also be exported as JSON for
    later analysis.
    """

    def __init__(self, name, runs_distributed, runs_cluster, chief_handle,
                 path_log_storage=None):
        self._name = name
        self._path_log_storage = path_log_storage
        if path_log_storage is not None:
            create_dir_if_not_exist(path_log_storage)

        self._chief_handle = chief_handle
        self._writers = {}
        self._custom_logs = {}

        self._ray = MaybeRay(runs_distributed=runs_distributed,
                             runs_cluster=runs_cluster)

    @property
    def name(self):
        return self._name

    @property
    def path_log_storage(self):
        return self._path_log_storage

    def clear(self):
        self._writers = {}

    def export_all(self, iter_nr):
        """Flush TensorBoard writers and dump custom logs to disk."""
        if self._path_log_storage is not None:
            path_tb = ospj(self._path_log_storage, str(self._name), str(iter_nr),
                           "tensorboard")
            path_json = ospj(self._path_log_storage, str(self._name), str(iter_nr),
                             "as_json")
            create_dir_if_not_exist(path=path_tb)
            create_dir_if_not_exist(path=path_json)
            for writer in self._writers.values():
                writer.flush()
            write_dict_to_file_json(dictionary=self._custom_logs, _dir=path_json,
                                    file_name="logs")

    def update_from_log_buffer(self):
        """Pull new logs from the Chief and add them to TensorBoard."""
        new_v, exp_names = self._get_new_vals()

        for e in exp_names:
            if e not in self._writers:
                self._custom_logs[e] = {}
                log_dir = None
                if self._path_log_storage is not None:
                    log_dir = ospj(self._path_log_storage, e)
                    create_dir_if_not_exist(log_dir)
                self._writers[e] = SummaryWriter(log_dir=log_dir)

        for name, vals_dict in new_v.items():
            for graph_name, data_points in vals_dict.items():
                for data_point in data_points:
                    step = int(data_point[0])
                    val = data_point[1]
                    self._writers[name].add_scalar(graph_name, val, step)
                    if graph_name not in self._custom_logs[name]:
                        self._custom_logs[name][graph_name] = []
                    self._custom_logs[name][graph_name].append({step: val})

    def _get_new_vals(self):
        """Returns the latest logs from the Chief."""
        return self._ray.get(self._ray.remote(self._chief_handle.get_new_values))
