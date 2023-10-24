from pyomo.common.timing import HierarchicalTimer, _HierarchicalHelper
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from mpi4py import MPI


comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()


class TimerCollection():

    def __init__(self, timer_list: List[HierarchicalTimer]):
        self._timer_list = timer_list
        self._num_timers: int = len(timer_list)

    def get_total_time(self, identifier) -> Tuple[float, float]:
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        total_time: float
            The total time spent with the specified timer active.
        """
        total_times = np.array([timer.get_total_time(identifier) for timer in self._timer_list])
        return np.mean(total_times), np.std(total_times)
    
    def get_num_calls(self, identifier) -> Tuple[float, float]:
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        n_calls: int
            The number of times start was called for the specified timer.
        """
        num_calls = np.array([timer.get_num_calls(identifier) for timer in self._timer_list])
        return np.mean(num_calls), np.std(num_calls)
    
    def get_relative_percent_time(self, identifier) -> Tuple[float, float]:
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        percent_time: float
            The percent of time spent in the specified timer
            relative to the timer's immediate parent.
        """
        rel_times = np.array([timer.get_relative_percent_time(identifier) for timer in self._timer_list])
        return np.mean(rel_times), np.std(rel_times)
    
    def get_percall_time(self, identifier) -> Tuple[float, float]:
        total_times = np.array([timer.get_total_time(identifier) for timer in self._timer_list])
        num_calls = np.array([timer.get_num_calls(identifier) for timer in self._timer_list])
        percall_times = total_times / num_calls
        return np.mean(percall_times), np.std(percall_times)
    
    def get_total_percent_time(self, identifier) -> Tuple[float, float]:
        """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        percent_time: float
            The percent of time spent in the specified timer
            relative to the total time in all timers.
        """
        total_percent_times = np.array([timer.get_total_percent_time(identifier) for timer in self._timer_list])
        return np.mean(total_percent_times), np.std(total_percent_times)

    def __str__(self):
        raise NotImplementedError("Not implemented, use MPIHierarchicalTimer instead")
    


def reduce_mean(x):
    return comm.allreduce(x, op=MPI.SUM) / comm.size

def reduce_var(x):
    mean = comm.allreduce(x, op=MPI.SUM) / comm.size
    return np.sqrt(comm.allreduce((x - mean)**2, op=MPI.SUM) / comm.size)


class _MPIHierarchicalHelper(_HierarchicalHelper):

    def __init__(self):
        #self._comm: MPI.Comm = comm
        super().__init__()


    def to_str(self, indent, stage_identifier_lengths, reduce: Callable):
        s = ''
        if len(self.timers) > 0:
            underline = indent + '-' * (sum(stage_identifier_lengths) + 36) + '\n'
            s += underline
            name_formatter = '{name:<' + str(sum(stage_identifier_lengths)) + '}'
            other_time = self.total_time
            sub_stage_identifier_lengths = stage_identifier_lengths[1:]
            for name, timer in sorted(self.timers.items()):
                if self.total_time > 0:
                    _percent = timer.total_time / self.total_time * 100
                else:
                    _percent = float('nan')
                s += indent
                s += (
                    name_formatter + '{ncalls:>9d} {cumtime:>9.3f} '
                    '{percall:>9.3f} {percent:>6.1f}\n'
                ).format(
                    name=name,
                    ncalls=int(reduce(timer.n_calls)),
                    cumtime=reduce(timer.total_time),
                    percall=reduce(timer.total_time / timer.n_calls),
                    percent=reduce(_percent),
                )
                s += timer.to_str(
                    indent=indent + ' ' * stage_identifier_lengths[0],
                    stage_identifier_lengths=sub_stage_identifier_lengths,
                    reduce=reduce,
                )
                other_time -= timer.total_time

            if self.total_time > 0:
                _percent = other_time / self.total_time * 100
            else:
                _percent = float('nan')
            s += indent
            s += (
                name_formatter + '{ncalls:>9} {cumtime:>9.3f} '
                '{percall:>9} {percent:>6.1f}\n'
            ).format(
                name='other',
                ncalls='n/a',
                cumtime=reduce(other_time),
                percall='n/a',
                percent=reduce(_percent),
            )
            s += underline.replace('-', '=')
        return s

class MPIHierarchicalTimer(HierarchicalTimer):

    def __init__(self):
        #self._comm: MPI.Comm = comm
        super().__init__()

    def _get_timer(self, identifier, should_exist=False):
        """
        This method gets the timer associated with the current state
        of self.stack and the specified identifier.

        Parameters
        ----------
        identifier: str
            The name of the timer
        should_exist: bool
            The should_exist is True, and the timer does not already
            exist, an error will be raised. If should_exist is False, and
            the timer does not already exist, a new timer will be made.

        Returns
        -------
        timer: _HierarchicalHelper

        """
        parent = self._get_timer_from_stack(self.stack)
        if identifier in parent.timers:
            return parent.timers[identifier]
        else:
            if should_exist:
                raise RuntimeError(
                    'Could not find timer {0}'.format(
                        '.'.join(self.stack + [identifier])
                    )
                )
            parent.timers[identifier] = _MPIHierarchicalHelper()
            return parent.timers[identifier]

    def __str__(self, reduction: Optional[str] = 'none'):
        if reduction == 'mean':
            reduce = lambda x: reduce_mean(x)
        elif reduction == 'variance':
            reduce = lambda x: reduce_var(x)
        elif reduction == 'none':
            reduce = lambda x: x
        else:
            raise NotImplementedError(f"Reduction {reduction} not implemented")
        const_indent = 4
        max_name_length = 200 - 36
        stage_identifier_lengths = self._get_identifier_len()
        name_field_width = sum(stage_identifier_lengths)
        if name_field_width > max_name_length:
            # switch to a constant indentation of const_indent spaces
            # (to hopefully shorten the line lengths
            name_field_width = max(
                const_indent * i + l for i, l in enumerate(stage_identifier_lengths)
            )
            for i in range(len(stage_identifier_lengths) - 1):
                stage_identifier_lengths[i] = const_indent
            stage_identifier_lengths[-1] = name_field_width - const_indent * (
                len(stage_identifier_lengths) - 1
            )
        name_formatter = '{name:<' + str(name_field_width) + '}'
        s = (
            name_formatter + '{ncalls:>9} {cumtime:>9} {percall:>9} {percent:>6}\n'
        ).format(
            name='Identifier',
            ncalls='ncalls',
            cumtime='cumtime',
            percall='percall',
            percent='%',
        )
        underline = '-' * (name_field_width + 36) + '\n'
        s += underline
        sub_stage_identifier_lengths = stage_identifier_lengths[1:]
        for name, timer in sorted(self.timers.items()):
            s += (
                name_formatter + '{ncalls:>9d} {cumtime:>9.3f} '
                '{percall:>9.3f} {percent:>6.1f}\n'
            ).format(
                name=name,
                ncalls=int(reduce(timer.n_calls)),
                cumtime=reduce(timer.total_time),
                percall=reduce(timer.total_time / timer.n_calls),
                percent=reduce(self.get_total_percent_time(name)),
            )
            s += timer.to_str(
                indent=' ' * stage_identifier_lengths[0],
                stage_identifier_lengths=sub_stage_identifier_lengths,
                reduce=reduce,
            )
        s += underline.replace('-', '=')
        return s