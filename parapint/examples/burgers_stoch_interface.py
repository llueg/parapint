import pyomo.environ as pe
from pyomo import dae
import parapint
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from mpi4py import MPI
import math
import logging
import argparse
import pickle
from typing import List, Tuple, Dict


"""
Run this example with, e.g., 

mpirun -np 4 python -m mpi4py burgers_stoch_interface.py --nfe_x 50 --nfe_t 200 --nblocks 4

If you run it with the --plot, make sure you don't use too many finite elements, or it will take forever to plot.
"""

comm: MPI.Comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    logging.basicConfig(level=logging.INFO)


class Args(object):
    def __init__(self):
        self.nfe_x = 50
        self.nfe_t = 200
        self.nblocks = 4
        self.plot = True
        self.show_plot = True

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--nfe_x', type=int, required=False, default=50, help='number of finite elements for x')
        parser.add_argument('--nfe_t', type=int, required=False, default=200, help='number of finite elements for t')
        parser.add_argument('--nblocks', type=int, required=False, default=4, help='number of time blocks for schur complement')
        parser.add_argument('--no_plot', action='store_true')
        parser.add_argument('--no_show_plot', action='store_true')
        args = parser.parse_args()
        self.nfe_x = args.nfe_x
        self.nfe_t = args.nfe_t
        self.nblocks = args.nblocks
        self.plot = not args.no_plot
        self.show_plot = not args.no_show_plot


class BurgersInterface(parapint.interfaces.MPIStochasticSchurComplementInteriorPointInterface):
    def __init__(self, start_t, end_t, num_time_blocks, nfe_t, nfe_x):
        self.nfe_x = nfe_x
        self.dt = (end_t - start_t) / float(nfe_t)
        self.last_t = None
        self.scenarios = list(range(num_time_blocks))
        self.delta_t = (end_t - start_t) / num_time_blocks
        self._start_t = start_t
        self._end_t = end_t
        self._num_time_blocks = num_time_blocks
        x_indices = np.round(np.linspace(0, 1, nfe_x + 1), 3)[1:-1].tolist()

        first_stage_var_ids = [('y', (x, i*self.delta_t)) for i in range(1, num_time_blocks)for x in x_indices] + \
                                [('u', (x, i*self.delta_t)) for i in range(1, num_time_blocks) for x in x_indices]
        super(BurgersInterface, self).__init__(scenarios=self.scenarios,
                                      nonanticipative_var_identifiers=first_stage_var_ids,
                                      comm=comm)
        #print(f'num first stage vars: {self._num_first_stage_vars}')


    def build_burgers_model(self, nfe_x=50, nfe_t=50, start_t=0, end_t=1, add_init_conditions=True):
        dt = (end_t - start_t) / float(nfe_t)

        start_x = 0
        end_x = 1
        dx = (end_x - start_x) / float(nfe_x)

        m = pe.Block(concrete=True)
        m.omega = pe.Param(initialize=1)
        m.v = pe.Param(initialize=0.02)
        m.r = pe.Param(initialize=0)

        m.x = dae.ContinuousSet(bounds=(start_x, end_x))
        m.t = dae.ContinuousSet(bounds=(start_t, end_t))

        m.y = pe.Var(m.x, m.t)
        m.dydt = dae.DerivativeVar(m.y, wrt=m.t)
        m.dydx = dae.DerivativeVar(m.y, wrt=m.x)
        m.dydx2 = dae.DerivativeVar(m.y, wrt=(m.x, m.x))

        m.u = pe.Var(m.x, m.t)

        def _y_init_rule(m, x):
            # return np.cos(2 * np.pi * x)
            # if x <= 0.5 * end_x:
            #     return 1
            # return 0
            return np.sin(4 * np.pi * x)
            

        m.y0 = pe.Param(m.x, default=_y_init_rule)

        def _upper_x_bound(m, t):
            return m.y[end_x, t] == 0

        m.upper_x_bound = pe.Constraint(m.t, rule=_upper_x_bound)

        def _lower_x_bound(m, t):
            return m.y[start_x, t] == 0

        m.lower_x_bound = pe.Constraint(m.t, rule=_lower_x_bound)

        def _upper_x_ubound(m, t):
            return m.u[end_x, t] == 0

        m.upper_x_ubound = pe.Constraint(m.t, rule=_upper_x_ubound)

        def _lower_x_ubound(m, t):
            return m.u[start_x, t] == 0

        m.lower_x_ubound = pe.Constraint(m.t, rule=_lower_x_ubound)

        def _lower_t_bound(m, x):
            if x == start_x or x == end_x:
                return pe.Constraint.Skip
            return m.y[x, start_t] == m.y0[x]

        def _lower_t_ubound(m, x):
            if x == start_x or x == end_x:
                return pe.Constraint.Skip
            return m.u[x, start_t] == 0

        if add_init_conditions:
            m.lower_t_bound = pe.Constraint(m.x, rule=_lower_t_bound)
            m.lower_t_ubound = pe.Constraint(m.x, rule=_lower_t_ubound)

        # PDE
        def _pde(m, x, t):
            if t == start_t or x == end_x or x == start_x:
                e = pe.Constraint.Skip
            else:
                # print(foo.last_t, t-dt, abs(foo.last_t - (t-dt)))
                # assert math.isclose(foo.last_t, t - dt, abs_tol=1e-6)
                e = m.dydt[x, t] - m.v * m.dydx2[x, t] + m.dydx[x, t] * m.y[x, t] == m.r + m.u[x, self.last_t]
            self.last_t = t
            return e

        m.pde = pe.Constraint(m.x, m.t, rule=_pde)

        # Discretize Model
        disc = pe.TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=nfe_t, wrt=m.t, scheme='BACKWARD')
        disc.apply_to(m, nfe=nfe_x, wrt=m.x, scheme='CENTRAL')

        # Solve control problem using Pyomo.DAE Integrals
        def _intX(m, x, t):
            return (m.y[x, t] - m.y0[x]) ** 2 + m.omega * m.u[x, t] ** 2

        m.intX = dae.Integral(m.x, m.t, wrt=m.x, rule=_intX)

        def _intT(m, t):
            return m.intX[t]

        m.intT = dae.Integral(m.t, wrt=m.t, rule=_intT)

        def _obj(m):
            e = 0.5 * m.intT
            for x in sorted(m.x):
                if x == start_x or x == end_x:
                    pass
                else:
                    e += 0.5 * 0.5 * dx * dt * m.omega * m.u[x, start_t] ** 2
            return e

        m.obj = pe.Objective(rule=_obj)

        return m

    def build_model_for_scenario(self, scenario_identifier: int):
        start_t = scenario_identifier * self.delta_t
        end_t = (scenario_identifier + 1) * self.delta_t
        if scenario_identifier == 0:
            add_init_conditions = True
        else:
            add_init_conditions = False
        dt = self.dt
        nfe_t = math.ceil((end_t - start_t) / dt)
        m = self.build_burgers_model(nfe_x=self.nfe_x, nfe_t=nfe_t, start_t=start_t, end_t=end_t,
                                     add_init_conditions=add_init_conditions)

        end_var = {('y', (x, end_t)): m.y[x, end_t] for x in sorted(m.x) if x not in {0, 1}} | \
                    {('u', (x, end_t)): m.u[x, end_t] for x in sorted(m.x) if x not in {0, 1}}
        start_var = {('y', (x, start_t)): m.y[x, start_t] for x in sorted(m.x) if x not in {0, 1}} | \
                    {('u', (x, start_t)): m.u[x, start_t] for x in sorted(m.x) if x not in {0, 1}}
        
        
        if scenario_identifier == 0:
            return m, end_var
        elif scenario_identifier == self._num_scenarios - 1:
            return m, start_var
        else:
            return m, start_var | end_var

    def plot_results(self, show_plot=True):
        y_pts = list()
        u_pts = list()
        for block_ndx in self.local_block_indices:
            m = self.pyomo_model(block_ndx)
            for x in m.x:
                for t in m.t:
                    y_pts.append((x, t, m.y[x, t].value))
                    u_pts.append((x, t, m.u[x, t].value))
        y_pts = comm.allgather(y_pts)
        u_pts = comm.allgather(u_pts)
        if rank == 0:
            _tmp_y = list()
            _tmp_u = list()
            for i in y_pts:
                _tmp_y.extend(i)
            for i in u_pts:
                _tmp_u.extend(i)
            y_pts = _tmp_y
            u_pts = _tmp_u
            y_pts.sort(key=lambda x: x[0])
            y_pts.sort(key=lambda x: x[1])
            u_pts.sort(key=lambda x: x[0])
            u_pts.sort(key=lambda x: x[1])
            x_set = set()
            t_set = set()
            y_dict = dict()
            u_dict = dict()
            for x, t, y in y_pts:
                x_set.add(x)
                t_set.add(t)
                y_dict[x, t] = y
            for x, t, u in u_pts:
                u_dict[x, t] = u
            x_list = list(x_set)
            t_list = list(t_set)
            x_list.sort()
            t_list.sort()
            y_list = list()
            u_list = list()
            all_x = list()
            all_t = list()
            for x in x_list:
                tmp_y = list()
                tmp_u = list()
                tmp_x = list()
                tmp_t = list()
                for t in t_list:
                    tmp_y.append(y_dict[x, t])
                    tmp_u.append(u_dict[x, t])
                    tmp_x.append(x)
                    tmp_t.append(t)
                y_list.append(tmp_y)
                u_list.append(tmp_u)
                all_x.append(tmp_x)
                all_t.append(tmp_t)

            # colors = cm.jet(y_list)
            # rcount, ccount, _ = colors.shape
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # surf = ax.plot_surface(np.array(all_x), np.array(all_t), np.array(y_list), rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
            # surf.set_facecolor((0, 0, 0, 0))
            # ax.set_xlabel('x')
            # ax.set_ylabel('t')
            # ax.set_zlabel('y')
            # if show_plot:
            #     plt.show()
            # plt.close()

            #plot surface from above, no 3d
            # set figsize relative to nfe_t and nfe_x
            # set axis font size
            
            font_size = 14
            n_t, n_x = np.array(all_x).T.shape
            t_to_x_ratio = n_t/n_x
            base_length = 5
            f, a = plt.subplots(figsize=(base_length * t_to_x_ratio, base_length))
            a.tick_params(axis='both', which='major', labelsize=font_size)
            surf = a.pcolormesh(np.array(all_t), np.array(all_x), np.array(y_list), cmap='jet')
            f.colorbar(surf, ax=a)
            a.set_ylabel('x', fontsize=font_size)
            a.set_xlabel('t', fontsize=font_size)
            a.set_title('y(x,t)', fontsize=font_size)
            # horizontal lines at t break points
            for t in range(self._num_time_blocks):
                tj = self._start_t + t * self.delta_t
                a.axvline(tj, color='black', lw=2)
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
            if show_plot:
                plt.show()
            else:
                plt.savefig(f'burgers_y_nfex_{n_x}_nfe_t_{n_t}_nb_{self._num_time_blocks}.png')
            plt.close()

            f, a = plt.subplots(figsize=(base_length * t_to_x_ratio, base_length))
            a.tick_params(axis='both', which='major', labelsize=font_size)
            surf = a.pcolormesh(np.array(all_t), np.array(all_x), np.array(u_list), cmap='jet')
            f.colorbar(surf, ax=a)
            a.set_ylabel('x', fontsize=font_size)
            a.set_xlabel('t', fontsize=font_size)
            a.set_title('u(x,t)', fontsize=font_size)
            # horizontal lines at t break points
            for t in range(self._num_time_blocks):
                tj = self._start_t + t * self.delta_t
                a.axvline(tj, color='black', lw=2)
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
            if show_plot:
                plt.show()
            else:
                plt.savefig(f'burgers_u_nfex_{n_x}_nfe_t_{n_t}_nb_{self._num_time_blocks}.png')
            plt.close()



            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # surf = ax.plot_surface(np.array(all_x), np.array(all_t), np.array(u_list), rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
            # surf.set_facecolor((0, 0, 0, 0))
            # ax.set_xlabel('x')
            # ax.set_ylabel('t')
            # ax.set_zlabel('u')
            # if show_plot:
            #     plt.show()
            # plt.close()


class BurgersInterface2D(parapint.interfaces.MPIStochasticSchurComplementInteriorPointInterface):
    def __init__(self, start_t, end_t, num_blocks, nfe_t, nfe_x):

        assert nfe_t % nfe_x == 0
        lx_per_lt = nfe_x / nfe_t
        assert (nfe_t * nfe_x) % num_blocks == 0
        nfes_per_block = (nfe_t * nfe_x) / num_blocks
        l_t = np.sqrt(nfes_per_block/lx_per_lt)
        assert l_t.is_integer()
        self.l_t = int(l_t)
        self.l_x = int(l_t * lx_per_lt)
        num_time_blocks = int(nfe_t / self.l_t)
        num_x_blocks = int(nfe_x / self.l_x)
        
        self.nfe_x = nfe_x
        self.dt = (end_t - start_t) / float(nfe_t)
        self.last_t = None
        self.scenarios = list(range(num_blocks))
        self.delta_t = (end_t - start_t) / num_time_blocks
        self.delta_x = 1/float(num_x_blocks)
        x_interface_indices = np.round(np.linspace(0, 1, num_x_blocks + 1), 3)[1:-1].tolist()
        t_interface_indices = np.round(np.linspace(0, 1, num_time_blocks + 1), 3)[1:-1].tolist()

        all_x_indices = np.round(np.linspace(0, 1, nfe_x + 1), 3)[1:-1].tolist()
        all_t_indices = np.round(np.linspace(0, 1, nfe_t + 1), 3)[1:].tolist()

        if rank == 0:
            first_stage_var_ids = [('y', (x, t)) for t in t_interface_indices for x in all_x_indices] + \
                                [('y', (x, t)) for t in all_t_indices for x in x_interface_indices] + \
                                [('u', (x, t)) for t in t_interface_indices for x in all_x_indices] + \
                                [('u', (x, t)) for t in all_t_indices for x in x_interface_indices]
            first_stage_var_ids = list(set(first_stage_var_ids))
            comm.bcast(first_stage_var_ids, root=0)
        else:
            first_stage_var_ids = comm.bcast(None, root=0)
        
        self._scenario_to_point = {}
        i=0
        for t in range(num_time_blocks):
            for x in range(num_x_blocks):
                self._scenario_to_point[i] = ((t * self.l_t)/nfe_t, (x * self.l_x)/nfe_x)
                i += 1
        
        super(BurgersInterface2D, self).__init__(scenarios=self.scenarios,
                                      nonanticipative_var_identifiers=first_stage_var_ids,
                                      comm=comm)
        #print(f'num first stage vars: {self._num_first_stage_vars}')


    def build_burgers_model(self, nfe_x=50, nfe_t=50, start_t=0, end_t=1, start_x=0, end_x=1,
                            add_init_t_conditions=True, add_init_x_conditions=True, add_final_x_conditions=True):
        dt = (end_t - start_t) / float(nfe_t)

        dx = (end_x - start_x) / float(nfe_x)

        m = pe.Block(concrete=True)
        m.omega = pe.Param(initialize=0.02)
        m.v = pe.Param(initialize=0.01)
        m.r = pe.Param(initialize=0)

        m.x = dae.ContinuousSet(bounds=(start_x, end_x))
        m.t = dae.ContinuousSet(bounds=(start_t, end_t))

        m.y = pe.Var(m.x, m.t)
        m.dydt = dae.DerivativeVar(m.y, wrt=m.t)
        m.dydx = dae.DerivativeVar(m.y, wrt=m.x)
        m.dydx2 = dae.DerivativeVar(m.y, wrt=(m.x, m.x))

        m.u = pe.Var(m.x, m.t)

        def _y_init_rule(m, x):
            return np.cos(2 * np.pi * x)
            # if x <= 0.5 * end_x:
            #     return 1
            #     #return np.cos(2 * np.pi * x)
            # return 0

        m.y0 = pe.Param(m.x, default=_y_init_rule)

        if add_final_x_conditions:
            def _upper_x_bound(m, t):
                return m.y[end_x, t] == 0
            m.upper_x_bound = pe.Constraint(m.t, rule=_upper_x_bound)

            def _upper_x_ubound(m, t):
                return m.u[end_x, t] == 0
            m.upper_x_ubound = pe.Constraint(m.t, rule=_upper_x_ubound)

        if add_init_x_conditions:
            def _lower_x_bound(m, t):
                return m.y[start_x, t] == 0
            m.lower_x_bound = pe.Constraint(m.t, rule=_lower_x_bound)

            def _lower_x_ubound(m, t):
                return m.u[start_x, t] == 0
            m.lower_x_ubound = pe.Constraint(m.t, rule=_lower_x_ubound)

        if add_init_t_conditions:
            def _lower_t_bound(m, x):
                if x == 0 or x == 1:
                    return pe.Constraint.Skip
                return m.y[x, start_t] == m.y0[x]
            m.lower_t_bound = pe.Constraint(m.x, rule=_lower_t_bound)

            def _lower_t_ubound(m, x):
                if x == 0 or x == 1:
                    return pe.Constraint.Skip
                return m.u[x, start_t] == 0
            m.lower_t_ubound = pe.Constraint(m.x, rule=_lower_t_ubound)


        # PDE
        def _pde(m, x, t):
            if t == start_t or x == 0 or x == 1:
                e = pe.Constraint.Skip
            else:
                # print(foo.last_t, t-dt, abs(foo.last_t - (t-dt)))
                # assert math.isclose(foo.last_t, t - dt, abs_tol=1e-6)
                e = m.dydt[x, t] - m.v * m.dydx2[x, t] + m.dydx[x, t] * m.y[x, t] == m.r + m.u[x, self.last_t]
            self.last_t = t
            return e

        m.pde = pe.Constraint(m.x, m.t, rule=_pde)

        # Discretize Model
        disc = pe.TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=nfe_t, wrt=m.t, scheme='BACKWARD')
        disc.apply_to(m, nfe=nfe_x, wrt=m.x, scheme='CENTRAL')

        # Solve control problem using Pyomo.DAE Integrals
        def _intX(m, x, t):
            return (m.y[x, t] - m.y0[x]) ** 2 + m.omega * m.u[x, t] ** 2

        m.intX = dae.Integral(m.x, m.t, wrt=m.x, rule=_intX)

        def _intT(m, t):
            return m.intX[t]

        m.intT = dae.Integral(m.t, wrt=m.t, rule=_intT)

        def _obj(m):
            e = 0.5 * m.intT
            for x in sorted(m.x):
                if x == 0 or x == 1:
                    pass
                else:
                    e += 0.5 * 0.5 * dx * dt * m.omega * m.u[x, start_t] ** 2
            return e

        m.obj = pe.Objective(rule=_obj)

        return m

    def build_model_for_scenario(self, scenario_identifier: int):
        start_t, start_x = self._scenario_to_point[scenario_identifier]

        end_t = start_t + self.delta_t
        end_x = start_x + self.delta_x

        if start_t == 0:
            add_init_t_conditions = True
        else:
            add_init_t_conditions = False
        
        if start_x == 0:
            add_init_x_conditions = True
        else:
            add_init_x_conditions = False
        
        if end_x == 1:
            add_final_x_conditions = True
        else:
            add_final_x_conditions = False
    
        nfe_t = self.l_t
        nfe_x = self.l_x
        m = self.build_burgers_model(nfe_x=nfe_x, nfe_t=nfe_t, start_t=start_t, end_t=end_t, start_x=start_x, end_x=end_x,
                                     add_init_t_conditions=add_init_t_conditions, add_init_x_conditions=add_init_x_conditions,
                                     add_final_x_conditions=add_final_x_conditions)

        
        start_var = {('y', (x, start_t)): m.y[x, start_t] for x in sorted(m.x) if x not in {0, 1}} | \
                    {('u', (x, start_t)): m.u[x, start_t] for x in sorted(m.x) if x not in {0, 1}}
        end_var = {('y', (x, end_t)): m.y[x, end_t] for x in sorted(m.x) if x not in {0, 1}} | \
                    {('u', (x, end_t)): m.u[x, end_t] for x in sorted(m.x) if x not in {0, 1}}
        down_var = {('y', (start_x, t)): m.y[start_x, t] for t in sorted(m.t) if t not in {0}} | \
                    {('u', (start_x, t)): m.u[start_x, t] for t in sorted(m.t) if t not in {0}}
        up_var = {('y', (end_x, t)): m.y[end_x, t] for t in sorted(m.t) if t not in {0}} | \
                    {('u', (end_x, t)): m.u[end_x, t] for t in sorted(m.t) if t not in {0}}
        
        first_stage_vars = {}
        if start_t != 0:
            #first_stage_vars.update(start_var)
            first_stage_vars = first_stage_vars | start_var
        if start_x != 0:
            #first_stage_vars.update(down_var)
            first_stage_vars = first_stage_vars | down_var
        if end_t != 1:
            #first_stage_vars.update(end_var)
            first_stage_vars = first_stage_vars | end_var
        if end_x != 1:
            #first_stage_vars.update(up_var)
            first_stage_vars = first_stage_vars | up_var
        
        return m, first_stage_vars

    def plot_results(self, show_plot=True):
        y_pts = list()
        u_pts = list()
        for block_ndx in self.local_block_indices:
            m = self.pyomo_model(block_ndx)
            for x in m.x:
                for t in m.t:
                    y_pts.append((x, t, m.y[x, t].value))
                    u_pts.append((x, t, m.u[x, t].value))
        y_pts = comm.allgather(y_pts)
        u_pts = comm.allgather(u_pts)
        if rank == 0:
            _tmp_y = list()
            _tmp_u = list()
            for i in y_pts:
                _tmp_y.extend(i)
            for i in u_pts:
                _tmp_u.extend(i)
            y_pts = _tmp_y
            u_pts = _tmp_u
            y_pts.sort(key=lambda x: x[0])
            y_pts.sort(key=lambda x: x[1])
            u_pts.sort(key=lambda x: x[0])
            u_pts.sort(key=lambda x: x[1])
            x_set = set()
            t_set = set()
            y_dict = dict()
            u_dict = dict()
            for x, t, y in y_pts:
                x_set.add(x)
                t_set.add(t)
                y_dict[x, t] = y
            for x, t, u in u_pts:
                u_dict[x, t] = u
            x_list = list(x_set)
            t_list = list(t_set)
            x_list.sort()
            t_list.sort()
            y_list = list()
            u_list = list()
            all_x = list()
            all_t = list()
            for x in x_list:
                tmp_y = list()
                tmp_u = list()
                tmp_x = list()
                tmp_t = list()
                for t in t_list:
                    tmp_y.append(y_dict[x, t])
                    tmp_u.append(u_dict[x, t])
                    tmp_x.append(x)
                    tmp_t.append(t)
                y_list.append(tmp_y)
                u_list.append(tmp_u)
                all_x.append(tmp_x)
                all_t.append(tmp_t)

            colors = cm.jet(y_list)
            rcount, ccount, _ = colors.shape
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(np.array(all_x), np.array(all_t), np.array(y_list), rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
            surf.set_facecolor((0, 0, 0, 0))
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('y')
            if show_plot:
                plt.show()
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(np.array(all_x), np.array(all_t), np.array(u_list), rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
            surf.set_facecolor((0, 0, 0, 0))
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('u')
            if show_plot:
                plt.show()
            plt.close()


def main(args, subproblem_solver_class, subproblem_solver_options):
    interface = BurgersInterface(start_t=0,
                                 end_t=1,
                                 num_time_blocks=args.nblocks,
                                 nfe_t=args.nfe_t,
                                 nfe_x=args.nfe_x)
    
    # interface = BurgersInterface2D(start_t=0,
    #                                 end_t=1,
    #                                 num_blocks=args.nblocks,
    #                                 nfe_t=args.nfe_t,
    #                                 nfe_x=args.nfe_x)

    from parapint.linalg.iterative.pcg import PcgOptions, LbfgsApproxOptions, LbfgsSamplingOptions
    pcg_options = parapint.linalg.PcgOptions()
    pcg_options.max_iter = 1000
    # pcg_options.lbfgs_approx_options = parapint.linalg.LbfgsApproxOptions()
    # pcg_options.lbfgs_approx_options.sampling = parapint.linalg.LbfgsSamplingOptions.uniform
    # pcg_options.lbfgs_approx_options.m = 5
    # pcg_options.lbfgs_approx_options.distributed = False
    precond_options = {}
    options = {'pcg': pcg_options, 'preconditioner': precond_options}

    # linear_solver = parapint.linalg.MPILbfgsImplicitSchurComplementLinearSolver(
    #     subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(args.nblocks)},
    #     options = options
    #      )

    linear_solver = parapint.linalg.MPISpiluImplicitSchurComplementLinearSolver(
        subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(args.nblocks)},
        options = options
         )

    # linear_solver = parapint.linalg.MPIASNoOverlapImplicitSchurComplementLinearSolver(
    #     subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(args.nblocks)},
    #     local_schur_complement_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(args.nblocks)},
    #     options=options
    # )

    # linear_solver = parapint.linalg.MPIASWithOverlapImplicitSchurComplementLinearSolver(
    #     subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(args.nblocks)},
    #     local_schur_complement_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(args.nblocks)},
    #     options=options
    # )

    # linear_solver = parapint.linalg.MPISchurComplementLinearSolver(
    #     subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(args.nblocks)},
    #     schur_complement_solver=subproblem_solver_class(**subproblem_solver_options))

    options = parapint.algorithms.IPOptions()
    options.linalg.solver = linear_solver
    status, history = parapint.algorithms.ip_solve(interface=interface, options=options)
    assert status == parapint.algorithms.InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()

    from parapint.utils import MPIHierarchicalTimer, TimerCollection
    timer_list: List[MPIHierarchicalTimer] = comm.gather(history.timer, root=0)
    if rank == 0:
        history.save('ip_log.csv')
        timer_collection = TimerCollection(timer_list)
        with open(f'timers.pickle', 'wb') as f:
            pickle.dump(timer_collection, f)

    if args.plot:
        interface.plot_results(show_plot=args.show_plot)

    return interface


if __name__ == '__main__':
    args = Args()
    args.parse_arguments()
    # cntl[1] is the MA27 pivot tolerance
    main(args=args,
         subproblem_solver_class=parapint.linalg.InteriorPointMA27Interface,
         subproblem_solver_options={'cntl_options': {1: 1e-6}})
