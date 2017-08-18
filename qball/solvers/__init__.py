
from qball.sphere import load_sphere
from qball.tools import truncate
from qball.tools.diff import staggered_diff_avgskips
from qball.tools.blocks import block_normest
import qball.util as util

import numpy as np
from numpy.linalg import norm

import logging

class PDHGModel(object):
    "Base class for PDHG solvers."

    def __init__(self):
        self.itervars = { 'xk': None, 'yk': None }
        self.constvars = {}
        self.extravars = {}

    def prepare_gpu(self):
        from qball.tools.cuda import prepare_kernels
        self.gpu_constvars['x_size'] = self.itervars['xk'].data.size;
        self.gpu_constvars['y_size'] = self.itervars['yk'].data.size;
        self.cuda_kernels, self.cuda_vars = prepare_kernels(self.cuda_files,
            self.cuda_templates, dict(self.constvars, **self.gpu_constvars),
            self.itervars)

    def iteration_step(self):
        i = self.itervars
        c = self.constvars

        # --- primals:
        if 'precond' in c:
            i['xkp1'][:] = i['xk'][:] - c['xtau'][:]*i['xgradk'][:]
            self.prox_primal(i['xkp1'], c['xtau'])
        else:
            tau = i['tauk'] if 'adaptive' in c else c['tau']
            i['xkp1'][:] = i['xk'] - tau*i['xgradk']
            self.prox_primal(i['xkp1'], tau)
        self.linop(i['xkp1'], i['ygradkp1'])
        i['ygradbk'][:] = (1 + c['theta'])*i['ygradkp1'] - c['theta']*i['ygradk']

        # --- duals:
        if 'precond' in c:
            i['ykp1'][:] = i['yk'][:] + c['ysigma'][:]*i['ygradbk'][:]
            self.prox_dual(i['ykp1'], c['ysigma'])
        else:
            sigma = i['sigmak'] if 'adaptive' in c else c['sigma']
            i['ykp1'][:] = i['yk'] + sigma*i['ygradbk']
            self.prox_dual(i['ykp1'], sigma)
        self.linop_adjoint(i['xgradkp1'], i['ykp1'])

        # --- step sizes:
        if 'adaptive' in c and i['alphak'] > 1e-10:
            i['res_pk'] = norm((i['xk'][:] - i['xkp1'][:])/i['tauk']
                        - (i['xgradk'][:] - i['xgradkp1'][:]), ord=1)
            i['res_dk'] = norm((i['yk'][:] - i['ykp1'][:])/i['sigmak']
                        - (i['ygradk'][:] - i['ygradkp1'][:]), ord=1)

            if i['res_pk'] > c['s']*i['res_dk']*c['Delta']:
                i['tauk'] *= 1.0/(1.0 - i['alphak'])
                i['sigmak'] *= (1.0 - i['alphak'])
                i['alphak'] *= c['eta']
            if i['res_pk'] < c['s']*i['res_dk']/c['Delta']:
                i['tauk'] *= (1.0 - i['alphak'])
                i['sigmak'] *= 1.0/(1.0 - i['alphak'])
                i['alphak'] *= c['eta']

        # --- update
        i['xk'][:] = i['xkp1']
        i['xgradk'][:] = i['xgradkp1']
        i['yk'][:] = i['ykp1']
        i['ygradk'][:] = i['ygradkp1']


    def obj_primal(self, x, ygrad):
        pass

    def obj_dual(self, xgrad, y):
        pass

    def linop(self, x, ygrad):
        pass

    def linop_adjoint(self, xgrad, y):
        pass

    def prox_primal(self, x, tau):
        pass

    def prox_dual(self, y, sigma):
        pass

    def solve(self, continue_at=None, step_bound=None, step_factor=1.0,
                    term_relgap=1e-5, term_infeas=None, term_maxiter=int(1e7),
                    granularity=5000, use_gpu=True, steps="const"):
        i = self.itervars
        c = self.constvars

        if continue_at is not None:
            i['xk'][:], i['yk'][:] = continue_at

        i['xkp1'] = i['xk'].copy()
        i['xgradk'] = i['xk'].copy()
        i['xgradkp1'] = i['xk'].copy()
        i['ykp1'] = i['yk'].copy()
        i['ygradk'] = i['yk'].copy()
        i['ygradkp1'] = i['yk'].copy()
        i['ygradbk'] = i['yk'].copy()

        self.linop(i['xk'], i['ygradk'])
        self.linop_adjoint(i['xgradk'], i['yk'])

        if term_infeas is None:
            term_infeas = term_relgap

        obj_p = obj_d = infeas_p = infeas_d = relgap = 0.
        c['theta'] = 1.0 # overrelaxation

        if steps == "precond":
            c['precond'] = True
            c['xtau'] = i['xk'].copy()
            c['ysigma'] = i['yk'].copy()
            logging.info("Determining diagonal preconditioners...")
            self.precond(c['xtau'], c['ysigma'])
        else:
            if step_bound is None:
                logging.info("Estimating optimal step bound...")
                op = lambda x,y: self.linop(x, y)
                opadj = lambda x,y: self.linop_adjoint(x, y)
                op_norm, itn = block_normest(i['xk'].copy(), i['yk'].copy(), op, opadj)
                # round (floor) to 3 significant digits
                bnd = truncate(1.0/op_norm**2, 3) # < 1/|K|^2
            else:
                bnd = step_bound
            if steps == "adaptive":
                c['adaptive'] = True
                c['eta'] = 0.95 # 0 < eta < 1
                c['Delta'] = 1.5 # > 1
                c['s'] = 255.0 # > 0
                i['alphak'] = 0.5
                i['sigmak'] = i['tauk'] = np.sqrt(bnd)
                i['res_pk'] = i['res_dk'] = 0.0
            else:
                fact = step_factor # tau/sigma
                c['sigma'] = np.sqrt(bnd/fact)
                c['tau'] = bnd/c['sigma']
                logging.info("Constant steps: %f (%f | %f)" \
                                                % (bnd, c['sigma'], c['tau']))

        if use_gpu:
            from qball.tools.cuda import iterate_on_gpu
            self.prepare_gpu()

        logging.info("Solving (steps<%d)..." % term_maxiter)

        with util.GracefulInterruptHandler() as interrupt_hdl:
            _iter = 0
            while _iter < term_maxiter:
                if use_gpu:
                    iterations = iterate_on_gpu(self.cuda_kernels,
                        self.cuda_vars, granularity)
                    _iter += iterations
                    if iterations < granularity:
                        interrupt_hdl.handle(None, None)
                else:
                    self.iteration_step()
                    _iter += 1

                if interrupt_hdl.interrupted or _iter % granularity == 0:
                    obj_p, infeas_p = self.obj_primal(i['xk'], i['ygradk'])
                    obj_d, infeas_d = self.obj_dual(i['xgradk'], i['yk'])

                    # compute relative primal-dual gap
                    relgap = (obj_p - obj_d) / max(np.spacing(1), obj_d)

                    logging.debug("#{:6d}: objp = {: 9.6g} ({: 9.6g}), " \
                        "objd = {: 9.6g} ({: 9.6g}), " \
                        "gap = {: 9.6g}, " \
                        "relgap = {: 9.6g} ".format(
                        _iter, obj_p, infeas_p,
                        obj_d, infeas_d,
                        obj_p - obj_d,
                        relgap
                    ))

                    if relgap < term_relgap and max(infeas_p, infeas_d) < term_infeas:
                        break

                    if interrupt_hdl.interrupted:
                        break

        return {
            'objp': obj_p,
            'objd': obj_d,
            'infeasp': infeas_p,
            'infeasd': infeas_d,
            'relgap': relgap
        }

    @property
    def state(self):
        return (self.itervars['xk'], self.itervars['yk'])

class PDHGModelHARDI(PDHGModel):
    "Base class for PDHG solvers for HARDI reconstruction."

    def __init__(self, data, gtab, lbd=1.0):
        PDHGModel.__init__(self)
        b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
        b_sph = load_sphere(vecs=b_vecs)

        imagedims = data.shape[:-1]
        n_image = np.prod(imagedims)
        d_image = len(imagedims)
        l_labels = b_sph.mdims['l_labels']
        s_manifold = b_sph.mdims['s_manifold']
        m_gradients = b_sph.mdims['m_gradients']
        r_points = b_sph.mdims['r_points']
        assert(data.shape[-1] == l_labels)

        self.extravars['b_sph'] = b_sph

        c = self.constvars

        c['avgskips'] = staggered_diff_avgskips(imagedims)
        c['lbd'] = lbd
        c['b'] = b_sph.b
        c['A'] = b_sph.A
        c['B'] = b_sph.B
        c['P'] = b_sph.P
        c['b_precond'] = b_sph.b_precond
        c['imagedims'] = imagedims
        c['l_labels'] = l_labels
        c['n_image'] = n_image
        c['m_gradients'] = m_gradients
        c['s_manifold'] = s_manifold
        c['d_image'] = d_image
        c['r_points'] = r_points

        c['f'] = np.zeros((l_labels, n_image), order='C')
        data_clipped = np.clip(data, np.spacing(1), 1-np.spacing(1))
        loglog_data = np.log(-np.log(data_clipped))
        c['f'][:] = loglog_data.reshape(-1, l_labels).T
        f_mean = np.einsum('ki,k->i', c['f'], c['b'])/(4*np.pi)
        c['f'] -= f_mean

        skips = (1,)
        for d in reversed(imagedims[1:]):
            skips += (skips[-1]*d,)
        self.gpu_constvars = {
            'imagedims': np.array(imagedims, dtype=np.int64, order='C'),
            'navgskips': 1 << (d_image - 1),
            'skips': np.array(skips, dtype=np.int64, order='C'),
            'nd_skip': d_image*n_image,
            'ld_skip': d_image*l_labels,
            'sd_skip': s_manifold*d_image,
            'ss_skip': s_manifold*s_manifold,
            'sr_skip': s_manifold*r_points,
            'sm_skip': s_manifold*m_gradients,
            'msd_skip': m_gradients*s_manifold*d_image,
            'ndl_skip': n_image*d_image*l_labels,
        }
