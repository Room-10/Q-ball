
from qball.tools import truncate, clip_hardi_data
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

    def prepare_stepsizes(self, step_bound, step_factor, steps):
        i = self.itervars
        c = self.constvars
        if steps == "precond":
            c['precond'] = True
            c['xtau'] = i['xk'].copy()
            c['ysigma'] = i['yk'].copy()
            logging.info("Determining diagonal preconditioners...")
            self.precond(c['xtau'], c['ysigma'])
        else:
            bnd = step_bound
            if step_bound is None:
                logging.info("Estimating optimal step bound...")
                op = lambda x,y: self.linop(x, y)
                opadj = lambda x,y: self.linop_adjoint(x, y)
                op_norm, itn = block_normest(i['xk'].copy(), i['yk'].copy(), op, opadj)
                # round (floor) to 3 significant digits
                bnd = truncate(1.0/op_norm**2, 3) # < 1/|K|^2
            if steps == "adaptive":
                c['adaptive'] = True
                c['eta'] = 0.95 # 0 < eta < 1
                c['Delta'] = 1.5 # > 1
                c['s'] = 255.0 # > 0
                i['alphak'] = 0.5
                i['sigmak'] = i['tauk'] = np.sqrt(bnd)
                i['res_pk'] = i['res_dk'] = 0.0
                logging.info("Adaptive steps: %f" % (bnd,))
            else:
                fact = step_factor # tau/sigma
                c['sigma'] = np.sqrt(bnd/fact)
                c['tau'] = bnd/c['sigma']
                logging.info("Constant steps: %f (%f | %f)"
                             % (bnd,c['sigma'],c['tau']))

    def prepare_gpu(self):
        from qball.tools.cuda import prepare_kernels
        from pycuda.elementwise import ElementwiseKernel
        from pycuda.reduction import ReductionKernel

        i = self.itervars
        c = self.constvars

        def dummy_linop_gpu(x, ygrad):
            x.get(ary=i['xk'].data)
            self.linop(i['xk'], i['yk'])
            ygrad.set(i['yk'].data)
        def dummy_linop_adjoint_gpu(xgrad, y):
            y.get(ary=i['yk'].data)
            self.linop_adjoint(i['xk'], i['yk'])
            xgrad.set(i['xk'].data)
        def dummy_prox_primal_gpu(x, tau):
            x.get(ary=i['xk'].data)
            if 'precond' in c:
                tau = c['xtau']
            self.prox_primal(i['xk'], tau)
            x.set(i['xk'].data)
        def dummy_prox_dual_gpu(y, sigma):
            y.get(ary=i['yk'].data)
            if 'precond' in c:
                tau = c['ysigma']
            self.prox_dual(i['yk'], sigma)
            y.set(i['yk'].data)

        p = ("*","[i]") if 'precond' in c else ("","")
        self.gpu_kernels = {
            'step_primal': ElementwiseKernel(
                "double *xkp1, double *xk, double %stau, double *xgradk" % p[0],
                "xkp1[i] = xk[i] - tau%s*xgradk[i]" % p[1]),
            'step_dual': ElementwiseKernel(
                "double *ykp1, double *yk, double %ssigma, double *ygradk" % p[0],
                "ykp1[i] = yk[i] + sigma%s*ygradk[i]" % p[1]),
            'overrelax': ElementwiseKernel(
                "double *ygradbk, double *ygradkp1, double *ygradk, double theta",
                "ygradbk[i] = (1 + theta)*ygradkp1[i] - theta*ygradk[i]"),
            'advance': ElementwiseKernel(
                "double *zk, double *zkp1, double *zgradk, double *zgradkp1",
                "zk[i] = zkp1[i]; zgradk[i] = zgradkp1[i]"),
            'residual': 0 if 'adaptive' not in c else ReductionKernel(
                np.float64, neutral="0", reduce_expr="a+b",
                map_expr="fabs((zk[i] - zkp1[i])/step - (zgradk[i] - zgradkp1[i]))",
                arguments="double step, double *zk, double *zkp1, "\
                         +"double *zgradk, double *zgradkp1"),
            # the following have to be defined by subclasses:
            'linop': dummy_linop_gpu,
            'linop_adjoint': dummy_linop_adjoint_gpu,
            'prox_primal': dummy_prox_primal_gpu,
            'prox_dual': dummy_prox_dual_gpu,
        }

        self.cuda_kernels, self.gpu_itervars, self.gpu_constvars = \
            prepare_kernels(self.cuda_files, self.cuda_templates,
                self.itervars, dict(self.constvars, **self.gpu_constvars),
                { 'x': i['xk'], 'y': i['yk'] })

    def iteration_step_gpu(self):
        c = self.constvars
        i = self.itervars
        gi = self.gpu_itervars
        gc = self.gpu_constvars
        gk = self.gpu_kernels

        if 'precond' in c:
            tau = gc['xtau']
            sigma = gc['ysigma']
        else:
            tau = i['tauk'] if 'adaptive' in c else c['tau']
            sigma = i['sigmak'] if 'adaptive' in c else c['sigma']

        # --- primals:
        gk['step_primal'](gi['xkp1'], gi['xk'], tau, gi['xgradk'])
        gk['prox_primal'](gi['xkp1'], tau)
        gk['linop'](gi['xkp1'], gi['ygradkp1'])
        gk['overrelax'](gi['ygradbk'], gi['ygradkp1'], gi['ygradk'], c['theta'])

        # --- duals:
        gk['step_dual'](gi['ykp1'], gi['yk'], sigma, gi['ygradbk'])
        gk['prox_dual'](gi['ykp1'], sigma)
        gk['linop_adjoint'](gi['xgradkp1'], gi['ykp1'])

        # --- step sizes:
        if 'adaptive' in c and i['alphak'] > 1e-10:
            i['res_pk'] = gk['residual'](i['tauk'],
                gi['xk'], gi['xkp1'], gi['xgradk'], gi['xgradkp1']).get()
            i['res_dk'] = gk['residual'](i['sigmak'],
                gi['yk'], gi['ykp1'], gi['ygradk'], gi['ygradkp1']).get()

            if i['res_pk'] > c['s']*i['res_dk']*c['Delta']:
                i['tauk'] *= 1.0/(1.0 - i['alphak'])
                i['sigmak'] *= (1.0 - i['alphak'])
                i['alphak'] *= c['eta']
            if i['res_pk'] < c['s']*i['res_dk']/c['Delta']:
                i['tauk'] *= (1.0 - i['alphak'])
                i['sigmak'] *= 1.0/(1.0 - i['alphak'])
                i['alphak'] *= c['eta']

        # --- update
        gk['advance'](gi['xk'], gi['xkp1'], gi['xgradk'], gi['xgradkp1'])
        gk['advance'](gi['yk'], gi['ykp1'], gi['ygradk'], gi['ygradkp1'])

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
        self.prepare_stepsizes(step_bound, step_factor, steps)

        if use_gpu:
            self.prepare_gpu()
            iteration_step = self.iteration_step_gpu
        else:
            iteration_step = self.iteration_step

        logging.info("Solving (steps<%d)..." % term_maxiter)

        interrupted = False
        with util.GracefulInterruptHandler() as interrupt_hdl:
            _iter = 0
            while _iter < term_maxiter:
                iteration_step()
                _iter += 1

                if interrupt_hdl.interrupted or _iter % granularity == 0:
                    if interrupt_hdl.interrupted:
                        interrupted = True
                        logging.info("Interrupt (SIGINT) at iter=%d" % _iter)

                    if use_gpu:
                        for n in ['xk','xgradk','yk','ygradk']:
                            self.gpu_itervars[n].get(ary=i[n].data)
                    obj_p, infeas_p = self.obj_primal(i['xk'], i['ygradk'])
                    obj_d, infeas_d = self.obj_dual(i['xgradk'], i['yk'])

                    # compute relative primal-dual gap
                    relgap = (obj_p - obj_d) / max(np.spacing(1), obj_d)

                    logging.info("#{:6d}: objp = {: 9.6g} ({: 9.6g}), " \
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
            'relgap': relgap,
            'iterations': _iter,
            'interrupted': interrupted
        }

    @property
    def state(self):
        return (self.itervars['xk'], self.itervars['yk'])

class PDHGModelHARDI(PDHGModel):
    "Base class for PDHG solvers for HARDI reconstruction."

    def __init__(self, data, model_params):
        PDHGModel.__init__(self)
        self.model_params = model_params
        self.data = data

        gtab = self.data['gtab']
        b_sph = self.data['b_sph']

        data = self.data['raw'][self.data['slice']]
        data = np.array(data, dtype=np.float64, copy=True)
        if not self.data['normed']:
            data.clip(1.0, out=data)
            b0 = data[...,(gtab.bvals == 0)].mean(-1)
            data /= b0[...,None]
        data = data[...,gtab.bvals > 0]

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
        c['lbd'] = self.model_params.get('lbd', 1.0)
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

        if 'constraint_u' in self.model_params:
            c['constraint_u'] = self.model_params['constraint_u']
        else:
            c['constraint_u'] = np.zeros((l_labels,) + imagedims, order='C')
            c['constraint_u'][:] = np.nan
        uconstrloc = np.any(np.logical_not(np.isnan(c['constraint_u'])), axis=0)
        c['uconstrloc'] = uconstrloc

        inpaintloc = self.model_params.get('inpaintloc', np.zeros(imagedims))
        c['inpaint_nloc'] = np.ascontiguousarray(np.logical_not(inpaintloc)).ravel()
        assert(c['inpaint_nloc'].shape == (n_image,))

        c['f'] = np.zeros((l_labels, n_image), order='C')
        clip_hardi_data(data)
        loglog_data = np.log(-np.log(data))
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

class PDHGModelHARDI_SHM(PDHGModelHARDI):
    "Base class for PDHG solvers for HARDI reconstr. using spherical harmonics."
    def __init__(self, *args):
        PDHGModelHARDI.__init__(self, *args)

        c = self.constvars

        sampling_matrix = self.model_params['sampling_matrix']
        c['Y'] = np.zeros(sampling_matrix.shape, order='C')
        c['Y'][:] = sampling_matrix
        l_shm = c['Y'].shape[1]
        c['l_shm'] = l_shm

        c['M'] = self.model_params['model_matrix']
        assert(c['M'].size == c['l_shm'])

        logging.info("HARDI PDHG setup ({l_labels} labels, {l_shm} shm; " \
                     "img: {imagedims}; lambda={lbd:.3g}) ready.".format(
                         lbd=c['lbd'],
                         l_labels=c['l_labels'],
                         l_shm=c['l_shm'],
                         imagedims="x".join(map(str,c['imagedims']))))
