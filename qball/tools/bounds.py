
import numpy as np
from dipy.segment.mask import median_otsu

def compute_bounds(b_sph, data, c=1.0):
    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = b_sph.mdims['l_labels']
    assert(data.shape[-1] == l_labels)

    f = np.zeros((l_labels, n_image), order='C')
    fl = np.zeros((l_labels, n_image), order='C')
    fu = np.zeros((l_labels, n_image), order='C')

    data_clipped = np.clip(data, np.spacing(1), 1-np.spacing(1))
    loglog_data = np.log(-np.log(data_clipped))
    f[:] = loglog_data.reshape(-1, l_labels).T

    if len(data.shape) == 2:
        maskdata, mask = median_otsu(data[:,None,None,:], 3, 1, True,
                                     vol_idx=range(10, 50), dilate=2)
    elif len(data.shape) == 3:
        maskdata, mask = median_otsu(data[:,:,None,:], 3, 1, True,
                                     vol_idx=range(10, 50), dilate=2)
    elif len(data.shape) == 4:
        maskdata, mask = median_otsu(data[:,:,:,:], 3, 1, True,
                                     vol_idx=range(10, 50), dilate=2)

    n_samples = np.sum(np.logical_not(mask))
    print('n_samples = ', n_samples)
    
    samples = data[np.logical_not(mask.reshape(imagedims))]
    samples -= samples.min(0)

    assert(samples.shape == (n_samples,l_labels))
    
#    print(samples[:,0])
    
    noise_l = np.percentile(samples, c/2, axis=0)
    noise_u = np.percentile(samples, 1.0-c/2, axis=0)
    
    print('Noise in (',noise_l[0],noise_u[0],')')
    
#    print('Data')
#    print(data[10,10,:])
    
    data_l = data - noise_u
    data_u = data - noise_l
    
    assert((data_l <= data_u).all())

    data_l_clipped = np.clip(data_l, np.spacing(1), 1-np.spacing(1))
    data_u_clipped = np.clip(data_u, np.spacing(1), 1-np.spacing(1))

    fl = np.log(-np.log(data_u_clipped)).reshape(-1, l_labels).T
    fu = np.log(-np.log(data_l_clipped)).reshape(-1, l_labels).T
    
#    for i in range(n_image):
#        for k in range(l_labels):
#            if f[k,i] > 0:
#                fl[k,i] = f[k,i]*(1.0-c)
#                fu[k,i] = f[k,i]*(1.0+c)
#            else:
#                fl[k,i] = f[k,i]*(1.0+c)
#                fu[k,i] = f[k,i]*(1.0-c)

    assert((fl <= fu).all())

    fl_mean = np.einsum('ki,k->i', fl, b_sph.b)/(4*np.pi)
    fu_mean = np.einsum('ki,k->i', fu, b_sph.b)/(4*np.pi)

    fu -= fl_mean
    fl -= fu_mean

    return fl, fu